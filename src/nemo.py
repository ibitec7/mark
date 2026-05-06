from nemo.core.classes import ModelPT
import time
import math

from omegaconf import OmegaConf, DictConfig
from transformers.configuration_utils import PretrainedConfig
import torch
import os
import json
from transformers.modeling_outputs import MaskedLMOutput

from typing import Union, Optional
from types import SimpleNamespace
import torch.nn as nn
import torch.nn.functional as F

import wandb
import logging
from lightning import Trainer

from .hydra_model import HydraForMaskedLM, HydraMaskedLMOutput
from .utils import load_config, TrainingMetrics, record_batch, arrow_dataloader, get_seq_idx, enable_cudnn_optimizations, log_setup
from .train import masking_process, sample_timestep, context_adaptive_reweight
from .scheduler import linear_warmup_cosine_decay, linear_warmup_polynomial_decay, inverse_sqrt_warmup

logger = log_setup("NemoLogger", os.path.join("logs", "nemo.log"), logging.INFO)

def get_flattened_ddp_modules(module: nn.Module):
    """
    Recursively collect all submodules that have a `.module` attribute.
    Ignores modules without `.module`.

    Args:
        module (nn.Module): Parent module to traverse.

    Returns:
        List[nn.Module]: Flattened list of modules with `.module`.
    """
    ddp_modules = []

    # If this module has .module, add it and stop recursion
    if hasattr(module, "module"):
        ddp_modules.append(module)
        return ddp_modules

    # Otherwise, recurse into children
    for child in module.children():
        ddp_modules.extend(get_flattened_ddp_modules(child))

    return ddp_modules


CONFIG = load_config(os.path.join("./configs/training_config.yaml"))

DATA_DIR = "./data/wikitext-103-v1"

class Benchmarks():
    def __init__(self, raw_nll=None, raw_ppl=None, raw_bpb=None, weighted_nll=None, weighted_ppl=None, weighted_bpb=None):
        self.raw_nll = raw_nll
        self.raw_ppl = raw_ppl
        self.raw_bpb = raw_bpb
        self.weighted_nll = weighted_nll
        self.weighted_ppl = weighted_ppl
        self.weighted_bpb = weighted_bpb
        self.mdlm_masked_nll = None
        self.mdlm_masked_ppl = None
        self.mdlm_masked_bpb = None
        self.mdlm_elbo_nll = None
        self.mdlm_elbo_ppl = None
        self.mdlm_elbo_bpb = None
        self.raw_nll_sum = None
        self.raw_weight_sum = None
        self.weighted_nll_sum = None
        self.weighted_weight_sum = None
        self.mdlm_masked_nll_sum = None
        self.mdlm_masked_token_count = None
        self.mdlm_elbo_nll_sum = None
        self.mdlm_elbo_token_count = None
        self.byte_count = None

class NemoForMaskedLM(ModelPT):
    def __init__(self, config: DictConfig, tracker: wandb.Run = None, trainer: Trainer = None, model: HydraForMaskedLM = None, inference: bool = False):

        if trainer is not None and not isinstance(trainer, Trainer):
            raise TypeError("trainer must either be a pytorch_lightning.Trainer or None.")
        
        super().__init__(config, trainer)

        self.save_hyperparameters(OmegaConf.to_container(config, resolve=True))

        if "hydra_config_path" in config and config.hydra_config_path:
            hf_cfg: PretrainedConfig = load_config(config.hydra_config_path, pretrained_config=True)
        elif "hydra_config" in config and isinstance(config.hydra_config, dict):
            hf_cfg: PretrainedConfig = PretrainedConfig(**config)
        elif config is None:
            raise ValueError("Provide either model.hydra_config_path or model.hydra_config dict in cfg.")

        self.inner = HydraForMaskedLM(config=hf_cfg) if model is None else model

        self._train_dl = None
        self._val_dl = None
        self._scheduler = None

        self._logged_unused = False

        self.module = self.inner

        self._cfg = config
        self.config = SimpleNamespace(**OmegaConf.to_container(config, resolve=True))

        self.val_losses = []

        self.data_dir = os.path.join("data", "training")

        self.optim = None

        self.recorder = TrainingMetrics(dir_name=self.data_dir, filename="hydra_train_metrics.parquet")

        self.train_dir = config.get("train_data_dir", "./data/train_shards")
        self.val_dir = config.get("val_data_dir", "./data/val_shards")

        # Defer W&B initialization to on_train_start() to avoid side effects on import
        self.tracker = tracker
        self._inference_mode = inference

        self.intervals = config.get("intervals", 10)
        self.multi_shot = config.get("multi_shot", False)

        self.gradient_accumulation = config.get("gradient_accumulation", False)

        self.cart = config.get("cart", False)
        self.cart_enabled = self.cart

        self.benchmark = config.get("benchmark", False)
        if self.benchmark:
            self.benchmark_track = Benchmarks(
                raw_nll=None,
                raw_ppl=None,
                raw_bpb=None,
                weighted_nll=None,
                weighted_ppl=None,
                weighted_bpb=None,
            )
            self._benchmark_batch_count: int = 0
        self.benchmark_path = config.get("benchmark_path", "./data/benchmark.json")

        if self.cart:
            self.cart_p = config.get("cart_p", 0.45)
            self.cart_scale = config.get("cart_scale", 1.0)
            self.cart_dtype = config.get("cart_dtype", "bf16")
            self.cart_vram_safety_ratio = float(config.get("cart_vram_safety_ratio", 0.6))

        # IMPORTANT: plain attribute, not DDP buffer
        self.cart_matrix_cache: Optional[torch.Tensor] = None

        if self.multi_shot:
            self.automatic_optimization = False
        else:
            self.automatic_optimization = True

    def on_train_start(self) -> None:
        """Initialize W&B tracker at training start (lazy initialization)."""
        if not self._inference_mode and self.tracker is None:
            try:
                self.tracker = wandb.init(
                    project="hydra-masked-lm",
                    config=OmegaConf.to_container(self._cfg, resolve=True),
                    reinit=True
                )
                logger.info("W&B tracker initialized at training start")
            except Exception as e:
                logger.warning(f"W&B initialization failed: {e}. Training will proceed without W&B tracking.")
                self.tracker = None

    def _cart_dtype_from_config(self) -> torch.dtype:
        if not self.cart:
            return torch.float32

        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        return mapping.get(str(self.cart_dtype).lower(), torch.float32)

    def _estimate_cart_cache_bytes(self, seq_len: int, dtype: torch.dtype) -> int:
        element_bytes = torch.tensor([], dtype=dtype).element_size()
        return int(seq_len * seq_len * element_bytes)

    def _initialize_cart_cache(self, device: torch.device):
        max_len = int(getattr(self.inner.config, "max_position_embeddings", 4096))
        dtype = self._cart_dtype_from_config()
        required_bytes = self._estimate_cart_cache_bytes(max_len, dtype)

        if device.type == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info(device=device)
            allowed_bytes = int(free_bytes * self.cart_vram_safety_ratio)
            if required_bytes > allowed_bytes:
                print(
                    f"[WARNING] CART cache disabled: requires ~{required_bytes / (1024 ** 2):.1f} MiB, "
                    f"but only ~{free_bytes / (1024 ** 2):.1f} MiB free on {device}."
                )
                self.cart_enabled = False
                self.cart_matrix_cache = torch.empty(0, device=device)
                return

        print(
            f"Initializing CART weight matrix cache (len={max_len}, dtype={dtype}, "
            f"size~{required_bytes / (1024 ** 2):.1f} MiB) on device: {device}"
        )
        cpu_matrix = context_adaptive_reweight(max_len, cart_p=self.cart_p).to(dtype=dtype)
        self.cart_matrix_cache = cpu_matrix.to(device, non_blocking=False)
        self.cart_enabled = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.BoolTensor = None,
        seq_idx: torch.Tensor = None,
        current_timestep: int = 0,
        total_timestep: int = 0,
        labels: torch.Tensor = None,
        p_mask: torch.Tensor = None,
        masked_tokens_mask: torch.Tensor = None,
        cart_weights: torch.Tensor = None,
    ) -> MaskedLMOutput:
        
        return self.inner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            seq_idx=seq_idx,
            current_timestep=current_timestep,
            total_timestep=total_timestep,
            labels=labels,
            p_mask=p_mask,
            masked_tokens_mask=masked_tokens_mask,
            cart_weights=cart_weights,
            benchmark=self.benchmark,
        )

    def setup_training_data(self, train_data_config: DictConfig=None):

        self._train_dl = arrow_dataloader(
            data_dir=self.train_dir,
            split="train",
            batch_size=self._cfg.get("batch_size", 8),
            num_workers=self._cfg.get("num_workers", 4),
            keep_in_memory=True,
            persistent_workers=True
        )

    def setup_validation_data(self, val_data_config: DictConfig=None):

        self._val_dl = arrow_dataloader(
            data_dir=self.val_dir,
            split="validation",
            batch_size=self._cfg.get("batch_size", 8),
            num_workers=self._cfg.get("num_workers", 4),
            keep_in_memory=True,
            persistent_workers=True
        )

    @classmethod
    def list_available_models(cls) -> list[str]:
        return ["HydraForMaskedLM", "HydraBase"]

    def setup_test_data(self):
        pass

    def train_dataloader(self):
        if self._train_dl is None:
            self.setup_training_data()

        return self._train_dl
    
    def val_dataloader(self):
        if self._val_dl is None:
            self.setup_validation_data()

        return self._val_dl
    
    def _setup_precision(self):
        """
        Configure mixed precision settings for better performance.
        TensorFloat32 allows using fast matrix operations while maintaining accuracy.
        """
        try:
            torch.set_float32_matmul_precision('high')
            print("[INFO] TensorFloat32 precision enabled for higher performance")
            return True
        except Exception as e:
            print(f"[WARNING] Failed to set TensorFloat32 precision: {e}")
            return False

    # [WIP] Configure safety for DDP hooks to prevent race conditions and deadlocks.
    def setup(self, stage: Optional[str] = None):
        super().setup(stage)

        device = torch.device("cpu")
        if torch.cuda.is_available() and next(self.parameters()).is_cuda:
            device = next(self.parameters()).device

        if self.inner.hydra.encoder.mark is not None and self.inner.hydra.encoder.mark.streams is None:
            print("Initializing MARK adapter CUDA streams on device: ", device)
            self.inner.hydra.encoder.mark.streams = [
                torch.cuda.Stream(device=device) for _ in range(self.inner.hydra.encoder.mark.num_adapters)
            ]

            self.inner.hydra.encoder.mark._streams_initialized = True
        
        # Setup precision optimizations (torch.compile disabled - incompatible with Triton kernels)
        self._setup_precision()
        enable_cudnn_optimizations()

        # Precompute CART matrix cache with VRAM safety checks.
        if self.cart:
            self._initialize_cart_cache(device)

    def on_after_backward(self):
        """
        This hook serves as my guarantee that there are no unused parameters in the loss computation for the backward pass.

        DDP has strict dynamic graph checks that fail the implementation of custom CUDA streams in the setup, to prevent this I enable static_graph
        and I have logged and confirmed that there are no unused parameters in the model during training, and using static_graph=True is safe.
        ."""
        if not self._logged_unused:
            unused = []
            for name, param in self.named_parameters():
                if param.requires_grad and (param.grad is None or not param.grad.any()):
                    unused.append(name)
            
            if unused:
                print(f"[WARNING]: Unused parameters ({len(unused)}):")
                for name in unused[:20]:  # print first 20
                    print(f"  - {name}")
                if len(unused) > 20:
                    print(f"  ... and {len(unused) - 20} more")
            else:
                print("[INFO]: No unused parameters detected.")
            
            self._logged_unused = True
        
        super().on_after_backward()

    # def on_before_optimizer_step(self, optimizer):
    #     "Synchronize the CUDA streams before calling the optimizer step."

    #     if (
    #         hasattr(self.inner.hydra.encoder, "mark") and
    #         self.inner.hydra.encoder.mark is not None and
    #         hasattr(self.inner.hydra.encoder.mark, "streams") and
    #         self.inner.hydra.encoder.mark.streams is not None
    #     ):
    #         current_stream = torch.cuda.current_stream()
    #         for stream in self.inner.hydra.encoder.mark.streams:
    #             if stream is not None:
    #                 current_stream.wait_stream(stream)
            

    #     super().on_before_optimizer_step(optimizer)
    
    # Lightning Module Methods

    def _prepare_batch_inputs(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict:

        batch_tokens = batch["input_ids"] if isinstance(batch, dict) else batch[0]
        # batch_tokens = batch_tokens[:, :64] if batch_tokens.dim() == 2 else batch_tokens
        # batch_mask = batch["attention_mask"] if isinstance(batch, dict) else batch[1]

        if batch_tokens.is_cuda == False:
            batch_tokens: torch.Tensor = batch_tokens.to(device=self.device, non_blocking=True)
            # batch_mask: torch.Tensor = batch_mask.to(device=self.device, non_blocking=True)
        
        ### ============================================================================================================= ###

        # 1. Sample both current_step and total_steps only on rank 0
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            c_step, t_steps = sample_timestep(device=batch_tokens.device) 
            step_tensor = torch.stack((c_step.squeeze(0), t_steps.squeeze(0))).to(dtype=torch.long, device=batch_tokens.device)
        else:
            step_tensor = torch.zeros(2, dtype=torch.long, device=batch_tokens.device)

        # 2. Broadcast the sampled sequence to all other DDP processes
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.broadcast(step_tensor, src=0)
            
        # 3. Unpack the synchronized values
        current_step = step_tensor[0].item()
        total_steps = step_tensor[1].item()

        # 4. Calculate ratio predictably across all nodes
        ratio: float = 1.0 - (float(current_step) / float(max(total_steps, 1)))  # Masking ratio based on current timestep

        # sep_id = 102
        # new_seq_pos = F.pad((batch_tokens[:, :-1] == sep_id).to(torch.int32), (1, 0))
        # seq_idx: torch.Tensor = torch.cumsum(new_seq_pos, dim=1).to(torch.int32)

        seq_idx = get_seq_idx(batch_tokens, cls_id=101, pad_id=0)

        corrupt_tokens, masked_indices, p_mask = masking_process(input_ids=batch_tokens, mask_ratio=ratio, attn_mask=None)

        return dict(
            corrupt_tokens=corrupt_tokens,
            labels=batch_tokens,
            p_mask=p_mask,
            seq_idx=seq_idx,
            attention_mask=(batch_tokens != 0),
            current_step=current_step,
            total_steps=total_steps,
            masked_indices=masked_indices
        )

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int=None):

        # if not self.automatic_optimization:
        #     opt: LightningOptimizer = self.optimizers()
        #     sch = self.lr_schedulers()

        start_time = time.time()

        # Empty the cache every 10 iterations
        # if batch_idx % 10 == 0:
            # torch.cuda.empty_cache()

        items: dict = self._prepare_batch_inputs(batch)

        ## ///            CART Weighting Implementation          ///
        ### ============================================================================================================= ###

        weights_for_loss = None
        if self.cart_enabled and self.cart_matrix_cache is not None and self.cart_matrix_cache.numel() > 0:
            with torch.no_grad():
                if self.cart_matrix_cache.device != items["masked_indices"].device:
                    self.cart_matrix_cache = self.cart_matrix_cache.to(items["masked_indices"].device)

                seq_len = items["corrupt_tokens"].shape[-1]
                _weight_matrix = self.cart_matrix_cache[:seq_len, :seq_len]
                non_mask: torch.Tensor = ~items["masked_indices"].to(
                    items["masked_indices"].device
                )  # loss_mask indicates where is mask
                
                cart_weights = (
                    non_mask.type_as(_weight_matrix)
                    .matmul(_weight_matrix)
                    .masked_fill(non_mask, 0)
                )

                # Now flatten and select masked positions later exactly like before:
                masked_idx_flat: torch.Tensor = items["masked_indices"].flatten()
                weights_for_loss: torch.Tensor = (cart_weights.flatten()[masked_idx_flat] + 1e-8).detach()  # avoid zero weights
        ### ============================================================================================================= ###

        ## ///       Single-shot Training Implementation     ///
        ### ============================================================================================================= ###
        
        outputs: MaskedLMOutput = self.forward(
            input_ids=items["corrupt_tokens"],
            attention_mask=items["attention_mask"],
            seq_idx=items["seq_idx"],
            current_timestep=items["current_step"],
            total_timestep=items["total_steps"],
            labels=items["labels"],
            masked_tokens_mask=items["masked_indices"],
            p_mask=items["p_mask"],
            cart_weights=weights_for_loss if self.cart_enabled and self.cart_matrix_cache.numel() > 0 else None,
        )

        loss: torch.Tensor = outputs.loss

        assert loss is not None

        if self._trainer is not None:
            self.log("train_loss", loss.item(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=items["corrupt_tokens"].shape[0])

        if items["current_step"] % 20 == 0:
            data: dict = record_batch(
                batch_id=batch_idx,
                epoch=self.current_epoch,
                loss=loss.item(),
                logits=outputs.logits.detach().cpu(),
                labels=items["labels"].detach().cpu(),
                masked_indices=items["masked_indices"].detach().cpu(),
                current_step=int(items["current_step"]),
                total_steps=int(items["total_steps"]),
                model=self.inner,
                optimizer=self.optim,
            )

            end_time = time.time()
            step_duration = end_time - start_time

            data["step_duration"] = step_duration

            self.tracker.log(data) if self.tracker is not None else None

        del items, batch, outputs

        return loss

        ### ============================================================================================================= ###
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        items: dict = self._prepare_batch_inputs(batch)

        ## ///            CART Weighting Implementation          ///
        ### ============================================================================================================= ###

        weights_for_loss = None
        if self.cart_enabled and self.cart_matrix_cache is not None and self.cart_matrix_cache.numel() > 0:
            with torch.no_grad():
                if self.cart_matrix_cache.device != items["masked_indices"].device:
                    self.cart_matrix_cache = self.cart_matrix_cache.to(items["masked_indices"].device)

                seq_len = items["corrupt_tokens"].shape[-1]
                _weight_matrix = self.cart_matrix_cache[:seq_len, :seq_len]
                non_mask: torch.Tensor = ~items["masked_indices"].to(
                    items["masked_indices"].device
                )  # loss_mask indicates where is mask
                
                cart_weights = (
                    non_mask.type_as(_weight_matrix)
                    .matmul(_weight_matrix)
                    .masked_fill(non_mask, 0)
                )

                # Now flatten and select masked positions later exactly like before:
                masked_idx_flat: torch.Tensor = items["masked_indices"].flatten()
                weights_for_loss: torch.Tensor = (cart_weights.flatten()[masked_idx_flat] + 1e-8).detach()  # avoid zero weights
        ### ============================================================================================================= ###

        outputs: MaskedLMOutput = self.forward(
            input_ids=items["corrupt_tokens"],
            current_timestep=items["current_step"],
            total_timestep=items["total_steps"],
            labels=items["labels"],
            p_mask=items["p_mask"],
            masked_tokens_mask=items["masked_indices"],
            cart_weights=weights_for_loss if self.cart_enabled and self.cart_matrix_cache.numel() > 0 else None,
        )

        if not self.benchmark:
            loss: torch.Tensor = outputs.loss.item()
            self.val_losses.append(loss)

            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=items["corrupt_tokens"].shape[0])

            if self.tracker is not None:
                self.tracker.log({
                    "val_loss": loss,
                    "epoch": self.current_epoch,
                    "batch_id": batch_idx,
                })
        elif self.benchmark:
            loss_value = outputs.weighted_nll.item() if outputs.weighted_nll is not None else 0.0
            loss: torch.Tensor = torch.tensor(loss_value, device=self.device)
            self.val_losses.append(loss)

            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=items["corrupt_tokens"].shape[0])

            if self.tracker is not None:
                self.tracker.log({
                    "val_loss": loss,
                    "epoch": self.current_epoch,
                    "batch_id": batch_idx,
                    "raw_nll": outputs.raw_nll.item() if isinstance(outputs.raw_nll, torch.Tensor) else outputs.raw_nll,
                    "raw_ppl": outputs.raw_ppl.item() if isinstance(outputs.raw_ppl, torch.Tensor) else outputs.raw_ppl,
                    "raw_bpb": outputs.raw_bpb.item() if isinstance(outputs.raw_bpb, torch.Tensor) else outputs.raw_bpb,
                    "weighted_nll": outputs.weighted_nll.item() if isinstance(outputs.weighted_nll, torch.Tensor) else outputs.weighted_nll,
                    "weighted_ppl": outputs.weighted_ppl.item() if isinstance(outputs.weighted_ppl, torch.Tensor) else outputs.weighted_ppl,
                    "weighted_bpb": outputs.weighted_bpb.item() if isinstance(outputs.weighted_bpb, torch.Tensor) else outputs.weighted_bpb,
                    "mdlm_masked_nll": outputs.mdlm_masked_nll.item() if isinstance(outputs.mdlm_masked_nll, torch.Tensor) else outputs.mdlm_masked_nll,
                    "mdlm_masked_ppl": outputs.mdlm_masked_ppl.item() if isinstance(outputs.mdlm_masked_ppl, torch.Tensor) else outputs.mdlm_masked_ppl,
                    "mdlm_masked_bpb": outputs.mdlm_masked_bpb.item() if isinstance(outputs.mdlm_masked_bpb, torch.Tensor) else outputs.mdlm_masked_bpb,
                    "mdlm_elbo_nll": outputs.mdlm_elbo_nll.item() if isinstance(outputs.mdlm_elbo_nll, torch.Tensor) else outputs.mdlm_elbo_nll,
                    "mdlm_elbo_ppl": outputs.mdlm_elbo_ppl.item() if isinstance(outputs.mdlm_elbo_ppl, torch.Tensor) else outputs.mdlm_elbo_ppl,
                    "mdlm_elbo_bpb": outputs.mdlm_elbo_bpb.item() if isinstance(outputs.mdlm_elbo_bpb, torch.Tensor) else outputs.mdlm_elbo_bpb,
                })

            self._benchmark_batch_count += 1
            self._update_benchmark_track(outputs)

        # torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        if self.benchmark:
            self._finalize_benchmark_track()

        if self.benchmark and self.benchmark_track.weighted_nll is not None:
            avg_val = self.benchmark_track.weighted_nll
            self.log("val_loss_epoch", avg_val, sync_dist=True, prog_bar=True)
        elif self.val_losses:
            avg_val = torch.tensor(self.val_losses, device=self.device).mean()
            self.log("val_loss_epoch", avg_val, sync_dist=True, prog_bar=True)

        self.val_losses.clear()

        if self.benchmark:
            benchmark_dict = {}
            for attr in (
                "raw_nll",
                "raw_ppl",
                "raw_bpb",
                "weighted_nll",
                "weighted_ppl",
                "weighted_bpb",
                "mdlm_masked_nll",
                "mdlm_masked_ppl",
                "mdlm_masked_bpb",
                "mdlm_elbo_nll",
                "mdlm_elbo_ppl",
                "mdlm_elbo_bpb",
            ):
                val = getattr(self.benchmark_track, attr, None)
                benchmark_dict[attr] = val.item() if isinstance(val, torch.Tensor) else val

            os.makedirs(os.path.dirname(self.benchmark_path), exist_ok=True)
            with open(self.benchmark_path, "w") as f:
                json.dump(benchmark_dict, f, indent=2)

        super().on_validation_epoch_end()

    def reset_benchmark_state(self):
        """Reset benchmark tracking state between dataset evaluations."""
        self.val_losses.clear()
        if self.benchmark:
            self._benchmark_batch_count = 0
            for attr in (
                "raw_nll",
                "raw_ppl",
                "raw_bpb",
                "weighted_nll",
                "weighted_ppl",
                "weighted_bpb",
                "mdlm_masked_nll",
                "mdlm_masked_ppl",
                "mdlm_masked_bpb",
                "mdlm_elbo_nll",
                "mdlm_elbo_ppl",
                "mdlm_elbo_bpb",
                "raw_nll_sum",
                "raw_weight_sum",
                "weighted_nll_sum",
                "weighted_weight_sum",
                "mdlm_masked_nll_sum",
                "mdlm_masked_token_count",
                "mdlm_elbo_nll_sum",
                "mdlm_elbo_token_count",
                "byte_count",
            ):
                setattr(self.benchmark_track, attr, None)

    def _update_benchmark_track(self, outputs: HydraMaskedLMOutput):
        for attr in (
            "raw_nll_sum",
            "raw_weight_sum",
            "weighted_nll_sum",
            "weighted_weight_sum",
            "mdlm_masked_nll_sum",
            "mdlm_masked_token_count",
            "mdlm_elbo_nll_sum",
            "mdlm_elbo_token_count",
            "byte_count",
        ):
            new = getattr(outputs, attr, None)
            if new is None:
                continue

            if isinstance(new, torch.Tensor):
                new = new.detach().to(device=self.device, dtype=torch.float64)
            else:
                new = torch.tensor(new, device=self.device, dtype=torch.float64)

            old = getattr(self.benchmark_track, attr)
            setattr(self.benchmark_track, attr, new if old is None else old + new)

    def _finalize_benchmark_track(self):
        max_exp = 709.78
        byte_count = self.benchmark_track.byte_count
        byte_den = byte_count.clamp_min(1.0) if byte_count is not None else None

        raw_nll_sum = self.benchmark_track.raw_nll_sum
        raw_weight_sum = self.benchmark_track.raw_weight_sum
        if raw_nll_sum is not None and raw_weight_sum is not None:
            raw_den = raw_weight_sum.clamp_min(1.0)
            raw_nll = raw_nll_sum / raw_den
            self.benchmark_track.raw_nll = raw_nll
            self.benchmark_track.raw_ppl = torch.exp(torch.clamp(raw_nll, max=raw_nll.new_tensor(max_exp)))
            if byte_den is not None:
                self.benchmark_track.raw_bpb = raw_nll_sum / math.log(2) / byte_den

        weighted_nll_sum = self.benchmark_track.weighted_nll_sum
        weighted_weight_sum = self.benchmark_track.weighted_weight_sum
        if weighted_nll_sum is not None and weighted_weight_sum is not None:
            weighted_den = weighted_weight_sum.clamp_min(1.0)
            weighted_nll = weighted_nll_sum / weighted_den
            self.benchmark_track.weighted_nll = weighted_nll
            self.benchmark_track.weighted_ppl = torch.exp(torch.clamp(weighted_nll, max=weighted_nll.new_tensor(max_exp)))
            if byte_den is not None:
                self.benchmark_track.weighted_bpb = weighted_nll_sum / math.log(2) / byte_den

        mdlm_masked_nll_sum = self.benchmark_track.mdlm_masked_nll_sum
        mdlm_masked_token_count = self.benchmark_track.mdlm_masked_token_count
        if mdlm_masked_nll_sum is not None and mdlm_masked_token_count is not None:
            mdlm_den = mdlm_masked_token_count.clamp_min(1.0)
            mdlm_masked_nll = mdlm_masked_nll_sum / mdlm_den
            self.benchmark_track.mdlm_masked_nll = mdlm_masked_nll
            self.benchmark_track.mdlm_masked_ppl = torch.exp(torch.clamp(mdlm_masked_nll, max=mdlm_masked_nll.new_tensor(max_exp)))
            if byte_den is not None:
                self.benchmark_track.mdlm_masked_bpb = mdlm_masked_nll_sum / math.log(2) / byte_den

        mdlm_elbo_nll_sum = self.benchmark_track.mdlm_elbo_nll_sum
        mdlm_elbo_token_count = self.benchmark_track.mdlm_elbo_token_count
        if mdlm_elbo_nll_sum is not None and mdlm_elbo_token_count is not None:
            mdlm_elbo_den = mdlm_elbo_token_count.clamp_min(1.0)
            mdlm_elbo_nll = mdlm_elbo_nll_sum / mdlm_elbo_den
            self.benchmark_track.mdlm_elbo_nll = mdlm_elbo_nll
            self.benchmark_track.mdlm_elbo_ppl = torch.exp(torch.clamp(mdlm_elbo_nll, max=mdlm_elbo_nll.new_tensor(max_exp)))
            if byte_den is not None:
                self.benchmark_track.mdlm_elbo_bpb = mdlm_elbo_nll_sum / math.log(2) / byte_den

    def setup_optimization(self) -> Union[dict, torch.optim.Optimizer]:
        lr_mark: float = float(self._cfg.get("learning_rate_mark", 6e-4))
        lr_hydra: float = float(self._cfg.get("learning_rate_hydra", 3e-5))
        lr_cls: float = float(self._cfg.get("learning_rate_cls", 3e-5))

        sched_cfg: dict = self._cfg.get("lr_scheduler", {})
        stage = self._cfg.get("stage", 1)

        layer_indices_cfg = self._cfg.get("unfrozen_layer_indices", None)

        unfrozen_layer_indices = None
        if layer_indices_cfg is not None:
            unfrozen_layer_indices = list(set(int(i) for i in layer_indices_cfg))
        else:
            unfrozen_ratio = self._cfg.get("unfrozen_ratio", None)
            if unfrozen_ratio is not None:
                unfrozen_layer_indices = [i for i in range(round((1.0 - unfrozen_ratio) * 23), 23)]

        mark_names = [name for name, _ in self.named_parameters()
                      if "mark" in name or "embeddings.cnn" in name or "embeddings.time" in name
                      or "embeddings.table" in name or "embeddings.mask" in name]
        mark_params = [param for name, param in self.named_parameters() if name in mark_names]

        cls_names = [name for name, _ in self.named_parameters() if "cls" in name]
        cls_params = [param for name, param in self.named_parameters() if name in cls_names]

        all_names = [name for name, _ in self.named_parameters()]
        hydra_all_names = [n for n in all_names if n not in mark_names and n not in cls_names]

        def in_target_layer(name: str) -> bool:
            if unfrozen_layer_indices is None:
                return True
            
            for idx in unfrozen_layer_indices:
                if idx < 0 or idx > 22:
                    continue
                if f"encoder.layer.{idx}." in name:
                    return True
            return False

        def is_ssm(name: str) -> bool:
            return ("ssm" in name) or ("mixer.ssm" in name)

        ssm_only: bool = bool(self._cfg.get("hydra_ssm_only", False))

        hydra_target_names = []
        for n in hydra_all_names:
            if in_target_layer(n) and (is_ssm(n) if ssm_only else True):
                hydra_target_names.append(n)

        hydra_params = [param for name, param in self.named_parameters() if name in hydra_target_names]
        hydra_names = hydra_target_names  # keep original variable name for downstream use

        assert len(set(mark_names) & set(hydra_all_names)) == 0
        assert len(set(mark_names) & set(cls_names)) == 0
        assert len(set(hydra_all_names) & set(cls_names)) == 0

        if stage == 1:
            for _, p in self.named_parameters():
                p.requires_grad = False
            for name, p in self.named_parameters():
                if name in mark_names:
                    p.requires_grad = True

            param_groups = [{"params": mark_params, "lr": lr_mark}]

        elif stage == 2:
            for p in self.parameters():
                p.requires_grad = False
            for name, p in self.named_parameters():
                if name in mark_names or name in hydra_names:
                    p.requires_grad = True

            param_groups = [
                {"params": hydra_params, "lr": lr_hydra},
                {"params": mark_params, "lr": lr_mark},
            ]

        elif stage == 3:
            for p in self.parameters():
                p.requires_grad = False
            for name, p in self.named_parameters():
                if name in cls_names:
                    p.requires_grad = True

            param_groups = [{"params": cls_params, "lr": lr_cls}]

        typ: str = sched_cfg.get("type", "none")

        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            fused=True
        )

        if typ == "none":
            return {
                "optimizer": optimizer,
                "lr_scheduler": None
            }
        
        if typ in ("cosine", "poly", "inverse_sqrt"):
            total_steps = sched_cfg.get("total_steps")
            if total_steps is None:
                epochs = int(self._cfg.get("epochs", 1))
                steps_per_epoch = len(self._train_dl) if self._train_dl is not None else 1
                total_steps = max(1, epochs * steps_per_epoch)
                
            warmup_steps = sched_cfg.get("warmup_steps", 0)

            if typ == "cosine":
                scheduler = linear_warmup_cosine_decay(
                    optimizer,
                    warmup_steps=warmup_steps,
                    total_steps=total_steps,
                    min_lr_ratio=sched_cfg.get("min_lr_ratio", 0.1)
                )

            if typ == "poly":
                poly = sched_cfg.get("polynomial", {})
                scheduler = linear_warmup_polynomial_decay(
                    optimizer,
                    warmup_steps=warmup_steps,
                    total_steps=total_steps,
                    end_lr_ratio=poly.get("end_lr_ratio", 0.0),
                    power=poly.get("power", 1.0)
                )

            if typ == "inverse_sqrt":
                scheduler = inverse_sqrt_warmup(optimizer, warmup_steps=warmup_steps)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "monitor": "train_loss"
                }
            }
            
        if typ == "plateau":
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            plateau = sched_cfg.get("plateau", {})
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=plateau.get("factor", 0.5),
                patience=plateau.get("patience", 3),
                min_lr=plateau.get("min_lr", 1e-6),
                verbose=plateau.get("verbose", True)
            )

            self.optim = optimizer

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "monitor": "train_loss"
                }
            }
        
        
        raise ValueError(f"Unknown learning rate scheduler type: {typ}")

    def configure_optimizers(self):
        return self.setup_optimization()

    @property
    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @property
    def cfg(self):
        return self._cfg
    
    def teardown(self, stage: str):
        del self._train_dl
        del self._val_dl

        # Only destroy the inner model during training teardown — validation runs
        # can be called multiple times (e.g., across benchmark datasets) and must
        # keep self.inner alive between calls.
        if stage == "fit":
            del self.inner
            self.inner = None

        self._train_dl = None
        self._val_dl = None

        torch.cuda.empty_cache()

        return super().teardown(stage)