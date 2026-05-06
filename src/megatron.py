import torch
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.trainer.trainer import Trainer
from omegaconf.dictconfig import DictConfig

from transformers.modeling_outputs import MaskedLMOutput
from .utils import record_batch, arrow_dataloader
from .train import sample_timestep, masking_process, context_adaptive_reweight
from .scheduler import linear_warmup_cosine_decay, linear_warmup_polynomial_decay, inverse_sqrt_warmup

from .hydra_model import HydraForMaskedLM

from nemo.utils import logging
from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel

import time
import wandb
from omegaconf import OmegaConf
from typing import Optional, Union

"""
Inheritance structure for the MegatronMambaModel:

ModelPT -> NLPModel -> MegatronBaseModel -> MegatronGPTModel -> MegatronMambaModel

Inheritance structure for the MegatronHydraModel:

NemoForMaskedLM (Custom ModelPT implementation) -> NLPModel -> MegatronBaseModel -> MegatronBertModel -> MegatronHydraModel

To refine we will move the NemoForMaskedLM implementation into the MegatronHydraModel to avoid inheritance conflicts, as both MegatronBertModel and NemoForMaskedLM inherit from ModelPT. This way, we can have a clean inheritance structure without conflicts.

ModelPT -> NLPModel -> MegatronBaseModel -> MegatronBertModel -> MegatronHydraModel

"""

class MegatronHydraModel(MegatronBertModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer, tracker:wandb.Run=None, inference: bool=False):
        self.vocab_size = cfg.get('vocab_size', 30522)
        self.cfg = cfg
        super().__init__(cfg=cfg, trainer=trainer)
        logging.warning("Overriding mcore_gpt=True")
        self.mcore_gpt = True

        self.cart = cfg.get("cart", False)
        self.cart_p = cfg.get("cart_p", 0.5)

        self.data_dir = cfg.get("data_dir", "./data")

        if not inference:
            self.tracker = tracker if tracker is not None else wandb.init(project="hydra-masked-lm", config=OmegaConf.to_container(cfg, resolve=True), reinit=True)
        else:
            self.tracker = None

        self.val_losses = []
        self._logged_unused = False

    def model_provider_func(self, pre_process, post_process):
        # [WIP] Need to implement this method. It will return my custom Hydra ModelPT instance.
        if not isinstance(self.cfg, DictConfig):
            raise ValueError("cfg must be an instance of DictConfig")

        model = HydraForMaskedLM(
            config=self.cfg,
        )

        return model
    
    def get_forward_output_and_loss_func(self, validation_step=False, tuning=False):
        """ Fetch data and run the Forward step """

        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=False):
            start_time = time.time()

            batch, batch_idx, dataloader_idx = next(dataloader_iter)
            items = self._prepare_batch_inputs(batch)

            tokens, labels, p_mask, attn_mask, current_step, total_steps, masked_indices = (
                items["corrupt_tokens"],
                items["labels"],
                items["p_mask"],
                items["attention_mask"],
                items["current_step"],
                items["total_steps"],
                items["masked_indices"]
            )

            weights_for_loss = None

            if self.cart:
                seq_len = attn_mask.sum().item() if not validation_step else tokens.shape[-1]
                weight_matrix = context_adaptive_reweight(
                    seq_len, cart_p=self.cart_p
                )
                _weight_matrix: torch.Tensor = weight_matrix[:seq_len, :seq_len].to(
                    masked_indices.device
                )
                non_mask: torch.Tensor = ~masked_indices.to(
                    masked_indices.device
                )
                
                cart_weights = (
                    non_mask.type_as(_weight_matrix)
                    .matmul(_weight_matrix)
                    .masked_fill(non_mask, 0)
                )

                masked_idx_flat: torch.Tensor = masked_indices.flatten()
                weights_for_loss: torch.Tensor = cart_weights.flatten()[masked_idx_flat] + 1e-8

            dataloader_iter._dataloader_idx = dataloader_idx
            dataloader_iter._batch_idx = batch_idx

            forward_args = {
                "input_ids": tokens,
                "attention_mask": attn_mask,
                "current_timestep": current_step,
                "total_timestep": total_steps,
                "labels": labels,
                "masked_tokens_mask": masked_indices,
                "p_mask": p_mask,
                "cart_weights": weights_for_loss
            }

            outputs = self.forward(**forward_args)

            end_time = time.time()
            step_duration = end_time - start_time

            # Training-specific logging
            if not validation_step:
                if self._trainer is not None:
                    self.log("train_loss", outputs.loss.item(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=tokens.shape[0])

                if self.tracker is not None:
                    data: dict = record_batch(
                        batch_id=batch_idx,
                        epoch=self.current_epoch,
                        loss=outputs.loss.item(),
                        logits=outputs.logits.detach().cpu(),
                        labels=labels.detach().cpu(),
                        masked_indices=masked_indices.detach().cpu(),
                        current_step=current_step.item() if hasattr(current_step, 'item') else current_step,
                        total_steps=total_steps.item() if hasattr(total_steps, 'item') else total_steps,
                        model=self.model,
                        optimizer=getattr(self, 'optim', None),
                    )

                    data["step_duration"] = step_duration
                    self.tracker.log(data)

            def loss_func(output_tensor: MaskedLMOutput):
                loss = output_tensor.loss

                reduced_loss = {"avg": loss}

                if validation_step:
                    reduced_loss["val_loss"] = loss
                    
                    # Validation-specific logic
                    if hasattr(self, 'val_losses'):
                        self.val_losses.append(loss.item())
                    
                    if self._trainer is not None:
                        self.log("val_loss", loss.item(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=tokens.shape[0])
                    
                    if self.tracker is not None:
                        self.tracker.log({
                            "val_loss": loss.item(),
                            "epoch": self.current_epoch,
                            "batch_id": batch_idx,
                        })

                return loss, reduced_loss
                    
            return outputs, loss_func

        return fwd_output_and_loss_func
    
    def inference(
            self,
            prompt: Optional[list],
            seq_len: int=128,
            sampling_steps: int=30,
            mask_id: int=103
    ):
        # Prepare the input tensor to feed for inference.
        input_ids: torch.Tensor = torch.tensor([prompt] + [101] + [mask_id] * seq_len).unsqueeze(0).cuda()
        attn_mask = (input_ids != 101 and input_ids != mask_id)

        self.iterative_decode(input_ids, )
        

    def iterative_decode(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            num_steps: int = 10,
            **kwargs
    ):
        
        self.eval()
        current_ids = input_ids.clone()

        with torch.no_grad():
            for step in range(num_steps):

                outputs: torch.Tensor = self.forward(
                    input_ids=current_ids, 
                    attention_mask=attention_mask, 
                    current_timestep=step, 
                    total_timestep=num_steps
                )

                mask_token_id = 103
                masked_positions = (current_ids == mask_token_id)
                predictions = outputs.logits.argmax(dim=-1)
                current_ids[masked_positions] = predictions[masked_positions]


                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                next_tokens = torch.argmax(probs, dim=-1)

                # Update only the masked positions
                masked_positions = kwargs.get("masked_tokens_mask", None)
                if masked_positions is not None:
                    current_ids[masked_positions] = next_tokens[masked_positions]
                else:
                    current_ids = next_tokens

    def generate():
        # [WIP] Need to implement this method. It will handle generation for the Hydra model.
        pass

    # [WIP] Move the NemoForMaskedLM implementation here to prevent inheritance conflicts.
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.BoolTensor = None,
        current_timestep: int = 0,
        total_timestep: int = 0,
        labels: torch.Tensor = None,
        p_mask: torch.Tensor = None,
        masked_tokens_mask: torch.Tensor = None,
        cart_weights: torch.Tensor = None,
    ) -> MaskedLMOutput:
        
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            current_timestep=current_timestep,
            total_timestep=total_timestep,
            labels=labels,
            p_mask=p_mask,
            masked_tokens_mask=masked_tokens_mask,
            cart_weights=cart_weights,
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
    
    # [WIP] Configure safety for DDP hooks to prevent race conditions and deadlocks.
    def setup(self, stage: Optional[str] = None):
        super().setup(stage)

        if torch.cuda.is_available() and next(self.parameters()).is_cuda:
                device = next(self.parameters()).device

        if self.inner.hydra.encoder.mark.streams is None:
            print("Initializing MARK adapter CUDA streams on device: ", device)
            self.inner.hydra.encoder.mark.streams = [
                torch.cuda.Stream(device=device) for _ in range(self.inner.hydra.encoder.mark.num_adapters)
            ]

            self.inner.hydra.encoder.mark._streams_initialized = True

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
            batch_tokens: torch.Tensor = batch_tokens.to(device="cuda", non_blocking=True)
            # batch_mask: torch.Tensor = batch_mask.to(device="cuda", non_blocking=True)
        
        ### ============================================================================================================= ###

        current_step, total_steps = sample_timestep()               # Sample timestep for each batch

        ratio: float = 1.0 - (float(current_step) / float(total_steps))  # Masking ratio based on current timestep

        # sep_id = 102
        # new_seq_pos = F.pad((batch_tokens[:, :-1] == sep_id).to(torch.int32), (1, 0))
        # seq_idx: torch.Tensor = torch.cumsum(new_seq_pos, dim=1).to(torch.int32)

        corrupt_tokens, masked_indices, p_mask = masking_process(input_ids=batch_tokens, mask_ratio=ratio, attn_mask=None)

        return dict(
            corrupt_tokens=corrupt_tokens,
            labels=batch_tokens,
            p_mask=p_mask,
            # seq_idx=seq_idx,
            attention_mask=(batch_tokens != 0),
            current_step=current_step,
            total_steps=total_steps,
            masked_indices=masked_indices
        )

    def on_validation_epoch_end(self):
        if self.val_losses:
            avg_val = torch.tensor(self.val_losses, device="cuda").mean()
            self.log("val_loss_epoch", avg_val, sync_dist=True, prog_bar=True)

        self.val_losses.clear()

        super().on_validation_epoch_end()

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

        print(len(mark_names))

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
                    "interval": "epoch",
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
                    "interval": "epoch",
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


