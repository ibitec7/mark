import torch
import torch.nn as nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional
import math

from transformers.modeling_outputs import MaskedLMOutput
from transformers import PretrainedConfig, AutoModelForMaskedLM, AutoConfig, AutoModel

from .utils import estimate_utf8_bytes_from_input_ids


@dataclass
class HydraMaskedLMOutput(MaskedLMOutput):
    """Extended MaskedLMOutput with benchmark metrics."""
    raw_nll: Optional[float] = None
    raw_ppl: Optional[float] = None
    raw_bpb: Optional[float] = None
    weighted_nll: Optional[torch.Tensor] = None
    weighted_ppl: Optional[float] = None
    weighted_bpb: Optional[float] = None
    mdlm_masked_nll: Optional[torch.Tensor] = None
    mdlm_masked_ppl: Optional[float] = None
    mdlm_masked_bpb: Optional[float] = None
    mdlm_elbo_nll: Optional[torch.Tensor] = None
    mdlm_elbo_ppl: Optional[float] = None
    mdlm_elbo_bpb: Optional[float] = None
    raw_nll_sum: Optional[torch.Tensor] = None
    raw_weight_sum: Optional[torch.Tensor] = None
    weighted_nll_sum: Optional[torch.Tensor] = None
    weighted_weight_sum: Optional[torch.Tensor] = None
    mdlm_masked_nll_sum: Optional[torch.Tensor] = None
    mdlm_masked_token_count: Optional[torch.Tensor] = None
    mdlm_elbo_nll_sum: Optional[torch.Tensor] = None
    mdlm_elbo_token_count: Optional[torch.Tensor] = None
    byte_count: Optional[torch.Tensor] = None


def _compute_benchmark_statistics(
    token_loss: torch.Tensor,
    masked_p: torch.Tensor,
    labels: torch.Tensor,
    pad_token_id: int,
    cart_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | float]:
    """Compute reportable benchmark statistics from masked-token losses."""
    stats_dtype = torch.float64
    token_loss_stats = token_loss.to(dtype=stats_dtype)
    masked_p_stats = masked_p.to(dtype=stats_dtype).clamp_min(1e-8)
    mdlm_weights = 1.0 / masked_p_stats

    real_token_count = (labels != pad_token_id).sum().to(dtype=stats_dtype).clamp_min(1.0)
    masked_token_count = torch.tensor(
        float(token_loss.numel()), dtype=stats_dtype, device=token_loss.device
    ).clamp_min(1.0)
    byte_count = torch.tensor(
        float(estimate_utf8_bytes_from_input_ids(labels, pad_token_id=pad_token_id)),
        dtype=stats_dtype,
        device=token_loss.device,
    ).clamp_min(1.0)

    raw_nll_sum = token_loss_stats.sum()
    raw_weight_sum = real_token_count
    raw_nll = raw_nll_sum / raw_weight_sum

    masked_cart_weights = None
    if cart_weights is not None:
        masked_cart_weights = cart_weights.to(dtype=stats_dtype)

    weighted_nll_sum = token_loss_stats
    if masked_cart_weights is not None:
        weighted_nll_sum = weighted_nll_sum * masked_cart_weights
    weighted_nll_sum = weighted_nll_sum.sum()

    weighted_weight_sum = masked_token_count
    weighted_nll = weighted_nll_sum / weighted_weight_sum

    mdlm_masked_nll_sum = (token_loss_stats * mdlm_weights).sum()
    mdlm_masked_token_count = masked_token_count
    mdlm_masked_nll = mdlm_masked_nll_sum / mdlm_masked_token_count

    mdlm_elbo_nll_sum = mdlm_masked_nll_sum
    mdlm_elbo_nll = mdlm_elbo_nll_sum / real_token_count

    max_exp = 709.78
    return {
        "raw_nll": raw_nll,
        "raw_ppl": math.exp(min(float(raw_nll), max_exp)),
        "raw_bpb": float(raw_nll_sum) / math.log(2) / float(byte_count),
        "weighted_nll": weighted_nll,
        "weighted_ppl": math.exp(min(float(weighted_nll), max_exp)),
        "weighted_bpb": float(weighted_nll_sum) / math.log(2) / float(byte_count),
        "mdlm_masked_nll": mdlm_masked_nll,
        "mdlm_masked_ppl": math.exp(min(float(mdlm_masked_nll), max_exp)),
        "mdlm_masked_bpb": float(mdlm_masked_nll_sum) / math.log(2) / float(byte_count),
        "mdlm_elbo_nll": mdlm_elbo_nll,
        "mdlm_elbo_ppl": math.exp(min(float(mdlm_elbo_nll), max_exp)),
        "mdlm_elbo_bpb": float(mdlm_elbo_nll_sum) / math.log(2) / float(byte_count),
        "raw_nll_sum": raw_nll_sum.detach(),
        "raw_weight_sum": raw_weight_sum.detach(),
        "weighted_nll_sum": weighted_nll_sum.detach(),
        "weighted_weight_sum": weighted_weight_sum.detach(),
        "mdlm_masked_nll_sum": mdlm_masked_nll_sum.detach(),
        "mdlm_masked_token_count": mdlm_masked_token_count.detach(),
        "mdlm_elbo_nll_sum": mdlm_elbo_nll_sum.detach(),
        "mdlm_elbo_token_count": real_token_count.detach(),
        "byte_count": byte_count.detach(),
    }

from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.generation.utils import GenerationMixin

from .hydra_modules import HydraEmbeddings, HydraEncoder, HydraPooler
from .hydra_heads import HydraOnlyMLMHead, HydraGuiderHead

class HydraForMaskedLMConfig(PretrainedConfig):
    model_type = "hydra_for_masked_lm"  # Required for AutoConfig registration

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30522,
        type_vocab_size: int = 2,
        pad_token_id: int = 0,
        use_position_embeddings: bool = False,
        max_position_embeddings: int = 4096,
        use_timestep_embeddings: bool = True,
        layer_norm_eps: float = 1e-12,
        dropout: float = 0.0,
        max_timestep_embeddings: int = 1000,
        current_timestep: int = 0,
        d_state: int = 64,
        d_conv: int = 7,
        head_dim: int = 64,
        expand: int = 2,
        chunk_size: int = 256,
        is_prenorm: bool = False,
        use_eff_compute: bool = False,
        gradient_checkpointing: bool = True,
        num_hidden_layers: int = 23,
        guider_hidden_layers: int = 12,
        device: str = "cpu",
        pool_all: bool = False,
        mark_kernel: str = "chebyshev",
        mark_ensemble: bool = True,
        rank: int = 2,
        degree: int = 5,
        L_timepoints: int = 256,
        n_freqs: int = 8,
        mark_mlp_dim: int = 256,
        embedding_dim: int = 256,
        hidden_act: str = "swish",
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.auto_map = {
            "AutoConfig": "hydra_model.HydraForMaskedLMConfig",
            "AutoModel": "hydra_model.HydraModel",
            "AutoModelForMaskedLM": "hydra_model.HydraForMaskedLM",
        }
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.use_position_embeddings = use_position_embeddings
        self.max_position_embeddings = max_position_embeddings
        self.use_timestep_embeddings = use_timestep_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.max_timestep_embeddings = max_timestep_embeddings
        self.current_timestep = current_timestep
        self.d_state = d_state
        self.d_conv = d_conv
        self.head_dim = head_dim
        self.expand = expand
        self.chunk_size = chunk_size
        self.is_prenorm = is_prenorm
        self.use_eff_compute = use_eff_compute
        self.gradient_checkpointing = gradient_checkpointing
        self.num_hidden_layers = num_hidden_layers
        self.guider_hidden_layers = guider_hidden_layers
        self.device = device
        self.pool_all = pool_all
        self.mark_kernel = mark_kernel
        self.mark_ensemble = mark_ensemble
        self.rank = rank
        self.degree = degree
        self.L_timepoints = L_timepoints
        self.n_freqs = n_freqs
        self.mark_mlp_dim = mark_mlp_dim
        self.embedding_dim = embedding_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range

class GuiderCore(nn.Module):
    """
    Core module for the Hydra Guider, which includes the main components of the model.
    """

    def __init__(
        self,
        config=None,
    ):
        super().__init__()

        self.embeddings: HydraEmbeddings = HydraEmbeddings(config=config)

        self.encoder: HydraEncoder = HydraEncoder(config=config, guider=True)

        self.prediction: HydraGuiderHead = HydraGuiderHead(config=config)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        output_all_encoded_layers: bool = False,
        labels: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if attention_mask is None:
            attention_mask: torch.Tensor = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids: torch.Tensor = torch.zeros_like(input_ids)

        embedding_output: torch.Tensor = self.embeddings.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )

        encoder_outputs: torch.Tensor = self.encoder.forward(
            hidden_states=embedding_output,
            attn_mask=attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            subset_mask=None
        )

        sequence_output: torch.Tensor = encoder_outputs[-1]

        if not output_all_encoded_layers:
            encoder_outputs = sequence_output

        if labels is None:
            confidence_scores: torch.Tensor = self.prediction.forward(encoder_outputs)

            return confidence_scores, None

        else:
            confidence_scores: torch.Tensor = self.prediction.forward(encoder_outputs)

            loss: torch.Tensor = F.binary_cross_entropy_with_logits(confidence_scores.flatten(), labels.flatten(), reduction="mean")

            return confidence_scores, loss

# Hydra Model
class HydraModel(BertPreTrainedModel, GenerationMixin):
    config_class = HydraForMaskedLMConfig

    def __init__(
        self,
        config=None,
        add_pooling_layer: bool = True,
    ):
        super().__init__(config)

        # The Hydra input embeddings layer.
        self.embeddings: HydraEmbeddings = HydraEmbeddings(config)

        # The Hydra encoder layer.subset_mask
        self.encoder: HydraEncoder = HydraEncoder(config)

        # The Hydra pooling layer.
        self.pooler: HydraPooler = HydraPooler(config) if add_pooling_layer else None

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:

        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):

        self.embeddings.word_embeddings = value

    def forward(
            self,
            input_ids: torch.Tensor,
            seq_idx: torch.Tensor = None,
            token_type_ids: torch.Tensor = None,
            current_timestep: torch.Tensor = None,
            total_timestep: torch.Tensor = None,
            attention_mask: torch.BoolTensor = None,
            position_ids: torch.Tensor = None,
            output_all_encoded_layers: bool = False,
    ) -> tuple[list[torch.Tensor] | torch.Tensor, torch.Tensor | None]:
        
        if attention_mask is None:
            attention_mask: torch.BoolTensor = torch.where(input_ids != self.config.pad_token_id, True, False)
        if token_type_ids is None:
            token_type_ids: torch.Tensor = torch.zeros_like(input_ids, dtype=torch.int32)

        embedding_output, timestep_cond = self.embeddings.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            current_timestep=current_timestep,
            total_timestep=total_timestep,
            attn_mask=attention_mask
        )

        encoder_outputs: torch.Tensor = self.encoder.forward(
            hidden_states=embedding_output,
            seq_idx=seq_idx,
            timestep=timestep_cond,
            attn_mask=attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            subset_mask=None
        )

        sequence_output: torch.Tensor = encoder_outputs[-1]

        pooled_output: torch.Tensor = self.pooler.forward(
            hidden_states=sequence_output, mask=attention_mask
        ) if self.pooler is not None else None

        if not output_all_encoded_layers:
            encoder_outputs = sequence_output

        if self.pooler is not None:
            return encoder_outputs, pooled_output
        
        return encoder_outputs, None
    
class HydraForMaskedLM(BertPreTrainedModel, GenerationMixin):
    config_class = HydraForMaskedLMConfig

    # MaRK adapter params may not exist in base weights
    _keys_to_ignore_on_load_missing = [r"mark\."]
    _keys_to_ignore_on_load_unexpected = [r"mark\."]

    def __init__(
        self,
        config,
    ):
        super().__init__(config)

        self.hydra: HydraModel = HydraModel(config=config, add_pooling_layer=False)

        self.cls: HydraOnlyMLMHead = HydraOnlyMLMHead(config=config,
                                    hydra_model_embedding_weights=self.hydra.embeddings.word_embeddings.weight)
        
        self.post_init()

    @classmethod
    def from_composer(
        cls,
        pretrained_checkpoint,
        state_dict: dict = None,
        cache_dir: str = None,
        config: dict = None,
        *inputs,
        **kwargs
    ):
        model = cls(config, *inputs, **kwargs)

        state_dict: dict = torch.load(pretrained_checkpoint)
        consume_prefix_in_state_dict_if_present(state_dict, prefix="model.")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if len(missing_keys) > 0:
            logging.warning(
                f"Found the following missing keys in the state dict: {missing_keys}"
            )

        if len(unexpected_keys) > 0:
            logging.warning(
                f"Found the following unexpected keys in the state dict: {unexpected_keys}"
            )

        return model
    
    def get_input_embeddings(self):
        return self.hydra.embeddings.word_embeddings
    
    def set_input_embeddings(self, new_embeddings):
        self.hydra.embeddings.word_embeddings = new_embeddings
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder
    
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.BoolTensor | None = None,
        seq_idx: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        current_timestep: torch.Tensor = None,
        total_timestep: torch.Tensor = None,
        input_embeddings: torch.Tensor | None = None,
        masked_tokens_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        p_mask: torch.Tensor | None = None,
        cart_weights: torch.Tensor | None = None,
        output_all_encoded_layers: bool = False,
        benchmark: bool = False,
    ) -> tuple[torch.Tensor] | MaskedLMOutput:
        if (input_ids is not None) == (input_embeddings is not None):
            raise ValueError(
                "You have to specify either input_ids or input_embeddings"
            )

        # Loss/scoring is defined over MASKED tokens only (MLM/MDLM style).
        # If the caller doesn't provide an explicit mask, infer it from the input.
        # Using `labels > 0` would score *all* non-pad tokens and can inflate PPL dramatically.
        if masked_tokens_mask is None:
            masked_tokens_mask = (input_ids == 103)

        if attention_mask is None:
            attention_mask = torch.where(input_ids != self.config.pad_token_id, True, False)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs: tuple[list[torch.Tensor] | torch.Tensor, torch.Tensor | None] = self.hydra.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            seq_idx=seq_idx,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            current_timestep=current_timestep,
            total_timestep=total_timestep,
            output_all_encoded_layers=output_all_encoded_layers,
        )

        encoder_out = outputs[0]
        seq_output: torch.Tensor = encoder_out[-1] if isinstance(encoder_out, list) else encoder_out
        prediction_scores: torch.Tensor = self.cls.forward(seq_output)

        if labels is not None:
            masked_token_idx = masked_tokens_mask.flatten()

            if masked_token_idx.sum().item() == 0:
                loss = prediction_scores.new_tensor(0.0)
                if benchmark:
                    return HydraMaskedLMOutput(
                        loss=loss,
                        logits=prediction_scores,
                        hidden_states=encoder_out if output_all_encoded_layers else None,
                        attentions=None,
                        raw_nll=None,
                        raw_ppl=None,
                        raw_bpb=None,
                        weighted_nll=None,
                        weighted_ppl=None,
                        weighted_bpb=None,
                        mdlm_masked_nll=None,
                        mdlm_masked_ppl=None,
                        mdlm_masked_bpb=None,
                        mdlm_elbo_nll=None,
                        mdlm_elbo_ppl=None,
                        mdlm_elbo_bpb=None,
                        raw_nll_sum=None,
                        raw_weight_sum=None,
                        weighted_nll_sum=None,
                        weighted_weight_sum=None,
                        mdlm_masked_nll_sum=None,
                        mdlm_masked_token_count=None,
                        mdlm_elbo_nll_sum=None,
                        mdlm_elbo_token_count=None,
                        byte_count=None,
                    )
                if not return_dict:
                    output = (prediction_scores,) + outputs[2:]
                    return (loss,) + output
                return MaskedLMOutput(
                    loss=loss,
                    logits=prediction_scores,
                    hidden_states=encoder_out if output_all_encoded_layers else None,
                    attentions=None,
                )

            token_loss: torch.Tensor = F.cross_entropy(
                input=prediction_scores.view(-1, prediction_scores.size(-1))[masked_token_idx],
                target=labels.flatten()[masked_token_idx],
                reduction="none"
            )

            if benchmark:
                if p_mask is None:
                    raise ValueError("p_mask is required to compute benchmark metrics.")

                masked_p = p_mask.flatten()[masked_token_idx].to(dtype=token_loss.dtype).clamp_min(1e-8)
                pad_token_id = getattr(self.config, "pad_token_id", 0)
                benchmark_stats = _compute_benchmark_statistics(
                    token_loss=token_loss,
                    masked_p=masked_p,
                    labels=labels,
                    pad_token_id=pad_token_id,
                    cart_weights=cart_weights,
                )
                loss = benchmark_stats["weighted_nll"]

                return HydraMaskedLMOutput(
                    loss=loss,
                    logits=prediction_scores,
                    hidden_states=encoder_out if output_all_encoded_layers else None,
                    attentions=None,
                    raw_nll=benchmark_stats["raw_nll"],
                    raw_ppl=benchmark_stats["raw_ppl"],
                    raw_bpb=benchmark_stats["raw_bpb"],
                    weighted_nll=benchmark_stats["weighted_nll"],
                    weighted_ppl=benchmark_stats["weighted_ppl"],
                    weighted_bpb=benchmark_stats["weighted_bpb"],
                    mdlm_masked_nll=benchmark_stats["mdlm_masked_nll"],
                    mdlm_masked_ppl=benchmark_stats["mdlm_masked_ppl"],
                    mdlm_masked_bpb=benchmark_stats["mdlm_masked_bpb"],
                    mdlm_elbo_nll=benchmark_stats["mdlm_elbo_nll"],
                    mdlm_elbo_ppl=benchmark_stats["mdlm_elbo_ppl"],
                    mdlm_elbo_bpb=benchmark_stats["mdlm_elbo_bpb"],
                    raw_nll_sum=benchmark_stats["raw_nll_sum"],
                    raw_weight_sum=benchmark_stats["raw_weight_sum"],
                    weighted_nll_sum=benchmark_stats["weighted_nll_sum"],
                    weighted_weight_sum=benchmark_stats["weighted_weight_sum"],
                    mdlm_masked_nll_sum=benchmark_stats["mdlm_masked_nll_sum"],
                    mdlm_masked_token_count=benchmark_stats["mdlm_masked_token_count"],
                    mdlm_elbo_nll_sum=benchmark_stats["mdlm_elbo_nll_sum"],
                    mdlm_elbo_token_count=benchmark_stats["mdlm_elbo_token_count"],
                    byte_count=benchmark_stats["byte_count"],
                )

            if p_mask is not None and cart_weights is None:
                weights = 1.0 / (p_mask.flatten()[masked_token_idx] + 1e-8)
                weighted_loss = token_loss * weights
                loss = weighted_loss.sum() / (masked_tokens_mask.sum() + 1e-8)
            elif cart_weights is not None:
                weighted_loss = token_loss * cart_weights
                loss = weighted_loss.sum() / (masked_tokens_mask.sum() + 1e-8)
            else:
                loss = token_loss.mean()

        else:
            loss = None

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=encoder_out if output_all_encoded_layers else None,
            attentions=None
        )

    def setup_training_data(self, config, input_ids: torch.Tensor = None, attn_mask: torch.Tensor = None, seq_idx: torch.Tensor = None) -> DataLoader:

        if input_ids is None:
            input_ids = torch.load(os.path.join(DATA_DIR, "train_tokens.pt")).to(dtype=torch.int32)

        if attn_mask is None and os.path.exists(os.path.join(DATA_DIR, "train_attention_mask.pt")):
            attn_mask = torch.load(os.path.join(DATA_DIR, "train_attention_mask.pt")).to(dtype=torch.int8)

        data: TensorDataset = TensorDataset(input_ids, attn_mask, seq_idx) if attn_mask is not None else TensorDataset(input_ids, seq_idx)

        self._train_dl = DataLoader(
            data,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )        
    
    def setup_validation_data(self, config, input_ids: torch.Tensor = None, attn_mask: torch.Tensor = None, seq_idx: torch.Tensor = None):
        if input_ids is None:
            input_ids: torch.Tensor = torch.load(os.path.join(DATA_DIR, "valid_tokens.pt")).to(dtype=torch.int32)

        if attn_mask is None and os.path.exists(os.path.join(DATA_DIR, "valid_attention_mask.pt")):
            attn_mask: torch.Tensor = torch.load(os.path.join(DATA_DIR, "valid_attention_mask.pt")).to(dtype=torch.int8)
            
        data: TensorDataset = TensorDataset(input_ids, attn_mask, seq_idx) if attn_mask is not None else TensorDataset(input_ids, seq_idx)

        self._val_dl = DataLoader(
            dataset=data,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **model_kwargs
    ):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        if self.config.pad_token_id is None:
            raise ValueError(
                "The PAD token should be defined for generation"
            )
        
        attention_mask = torch.cat([
            attention_mask, attention_mask.new_zeros(attention_mask.shape[0], 1 )
        ], dim=-1)

        dummy_token: torch.Tensor = torch.full(
            (effective_batch_size, 1),
            self.config.pad_token_id,
            dtype=torch.long,
            device=input_ids.device
        )

        input_ids: torch.Tensor = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
   # Inference Methods
    def iterative_decode(
            self,
            prompt_ids: torch.Tensor,
            response_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            num_steps: int = 10,
            mask_idx_func: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
            guider_scorer: Any = None,
            **kwargs
    ) -> torch.Tensor:
        """
        Iteratively decode masked response tokens using confidence-based remasking.

        At each step the model predicts all tokens, keeps the most confident ones,
        and re-masks the least confident positions for the next iteration. The number
        of re-masked tokens decreases linearly to zero over ``num_steps``.

        Args:
            prompt_ids (torch.Tensor): Prompt token IDs, shape ``[batch, prompt_len]`` or ``[prompt_len]``.
            response_ids (torch.Tensor): Initial (masked) response IDs, shape ``[batch, response_len]`` or ``[response_len]``.
            attention_mask (torch.Tensor): Attention mask for the full sequence (prompt + response).
            num_steps (int): Number of iterative decoding steps.
            mask_idx_func (callable | None): ``(resp_proba, k) -> [batch, k]`` int indices
                over the **response** region (least-confident selection is up to the caller).
                Ignored when ``guider_scorer`` is set.
            guider_scorer: Optional :class:`~src.mamba_guider.MambaGuiderScorer`. When set,
                remasking uses Mamba2 target-token probabilities instead of Hydra MLM confidence.

        Returns:
            torch.Tensor: Decoded response token IDs on CPU, shape ``[batch, response_len]``.
        """
        self.eval()
        device = next(self.parameters()).device

        # Ensure batch dimension
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if response_ids.ndim == 1:
            response_ids = response_ids.unsqueeze(0)

        prompt_ids = prompt_ids.to(device)
        response_ids = response_ids.to(device)

        prompt_len = prompt_ids.shape[-1]
        response_len = response_ids.shape[-1]

        # Build full input: [prompt | response]
        input_ids: torch.Tensor = torch.cat((prompt_ids, response_ids), dim=1)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Boolean mask over the response region (True = currently masked)
        response_mask: torch.BoolTensor = torch.ones(
            input_ids.shape[0], response_len, dtype=torch.bool, device=device
        )

        with torch.no_grad():
            for step in range(num_steps):
                # Ensure prompt tokens remain unchanged
                input_ids[:, :prompt_len] = prompt_ids

                # Match training conditioning: training samples (current_step, total_steps) and sets
                # mask_ratio = 1 - current_step/total_steps. Here we approximate mask_ratio by the
                # current fraction of [MASK] tokens in the response region.
                resp_region = input_ids[:, prompt_len:]
                resp_mask_ratio = (resp_region == 103).float().mean().item()
                current_step = int(max(0, min(999, round((1.0 - resp_mask_ratio) * 1000))))

                outputs: MaskedLMOutput = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    current_timestep=current_step,
                    total_timestep=1000,
                )

                # Token predictions and their confidence scores
                proba = torch.softmax(outputs.logits, dim=-1)
                pred_tokens_proba, pred_tokens = proba.max(dim=-1)

                # Extract response portion only
                resp_proba = pred_tokens_proba[:, prompt_len:]     # [batch, response_len]
                resp_tokens = pred_tokens[:, prompt_len:]          # [batch, response_len]

                # Number of tokens to re-mask (decreases linearly to 0)
                mask_k = max(1, math.ceil((1.0 - (float(step + 1) / float(num_steps))) * response_len))

                if step == num_steps - 1:
                    # Final step: accept all predictions
                    input_ids[:, prompt_len:] = resp_tokens
                    break

                # Build new mask: re-mask the least confident positions
                if guider_scorer is not None:
                    full_pred = torch.cat((prompt_ids, resp_tokens), dim=1)
                    score_tensor = guider_scorer.response_confidence_scores(
                        full_pred, prompt_len
                    )
                    remask_indices = torch.topk(
                        score_tensor, k=mask_k, dim=-1, largest=False
                    ).indices
                elif mask_idx_func is not None:
                    remask_indices = mask_idx_func(resp_proba, mask_k)
                else:
                    remask_indices = torch.topk(
                        resp_proba, k=mask_k, dim=-1, largest=False
                    ).indices  # [batch, mask_k]

                # Convert integer indices to boolean mask over response region
                response_mask.fill_(False)
                response_mask.scatter_(1, remask_indices, True)

                # Accept predictions for unmasked positions, re-mask the rest
                new_response = resp_tokens.clone()
                new_response[response_mask] = 103  # [MASK] token

                input_ids[:, prompt_len:] = new_response

        return input_ids[:, prompt_len:].detach().cpu()

    def inference(
            self,
            prompt: list | torch.Tensor,
            seq_len: int = 128,
            sampling_steps: int = 30,
            mask_id: int = 103,
            guider_scorer: Any = None,
    ) -> torch.Tensor:
        """
        Generate a response for a given prompt using iterative masked decoding.

        Args:
            prompt (list | torch.Tensor): Prompt token IDs.
            seq_len (int): Length of the response to generate.
            sampling_steps (int): Number of iterative decoding steps.
            mask_id (int): Token ID used for masking (default: 103 for BERT [MASK]).
            guider_scorer: Optional Mamba guider for remasking (see ``iterative_decode``).

        Returns:
            torch.Tensor: Decoded response token IDs on CPU, shape ``[batch, seq_len]``.
        """
        device = next(self.parameters()).device
        if isinstance(prompt, list):
            prompt_ids = torch.tensor(prompt, dtype=torch.long).unsqueeze(0).to(device)
        else:
            prompt_ids = prompt.unsqueeze(0).to(device) if prompt.ndim == 1 else prompt.to(device)

        response_ids = torch.full(
            (prompt_ids.shape[0], seq_len), mask_id, dtype=torch.long, device=device
        )
        # Set last token to [SEP]
        response_ids[:, -1] = 102

        full_ids = torch.cat((prompt_ids, response_ids), dim=1)
        attn_mask = (full_ids != 0)

        return self.iterative_decode(
            prompt_ids,
            response_ids,
            attention_mask=attn_mask,
            num_steps=sampling_steps,
            guider_scorer=guider_scorer,
        )

# Register with AutoConfig; ensure model's config_class matches for Auto registration
HydraForMaskedLM.config_class = HydraForMaskedLMConfig
AutoConfig.register("hydra_for_masked_lm", HydraForMaskedLMConfig)
AutoModel.register(HydraForMaskedLMConfig, HydraModel)
AutoModelForMaskedLM.register(HydraForMaskedLMConfig, HydraForMaskedLM)