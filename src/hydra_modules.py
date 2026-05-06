from . import padding
import torch
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
import math

import logging
import os

from transformers.activations import ACT2FN
from .hydra import Hydra
from .guider import Guider
from .mark import MarkEnsemble

logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename=os.path.join("logs", "bert_layers.log"), mode='w')
    ]
)

torch.backends.cudnn.allow_tf32 = True

# The embedding layer for the Hydra model.
class HydraEmbeddings(nn.Module):
    """
    
    Construct the embeddings for the input words. Used to convert token IDs into dense embedding vectors.

    Positional embeddings are not calcula ted when they are set to None.

    Timestep embeddings are calculated based on the current timestep and total timestep.

    Args:
        vocab_size {int}: Vocabulary size of the input embeddings.

        type_vocab_size {int}: Vocabulary size for token type embeddings.

        d_model {int}: Dimension of the embeddings.

        pad_token_id {int}: Padding token ID.
        
        use_position_embeddings {bool}: Whether to use positional embeddings.
        
        max_position_embeddings {int}: Maximum sequence length for positional embeddings.
        
        use_timestep_embeddings {bool}: Whether to use timestep embeddings.
        
        current_timestep {int}: Current timestep for timestep embeddings.
        
        max_timestep_embeddings {int}: Maximum sequence length for timestep embeddings.
        
        layer_norm_eps {float}: Epsilon for layer normalization.
        
        dropout {float}: Dropout probability.

    Attributes:
        
        word_embeddings {nn.Embedding}: Embedding layer for input words.
        
        position_embeddings {nn.Embedding}: Embedding layer for positional encodings.
        
        token_type_ids {torch.Tensor}: Tensor to hold token type IDs.
        
        LayerNorm {nn.LayerNorm}: Layer normalization layer.
        
        dropout {nn.Dropout}: Dropout layer.
    """

    def __init__(
            self,
            config=None
    ):
        super().__init__()

        factory_kwargs = {"dtype": torch.float32}

        self.word_embeddings: nn.Embedding = nn.Embedding(
            config.vocab_size if config.vocab_size is not None else 30522,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            **factory_kwargs
        )

        self.max_timestep = config.max_timestep_embeddings

        # Timestep embedding components

        half = 128 // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        # Register as buffer so it moves with the model during DDP
        self.register_buffer('freqs', freqs)

        self.time_mlp = nn.Sequential(
            nn.Linear(128, config.embedding_dim),
            nn.SiLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        
        self.use_position_embeddings: bool = config.use_position_embeddings

        if self.use_position_embeddings:
            self.position_embeddings: nn.Embedding = nn.Embedding(
                config.max_position_embeddings,
                config.hidden_size,
                **factory_kwargs
            )

            self.register_buffer(
                "position_ids",
                torch.arange(config.max_position_embeddings).expand((1, -1))
            )

        self.token_type_embeddings: nn.Embedding = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size,
            **factory_kwargs
        )

        self.register_buffer(
            "token_type_ids",
            torch.zeros(
                config.max_position_embeddings,
                dtype=torch.long
            ),
            persistent=False
        )

        self.LayerNorm: nn.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, **factory_kwargs)
        self.dropout: nn.Dropout = nn.Dropout(p=config.dropout)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            token_type_ids: torch.LongTensor = None,
            position_ids: torch.LongTensor = None,
            input_embeddings: torch.FloatTensor = None,
            current_timestep: int = None,
            total_timestep: int = None,
            attn_mask: torch.BoolTensor = None,
            past_key_values_length: int = 0,
    ) -> list[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if (input_ids is not None) == (input_embeddings is not None):
            raise ValueError("Must specify either input_ids or input_embeddings, but not both.")
        
        if input_ids is not None:
            input_shape: torch.Size = input_ids.size()
        else:
            assert input_embeddings is not None, "Must specify either input_ids or input_embeddings."
            input_shape: torch.Size = input_embeddings.size()[:-1]

        # Process timestep embeddings
        timestep_ratio: float = (float(current_timestep) / float(total_timestep)) \
            if (current_timestep is not None and total_timestep is not None) else 0.0       # First normalize to get a ratio [0,1]
        
        # Get the timestep embedding - use device from input_ids to ensure consistency with DDP
        #t: torch.Tensor = torch.tensor(timestep_ratio, device=self.device, dtype=torch.float32)
        device = input_ids.device if input_ids is not None else input_embeddings.device
        t: torch.Tensor = torch.tensor(timestep_ratio, device=device, dtype=torch.float32)      # First create a tensor of shape (1,)

        args = (t * self.max_timestep) * self.freqs                                  # Shape (128,)  

        sin_embed: torch.Tensor = torch.cat([args.sin(), args.cos()], dim=-1)        # Shape (128,)

        t_embedding: torch.Tensor = self.time_mlp(sin_embed)                         # Shape (128,)

        seq_len: int = input_shape[1]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                assert isinstance(self.token_type_ids, torch.LongTensor), "token_type_ids must be a LongTensor."
                buffered_token_type_ids: torch.Tensor = self.token_type_ids[:, :seq_len]
                buffered_token_type_ids_expanded: torch.Tensor = buffered_token_type_ids.expand(input_ids.shape[0], seq_len)

                token_type_ids: torch.Tensor = buffered_token_type_ids_expanded

            else:
                token_type_ids: torch.Tensor = torch.zeros(input_shape,
                                             dtype=torch.long,
                                             device=self.word_embeddings.weight.device)
                
        if input_embeddings is None:
            assert input_ids is not None, "Must specify either input_ids or input_embeddings."

            input_embeddings: torch.Tensor = self.word_embeddings(input_ids)
        
        token_type_embeddings: torch.Tensor = self.token_type_embeddings(token_type_ids)

        embeddings = input_embeddings + token_type_embeddings

        if self.use_position_embeddings:
            if position_ids is None:
                position_ids = self.position_ids[:, past_key_values_length : seq_len + past_key_values_length]

            position_embeddings: torch.Tensor = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        assert embeddings is not None, "embeddings should not be None after computation."

        return embeddings, t_embedding
    
class HydraUnpadMixer(nn.Module):

    """
        HydraUnpadMixer is a module that processes hidden states by applying a Hydra mixer and unpadding the output.

        Args:
            
            d_model {int}: Dimensionality of the model, default is 768.
            
            d_state {int}: Dimensionality of the state, default is 64.
            
            d_conv {int}: Dimensionality of the convolution, default is 17.
            
            head_dim {int}: Dimensionality of the head, default is 64.
            
            expand {int}: Expansion factor, default is 2.
            
            chunk_size {int}: Size of the chunks for processing, default is 256.
            
            max_position_embeddings {int}: Maximum sequence length for positional embeddings, default is 512.
            
            is_prenorm {bool}: Whether to apply prenormalization, default is True.

        Attributes:
            mixer {Hydra}: An instance of the Hydra class for mixing hidden states.
            
            norm {nn.LayerNorm}: Layer normalization for the output.
            
            is_prenorm {bool}: Flag to indicate if prenormalization is applied.
    """

    def __init__(
            self,
            config=None,
            guider: bool = False
    ):
        super().__init__()
        self.guider = guider

        if not guider:
            self.mixer: Hydra = Hydra(
                d_model=config.hidden_size,
                d_state=config.d_state,
                d_conv=config.d_conv,
                head_dim=config.head_dim,
                activation=config.hidden_act,
                expand=config.expand,
                device=config.device,
                use_eff_compute=config.use_eff_compute,
                mark_kernel=config.mark_kernel,
                mark_ensemble=config.mark_ensemble,
                rank=config.rank,
                degree=config.degree,
                L_timepoints=config.L_timepoints,
                n_freqs=config.n_freqs,
                mark_mlp_dim=config.mark_mlp_dim,
                embedding_dim=config.embedding_dim,
                chunk_size=min(config.chunk_size, config.max_position_embeddings)
            )
        else:
            self.mixer: Guider = Guider(
                d_model=config.hidden_size,
                d_state=config.d_state,
                d_conv=config.d_conv,
                head_dim=config.head_dim,
                expand=config.expand,
                device=config.device,
                chunk_size=min(config.chunk_size, config.max_position_embeddings)
            )

        self.norm: nn.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-5, device=config.device)
        self.is_prenorm: bool = config.is_prenorm

    def forward(
            self,
            hidden_states: torch.Tensor,
            seq_idx: torch.Tensor,
            cu_seqlens: torch.Tensor,
            timestep: torch.Tensor = None,
            max_seq_len: int=512,
            subset_idx: torch.Tensor=None,
            indices: torch.Tensor=None,
            attn_mask: torch.BoolTensor=None,
            param_update: tuple = None
    ) -> torch.Tensor:
        
        residual: torch.Tensor = hidden_states

        if self.is_prenorm:
            hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))

        hidden_states = padding.pad_input(
            hidden_states=hidden_states, indices=indices, batch_size=cu_seqlens.shape[0] - 1, seq_len=max_seq_len
        )
        
        if not self.guider:
            output: torch.Tensor = self.mixer(
                hidden_states,
                timestep,
                seq_idx,
                param_update=param_update
            )
        else:
            output: torch.Tensor = self.mixer(
                hidden_states,
                seq_idx
            )
        output = padding.unpad_input_only(output, (torch.squeeze(attn_mask) == True))
        output = output + residual

        if not self.is_prenorm:
            output = self.norm(output.to(dtype=self.norm.weight.dtype))

        if subset_idx is not None:
            output = padding.index_first_axis(output, subset_idx)

        return output
    
class HydraLayer(nn.Module):
    """
        Hydra layer that includes the sequence mixing (such as Hydra) and the state mixing (such as MLP).
    """

    def __init__(
            self,
            config=None,
            guider: bool = False
    ):
        super().__init__()
        self.guider = guider

        self.layer = HydraUnpadMixer(config=config, guider=guider)

    def forward(
            self,
            hidden_states: torch.Tensor,
            seq_idx: torch.Tensor,
            timestep: torch.Tensor,
            cu_seqlens: torch.Tensor,
            seq_len: int=512,
            subset_idx: torch.Tensor=None,
            indices: torch.Tensor=None,
            attn_mask: torch.BoolTensor=None,
            param_update: tuple = None
    ) -> torch.Tensor:
        
        if not self.guider:
            layer_output: torch.Tensor = self.layer.forward(
                hidden_states=hidden_states,
                seq_idx=seq_idx,
                timestep=timestep,
                cu_seqlens=cu_seqlens,
                max_seq_len=seq_len,
                subset_idx=subset_idx,
                indices=indices,
                attn_mask=attn_mask,
                param_update=param_update
            )
        else:
            layer_output: torch.Tensor = self.layer.forward(
                hidden_states=hidden_states,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                max_seq_len=seq_len,
                subset_idx=subset_idx,
                indices=indices,
                attn_mask=attn_mask
            )

        if type(layer_output) == tuple:
            layer_output, _ = layer_output

        return layer_output
    
class HydraEncoder(nn.Module):

    def __init__(
            self,
            config=None,
            guider: bool = False
    ):
        super().__init__()
        self.guider = guider

        if not guider:
            self.layer: HydraLayer = nn.ModuleList(
                [HydraLayer(config=config, guider=False) for _ in range(config.num_hidden_layers)]
            )
        else:
            self.layer: HydraLayer = nn.ModuleList(
                [HydraLayer(config=config, guider=True) for _ in range(config.guider_hidden_layers)]
            )

        if config.mark_ensemble == True:
            self.mark = MarkEnsemble(
                num_adapters=config.num_hidden_layers,
                adapter_class=config.mark_kernel,
                cond_dim=config.embedding_dim,
                n_heads= config.hidden_size // config.head_dim,
                n_groups=1,
                d_state=config.d_state,
                degree=config.degree,
                rank=config.rank,
                hidden_dim=config.mark_mlp_dim,
                factory_kwargs={"device": config.device, "dtype": config.dtype}
            )
        else:
            self.mark = None

        self.is_prenorm: bool = config.is_prenorm
        if self.is_prenorm:
            self.norm: nn.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-5)

        # Gradient checkpointing fallback to False if not specified in config
        # self.gradient_checkpointing = config.gradient_checkpointing
        self.gradient_checkpointing = getattr(config, 'gradient_checkpointing', False)

    def forward(
            self,
            hidden_states: torch.Tensor,
            seq_idx: torch.Tensor,
            timestep: torch.Tensor,     # The timestep embedding
            attn_mask: torch.BoolTensor,
            output_all_encoded_layers: bool=False,
            subset_mask: torch.Tensor=None,
    ) -> list[torch.Tensor]:

        batch_size, seq_len = hidden_states.shape[:2]

        hidden_states, indices, cu_seqlens, _ = padding.unpad_input(
            hidden_states=hidden_states, attention_mask=attn_mask
        )

        # Run the MaRK adapter to get the Hydra parameter updates based on timestep and mask embeddings

# ==================================================================================================================#
        if self.mark is not None:
            cond_embedding: torch.Tensor = timestep   # Shape: (embedding_dim,)

            param_updates: list[tuple] = self.mark.forward(
                cond=cond_embedding,
            )
        else:
            param_updates: list[tuple] = [None for _ in range(len(self.layer))]
# =======================================================================================================================#

        all_encoder_layers: list[torch.Tensor] = []

        use_checkpoint = self.training and self.gradient_checkpointing

        if subset_mask is None and not use_checkpoint:             # For inference/eval
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module.forward(
                    hidden_states=hidden_states,
                    seq_idx=seq_idx,
                    timestep=timestep,
                    cu_seqlens=cu_seqlens,
                    seq_len=seq_len,
                    subset_idx=None,
                    indices=indices,
                    attn_mask=attn_mask,
                    param_update=param_updates[i]
                )

                # Add current layer hidden states to all_encoder_layers if needed
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)

            if self.is_prenorm:
                hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))

            hidden_states = padding.pad_input(
                hidden_states=hidden_states, indices=indices, batch_size=batch_size, seq_len=seq_len
            )

        else:           # For training with checkpointing (or subset_mask path)
            for i in range(len(self.layer) - 1):
                layer_module: HydraLayer = self.layer[i]

                if use_checkpoint:
                    hidden_states = checkpoint(
                        layer_module.forward,
                        hidden_states,
                        seq_idx,
                        timestep,
                        cu_seqlens,
                        seq_len,
                        None,
                        indices,
                        attn_mask,
                        use_reentrant=False,
                        param_update=param_updates[i]
                    )
                else:
                    hidden_states: torch.Tensor = layer_module.forward(
                        hidden_states=hidden_states,
                        seq_idx=seq_idx,
                        timestep=timestep,
                        cu_seqlens=cu_seqlens,
                        seq_len=seq_len,
                        subset_idx=None,
                        indices=indices,
                        attn_mask=attn_mask,
                        param_update=param_updates[i]
                    )

                # Add current layer hidden states to all_encoder_layers if needed
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)

            hidden_states: torch.Tensor = self.layer[-1].forward(
                hidden_states=hidden_states,
                seq_idx=seq_idx,
                timestep=timestep,
                cu_seqlens=cu_seqlens,
                seq_len=seq_len,
                subset_idx=None,
                indices=indices,
                attn_mask=attn_mask,
                param_update=param_updates[-1]
            )

            if self.is_prenorm:
                hidden_states: torch.Tensor = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))

            hidden_states = padding.pad_input(
                hidden_states=hidden_states, indices=indices, batch_size=batch_size, seq_len=seq_len
            )

        # Add the last layer's hidden state if output_all_encoded_layers is False
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers

# This class is optional and can be used for pooling the output of the Hydra model.
# It is not always necessary, depending on the task. Mostly used for classification tasks or sentence regression and GLUE tasks.   
class HydraPooler(nn.Module):

    def __init__(
            self,
            config=None,
            activation: str=None
    ):
        super().__init__()

        factory_kwargs = {"device": config.device, "dtype": config.dtype}

        self.dense: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size, **factory_kwargs)

        self.act: nn.Module = nn.Tanh() if activation is None else ACT2FN[activation]

        self.pool_all: bool = config.pool_all

    def forward(
            self,
            hidden_states: torch.Tensor,
            pool: bool = True,
            mask: torch.Tensor = None
    ) -> torch.Tensor:
        
        if not self.pool_all:

            first_token_tensor: torch.Tensor = hidden_states[:, 0] if pool  else hidden_states
            pooled_output: torch.Tensor = self.dense(first_token_tensor)
            pooled_output: torch.Tensor = self.act(pooled_output)

        else:

            denom = torch.sum(mask, dim=1, keepdim=True)
            mean_tensor = torch.sum((hidden_states) * mask.unsqueeze(-1), dim=1) / denom
            pooled_output: torch.Tensor = self.dense(mean_tensor)
            pooled_output: torch.Tensor = self.act(pooled_output)

        return pooled_output