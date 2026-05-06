"""
    Hydra heads for language modeling and next sentence prediction tasks.
"""

import torch
import torch.nn as nn

from transformers.activations import ACT2FN

class HydraGuiderHead(nn.Module):

    """
        Hydra head for guiding the model's attention and predictions.
    """
    
    def __init__(
        self,
        config=None,
    ):
        super().__init__()

        self.transform: HydraPredictionHeadTransform = HydraPredictionHeadTransform(config=config)

        self.decoder = nn.Linear(config.hidden_size, 1, device=config.device)

        self.activation = nn.Sigmoid()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        
        transformed_states: torch.Tensor = self.transform(hidden_states)

        logits: torch.Tensor = self.decoder(transformed_states)

        prediction_scores: torch.Tensor = self.activation(logits)

        return prediction_scores

class HydraPredictionHeadTransform(nn.Module):

    """
        A last stage transformation layer for the Hydra model's prediction head meant to be used internally.
        It applies a linear transformation, an activation function, and a layer normalization on the hidden states from the Encoder.

        Args:
            d_model (int): The dimensionality of the model.

            device (torch.device): The device on which the model will be loaded.

            dtype (torch.dtype): The data type of the model parameters.

            hidden_act (str | nn.Module): The activation function to be applied. It can be a string representing the activation function name or an instance of an activation function module.
    """

    def __init__(
        self,
        config=None,
    ):
        super().__init__()

        factory_kwargs = {'device': config.device}

        self.dense = nn.Linear(config.hidden_size, config.hidden_size, **factory_kwargs)

        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act

        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=1e-12, **factory_kwargs
        )

    def forward(
            self,
            d_model: torch.Tensor
    ) -> torch.Tensor:
        
        """
        Forward pass of the transformation layer. Applies a linear transformation, an activation function, and layer normalization to the input hidden states.

        Args:
            d_model (torch.Tensor): The input hidden states from the Encoder.

        Returns:
            torch.Tensor: The transformed hidden states after applying the linear transformation, activation function, and layer normalization.
        """
        
        hidden_states: torch.Tensor = self.dense(d_model)
        hidden_states: torch.Tensor = self.transform_act_fn(hidden_states)
        hidden_states: torch.Tensor = self.LayerNorm(hidden_states)

        return hidden_states
    
class HydraLMPredictionHead(nn.Module):

    """
        A prediction head for the Hydra model used for masked language modeling tasks.
        It applies the final transformations (linear transformation, activation function, and layer normalization) to the hidden states from the Encoder,
        and then projects the transformed hidden states to the vocabulary size to obtain the prediction scores.
        It also computes the loss if labels are provided.

        Args:
            d_model (int): The dimensionality of the model.

            device (torch.device): The device on which the model will be loaded.

            dtype (torch.dtype): The data type of the model parameters.

            hidden_act (str | nn.Module): The activation function to be applied. It can be a string representing the activation function name or an instance of an activation function module.

            hydra_model_embedding_weights (torch.Tensor | nn.Module): The weights of the Hydra model's embedding layer, used for the final linear projection.
    """

    def __init__(
        self,
        hydra_model_embedding_weights: torch.Tensor | nn.Module = None,
        config=None
    ):
        super().__init__()
        
        self.transform = HydraPredictionHeadTransform(config=config)

        self.decoder: nn.Linear = nn.Linear(
            hydra_model_embedding_weights.size(1),
            hydra_model_embedding_weights.size(0),
            device=config.device
        )

        self.decoder.weight = hydra_model_embedding_weights

    def forward(
            self,
            hidden_states: torch.Tensor
    ) -> torch.Tensor:
        
        """
        Forward pass of the prediction head. Applies the transformation layer to the hidden states and then projects them to the vocabulary size.
        Transformation includes a linear transformation, an activation function, and layer normalization.

        Args:
            hidden_states (torch.Tensor): The input hidden states from the Encoder.

        Returns:
            torch.Tensor: The output prediction scores for the masked language modeling task.
        """
        
        hidden_states: torch.Tensor = self.transform(hidden_states)
        hidden_states: torch.Tensor = self.decoder(hidden_states)

        return hidden_states
    
class HydraOnlyMLMHead(nn.Module):

    """
        A Hydra model's head used for masked language modeling tasks.
        A wrapper around the HydraLMPredictionHead that initializes it with the necessary parameters.
        Applies the final transformations to the hidden states from the Encoder and projects them to the vocabulary size to ouptut prediction scores.

        Args:
            d_model (int): The dimensionality of the model.

            device (torch.device): The device on which the model will be loaded.

            dtype (torch.dtype): The data type of the model parameters.

            hidden_act (str | nn.Module): The activation function to be applied. It can be a string representing the activation function name or an instance of an activation function module.
            
            hydra_model_embedding_weights (torch.Tensor | nn.Module): The weights of the Hydra model's embedding layer, used for the final linear projection.
    """
    
    def __init__(
        self,
        hydra_model_embedding_weights: torch.Tensor | nn.Module = None,
        config=None
    ):
        
        super().__init__()

        self.predictions = HydraLMPredictionHead(
            config=config,
            hydra_model_embedding_weights=hydra_model_embedding_weights
        )

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:

        """
        Forward pass of the HydraOnlyMLMHead. Applies the prediction head to the sequence output from the Encoder to obtain prediction scores.

        Returns:
            torch.Tensor: The output prediction scores for the masked language modeling task. Shape: (batch_size, seq_len, d_model)
        """
        
        prediction_scores: torch.Tensor = self.predictions(sequence_output)

        return prediction_scores