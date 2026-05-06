import torch

import logging
import os
import subprocess

from .hydra_model import HydraForMaskedLM, HydraForMaskedLMConfig
from .utils import log_setup, load_config

torch.manual_seed(42)

if not os.path.exists("logs"):
    os.makedirs("logs", exist_ok=True)
    subprocess.run(["touch", "logs/transfer.log"])

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "transfer.log")
LOG_LEVEL = logging.INFO

TRAIN_CONFIG = load_config("configs/training_config_dct_stage1.yaml", dict_config=True)

logger: logging.Logger = log_setup("TransferLogger", LOG_FILE, LOG_LEVEL) 

# Load Hydra BERT model weights

logger.debug("Loading Hydra BERT model weights...")
model_dict = torch.load("models/hydra_bert_23layers.pt", weights_only=False)
model_dict = model_dict["state"]["model"]

logger.info("Hydra BERT model weights loaded successfully.")

config = load_config("configs/training_config_dct_stage1.yaml", pretrained_config=True)

logger.info("Hydra configuration loaded successfully.")

# Initialize a Hydra model class for the task of masked language modeling.
model: HydraForMaskedLM = HydraForMaskedLM(config=config)
model.config_class = HydraForMaskedLMConfig

logger.info("Hydra model initialized successfully.")

## Transfer the weights from the Hydra BERT model to the Hydra model.

# Transfer the embedding layer weights.

# Assert the shapes of the embedding layer norm weights.
assert model.hydra.embeddings.LayerNorm.weight.data.shape == model_dict["model.bert.embeddings.LayerNorm.weight"].shape
assert model.hydra.embeddings.LayerNorm.bias.data.shape == model_dict["model.bert.embeddings.LayerNorm.bias"].shape

# Transfer the embedding layer norm weights.
model.hydra.embeddings.LayerNorm.weight.data = model_dict["model.bert.embeddings.LayerNorm.weight"]
model.hydra.embeddings.LayerNorm.bias.data = model_dict["model.bert.embeddings.LayerNorm.bias"]

# Assert the shapes of the word and token type embeddings weights.
assert model.hydra.embeddings.word_embeddings.weight.data.shape == model_dict["model.bert.embeddings.word_embeddings.weight"][:30522, :].shape
assert model.hydra.embeddings.token_type_embeddings.weight.data.shape == model_dict["model.bert.embeddings.token_type_embeddings.weight"].shape

# Transfer the word and token type embeddings weights.
model.hydra.embeddings.word_embeddings.weight.data = model_dict["model.bert.embeddings.word_embeddings.weight"][:30522, :]
model.hydra.embeddings.token_type_embeddings.weight.data = model_dict["model.bert.embeddings.token_type_embeddings.weight"]

logger.info("Embedding layer weights transferred successfully.")

# Transfer the encoder layer weights.
for layer_idx in range(0, 23):

    # Extract the input_proj and head dimensions from the model
    in_proj_dim: int = model.hydra.encoder.layer[layer_idx].layer.mixer.in_proj.weight.data.shape[0]
    head_dim: int = model.hydra.encoder.layer[layer_idx].layer.mixer.A_log.data.shape[0]

    # Assert the shapes of the convolutional layer weights and biases match.
    assert model.hydra.encoder.layer[layer_idx].layer.mixer.conv1d.bias.data.shape == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.conv1d.bias"].shape
    assert model.hydra.encoder.layer[layer_idx].layer.mixer.conv1d.weight.data.shape == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.conv1d.weight"].shape

    # Transfer the convolution layer weights and biases.
    model.hydra.encoder.layer[layer_idx].layer.mixer.conv1d.bias.data = model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.conv1d.bias"]
    model.hydra.encoder.layer[layer_idx].layer.mixer.conv1d.weight.data = model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.conv1d.weight"]

    # Assert and transfer the in_proj weights using truncation to prune the extra weights.
    assert model.hydra.encoder.layer[layer_idx].layer.mixer.in_proj.weight.data.shape == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.in_proj.weight"][:in_proj_dim, :].shape
    model.hydra.encoder.layer[layer_idx].layer.mixer.in_proj.weight.data = model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.in_proj.weight"][:in_proj_dim, :]

    # Assert and transfer the A_log, D, dt_bias, fc_D.weight, norm.weight, out_proj.weight, and layer norm weights.
    assert model.hydra.encoder.layer[layer_idx].layer.mixer.A_log.data.shape == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.A_log"][:head_dim].shape
    model.hydra.encoder.layer[layer_idx].layer.mixer.A_log.data = model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.A_log"][:head_dim]

    model.hydra.encoder.layer[layer_idx].layer.mixer.D.data = model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.D"][:head_dim]

    model.hydra.encoder.layer[layer_idx].layer.mixer.dt_bias.data = model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.dt_bias"][:head_dim]

    assert model.hydra.encoder.layer[layer_idx].layer.mixer.fc_D.weight.data.shape == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.fc_D.weight"][:head_dim, :].shape
    model.hydra.encoder.layer[layer_idx].layer.mixer.fc_D.weight.data = model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.fc_D.weight"][:head_dim, :]

    assert model.hydra.encoder.layer[layer_idx].layer.mixer.norm.weight.data.shape == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.norm.weight"].shape
    model.hydra.encoder.layer[layer_idx].layer.mixer.norm.weight.data = model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.norm.weight"]

    assert model.hydra.encoder.layer[layer_idx].layer.mixer.out_proj.weight.data.shape == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.out_proj.weight"].shape
    model.hydra.encoder.layer[layer_idx].layer.mixer.out_proj.weight.data = model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.out_proj.weight"]

    assert model.hydra.encoder.layer[layer_idx].layer.norm.bias.data.shape == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.norm.bias"].shape
    assert model.hydra.encoder.layer[layer_idx].layer.norm.weight.data.shape == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.norm.weight"].shape
    
    model.hydra.encoder.layer[layer_idx].layer.norm.bias.data = model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.norm.bias"]
    model.hydra.encoder.layer[layer_idx].layer.norm.weight.data = model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.norm.weight"]

logger.info("Encoder layer weights transferred successfully.")

# Assert and Transfer the prediction head weights.
vocab_size = model.cls.predictions.decoder.bias.data.shape[0]

assert model.cls.predictions.decoder.bias.data.shape == model_dict["model.cls.predictions.decoder.bias"][:vocab_size].shape
assert model.cls.predictions.decoder.weight.data.shape == model_dict["model.cls.predictions.decoder.weight"][:vocab_size, :].shape

model.cls.predictions.decoder.bias.data = model_dict["model.cls.predictions.decoder.bias"][:vocab_size]
model.cls.predictions.decoder.weight.data = model_dict["model.cls.predictions.decoder.weight"][:vocab_size, :]

logger.info("Prediction head weights transferred successfully.")

# Assert and Transfer the prediction transform layer norm weights and biases

assert model.cls.predictions.transform.LayerNorm.bias.data.shape == model_dict["model.cls.predictions.transform.LayerNorm.bias"].shape
assert model.cls.predictions.transform.LayerNorm.weight.data.shape == model_dict["model.cls.predictions.transform.LayerNorm.weight"].shape

model.cls.predictions.transform.LayerNorm.bias.data = model_dict["model.cls.predictions.transform.LayerNorm.bias"]
model.cls.predictions.transform.LayerNorm.weight.data = model_dict["model.cls.predictions.transform.LayerNorm.weight"]

# Assert and Transfer the prediction transform dense layer weights and biases

assert model.cls.predictions.transform.dense.bias.data.shape == model_dict["model.cls.predictions.transform.dense.bias"].shape
assert model.cls.predictions.transform.dense.weight.data.shape == model_dict["model.cls.predictions.transform.dense.weight"].shape

model.cls.predictions.transform.dense.bias.data = model_dict["model.cls.predictions.transform.dense.bias"]
model.cls.predictions.transform.dense.weight.data = model_dict["model.cls.predictions.transform.dense.weight"]

logger.info("Prediction transform layer weights transferred successfully.")

logger.info("All model parameters have been transferred successfully!")

torch.save(model.state_dict(), TRAIN_CONFIG["weights_path"])

logger.info(f"Hydra model state dict saved successfully to: {TRAIN_CONFIG['weights_path']}")


# save_dir = TRAIN_CONFIG["weights_path"]
# os.makedirs(save_dir, exist_ok=True)
# model.save_pretrained(save_directory=save_dir, safe_serialization=False)

# logger.info(f"Hydra model and config saved successfully to: {save_dir}")

# model: GuiderCore = GuiderCore(config=config)

# logger.info("Guider model initialized successfully.")

# ## Transfer the weights from the Hydra BERT model to the Guider model.

# # Assert and transfer the shapes of the embedding layer norm weights
# assert model.embeddings.LayerNorm.weight.data.shape == model_dict["model.bert.embeddings.LayerNorm.weight"].shape
# assert model.embeddings.LayerNorm.bias.data.shape == model_dict["model.bert.embeddings.LayerNorm.bias"].shape

# model.embeddings.LayerNorm.weight.data = model_dict["model.bert.embeddings.LayerNorm.weight"]
# model.embeddings.LayerNorm.bias.data = model_dict["model.bert.embeddings.LayerNorm.bias"]

# # Assert and Transfer the word and token type embeddings weights.
# assert model.embeddings.word_embeddings.weight.data.shape == model_dict["model.bert.embeddings.word_embeddings.weight"][:vocab_size, :].shape
# assert model.embeddings.token_type_embeddings.weight.data.shape == model_dict["model.bert.embeddings.token_type_embeddings.weight"][:vocab_size, :].shape

# model.embeddings.word_embeddings.weight.data = model_dict["model.bert.embeddings.word_embeddings.weight"][:vocab_size, :]
# model.embeddings.token_type_embeddings.weight.data = model_dict["model.bert.embeddings.token_type_embeddings.weight"][:vocab_size, :]

# logger.info("Guider embedding layer weights transferred successfully.")

# # Transfer the encoder layer weights to Guider.
# for layer_idx in range(0, 12):

#     conv = model.encoder.layer[layer_idx].layer.mixer.conv1d
#     out_c, in_c, k = conv.weight.shape

#     weights = model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.conv1d.weight"]
#     bias = model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.conv1d.bias"]

    
#     assert conv.weight.data[:out_c, :in_c, :].shape == weights[:out_c, :in_c, :].shape
#     assert conv.bias.data[:out_c].shape == bias[:out_c].shape

#     model.encoder.layer[layer_idx].layer.mixer.conv1d.weight.data[:out_c, :in_c, :] = weights[:out_c, :in_c, :]
#     model.encoder.layer[layer_idx].layer.mixer.conv1d.bias.data[:out_c] = bias[:out_c]

#     idx = model.encoder.layer[layer_idx].layer.mixer.in_proj.weight.data.shape

#     assert model.encoder.layer[layer_idx].layer.mixer.in_proj.weight.data.shape == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.in_proj.weight"][slice(0, idx[0])].shape
#     model.encoder.layer[layer_idx].layer.mixer.in_proj.weight.data = model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.in_proj.weight"][slice(0, idx[0])]

#     n_heads = model.encoder.layer[layer_idx].layer.mixer.A_log.data.shape[0]

#     assert model.encoder.layer[layer_idx].layer.mixer.A_log.data.shape == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.A_log"][:n_heads].shape
#     model.encoder.layer[layer_idx].layer.mixer.A_log.data = model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.A_log"][:n_heads]

#     assert model.encoder.layer[layer_idx].layer.mixer.D.data.shape == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.D"][:n_heads].shape
#     model.encoder.layer[layer_idx].layer.mixer.D.data = model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.D"][:n_heads]

#     assert model.encoder.layer[layer_idx].layer.mixer.dt_bias.data.shape == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.dt_bias"][:n_heads].shape
#     model.encoder.layer[layer_idx].layer.mixer.dt_bias.data = model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.dt_bias"][:n_heads]

#     idx = model.encoder.layer[layer_idx].layer.mixer.out_proj.weight.data.shape

#     assert model.encoder.layer[layer_idx].layer.mixer.out_proj.weight.data.shape == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.out_proj.weight"][slice(0, idx[0])].shape
#     model.encoder.layer[layer_idx].layer.mixer.out_proj.weight.data = model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.out_proj.weight"][slice(0, idx[0])]

# logger.info("Guider encoder layer weights transferred successfully.")

# # Decoder layer weights not transferred as they are different in Guider.

# # Assert and Transfer the prediction head weights from the transform layer.

# assert model.prediction.transform.LayerNorm.bias.data.shape == model_dict["model.cls.predictions.transform.LayerNorm.bias"].shape
# assert model.prediction.transform.LayerNorm.weight.data.shape == model_dict["model.cls.predictions.transform.LayerNorm.weight"].shape

# model.prediction.transform.LayerNorm.bias.data = model_dict["model.cls.predictions.transform.LayerNorm.bias"]
# model.prediction.transform.LayerNorm.weight.data = model_dict["model.cls.predictions.transform.LayerNorm.weight"]

# # Assert and transfer the prediction transform dense layer weights and biases.

# assert model.prediction.transform.dense.bias.data.shape == model_dict["model.cls.predictions.transform.dense.bias"].shape
# assert model.prediction.transform.dense.weight.data.shape == model_dict["model.cls.predictions.transform.dense.weight"].shape

# model.prediction.transform.dense.bias.data = model_dict["model.cls.predictions.transform.dense.bias"]
# model.prediction.transform.dense.weight.data = model_dict["model.cls.predictions.transform.dense.weight"]

# logger.info("Guider prediction transform layer weights transferred successfully.")

# logger.info("All Guider model parameters have been transferred successfully!")

# torch.save(model.state_dict(), "models/hydra_guider_12layers.pt")

# logger.info("Guider model state dict saved successfully to: models/hydra_guider_12layers.pt")