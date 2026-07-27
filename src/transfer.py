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

logger: logging.Logger = log_setup("TransferLogger", LOG_FILE, LOG_LEVEL)


def transfer_weights(
    source_path: str = "models/hydra_bert_23layers.pt",
    config_path: str = "configs/benchmark_config_chebyshev_stage1.yaml",
    output_path: str = "models/hydra_bert_23layers_mark_base.pt",
) -> str:
    """Transfer Hydra BERT weights into a HydraForMaskedLM model with MaRK config.

    Args:
        source_path: Path to the source ``hydra_bert_23layers.pt`` checkpoint.
        config_path: Path to a benchmark/training config YAML (any kernel works —
                     the remapped weights are kernel-agnostic).
        output_path: Where to save the remapped ``state_dict``.

    Returns:
        str: The output path.
    """
    logger.info(f"Loading source weights from: {source_path}")
    model_dict = torch.load(source_path, weights_only=False)
    model_dict = model_dict["state"]["model"]

    logger.info(f"Loading config from: {config_path}")
    config = load_config(config_path, pretrained_config=True)

    # Initialize a Hydra model class for the task of masked language modeling.
    model: HydraForMaskedLM = HydraForMaskedLM(config=config)
    model.config_class = HydraForMaskedLMConfig

    logger.info("Hydra model initialized successfully.")

    # ---- Transfer the embedding layer weights ----
    assert model.hydra.embeddings.LayerNorm.weight.data.shape == model_dict["model.bert.embeddings.LayerNorm.weight"].shape
    assert model.hydra.embeddings.LayerNorm.bias.data.shape == model_dict["model.bert.embeddings.LayerNorm.bias"].shape

    model.hydra.embeddings.LayerNorm.weight.data = model_dict["model.bert.embeddings.LayerNorm.weight"]
    model.hydra.embeddings.LayerNorm.bias.data = model_dict["model.bert.embeddings.LayerNorm.bias"]

    assert model.hydra.embeddings.word_embeddings.weight.data.shape == model_dict["model.bert.embeddings.word_embeddings.weight"][:30522, :].shape
    assert model.hydra.embeddings.token_type_embeddings.weight.data.shape == model_dict["model.bert.embeddings.token_type_embeddings.weight"].shape

    model.hydra.embeddings.word_embeddings.weight.data = model_dict["model.bert.embeddings.word_embeddings.weight"][:30522, :]
    model.hydra.embeddings.token_type_embeddings.weight.data = model_dict["model.bert.embeddings.token_type_embeddings.weight"]

    logger.info("Embedding layer weights transferred successfully.")

    # ---- Transfer the encoder layer weights ----
    for layer_idx in range(0, 23):
        in_proj_dim: int = model.hydra.encoder.layer[layer_idx].layer.mixer.in_proj.weight.data.shape[0]
        head_dim: int = model.hydra.encoder.layer[layer_idx].layer.mixer.A_log.data.shape[0]

        assert model.hydra.encoder.layer[layer_idx].layer.mixer.conv1d.bias.data.shape == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.conv1d.bias"].shape
        assert model.hydra.encoder.layer[layer_idx].layer.mixer.conv1d.weight.data.shape == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.conv1d.weight"].shape

        model.hydra.encoder.layer[layer_idx].layer.mixer.conv1d.bias.data = model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.conv1d.bias"]
        model.hydra.encoder.layer[layer_idx].layer.mixer.conv1d.weight.data = model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.conv1d.weight"]

        assert model.hydra.encoder.layer[layer_idx].layer.mixer.in_proj.weight.data.shape == model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.in_proj.weight"][:in_proj_dim, :].shape
        model.hydra.encoder.layer[layer_idx].layer.mixer.in_proj.weight.data = model_dict[f"model.bert.encoder.layer.{layer_idx}.layer.mixer.in_proj.weight"][:in_proj_dim, :]

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

    # ---- Transfer the prediction head weights ----
    vocab_size = model.cls.predictions.decoder.bias.data.shape[0]

    assert model.cls.predictions.decoder.bias.data.shape == model_dict["model.cls.predictions.decoder.bias"][:vocab_size].shape
    assert model.cls.predictions.decoder.weight.data.shape == model_dict["model.cls.predictions.decoder.weight"][:vocab_size, :].shape

    model.cls.predictions.decoder.bias.data = model_dict["model.cls.predictions.decoder.bias"][:vocab_size]
    model.cls.predictions.decoder.weight.data = model_dict["model.cls.predictions.decoder.weight"][:vocab_size, :]

    logger.info("Prediction head weights transferred successfully.")

    # ---- Transfer the prediction transform layer weights ----
    assert model.cls.predictions.transform.LayerNorm.bias.data.shape == model_dict["model.cls.predictions.transform.LayerNorm.bias"].shape
    assert model.cls.predictions.transform.LayerNorm.weight.data.shape == model_dict["model.cls.predictions.transform.LayerNorm.weight"].shape

    model.cls.predictions.transform.LayerNorm.bias.data = model_dict["model.cls.predictions.transform.LayerNorm.bias"]
    model.cls.predictions.transform.LayerNorm.weight.data = model_dict["model.cls.predictions.transform.LayerNorm.weight"]

    assert model.cls.predictions.transform.dense.bias.data.shape == model_dict["model.cls.predictions.transform.dense.bias"].shape
    assert model.cls.predictions.transform.dense.weight.data.shape == model_dict["model.cls.predictions.transform.dense.weight"].shape

    model.cls.predictions.transform.dense.bias.data = model_dict["model.cls.predictions.transform.dense.bias"]
    model.cls.predictions.transform.dense.weight.data = model_dict["model.cls.predictions.transform.dense.weight"]

    logger.info("Prediction transform layer weights transferred successfully.")
    logger.info("All model parameters have been transferred successfully!")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    logger.info(f"Hydra model state dict saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transfer hydra_bert_23layers.pt into a HydraForMaskedLM state_dict"
    )
    parser.add_argument(
        "--source",
        default="models/hydra_bert_23layers.pt",
        help="Path to source hydra_bert_23layers.pt",
    )
    parser.add_argument(
        "--config",
        default="configs/benchmark_config_chebyshev_stage1.yaml",
        help="Any benchmark config YAML (provides model architecture)",
    )
    parser.add_argument(
        "--output",
        default="models/hydra_bert_23layers_mark_base.pt",
        help="Output path for the remapped state_dict",
    )
    args = parser.parse_args()
    transfer_weights(args.source, args.config, args.output)
