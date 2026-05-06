# Trigger HuggingFace Auto class registration on package import.
# The register() calls live at the bottom of hydra_model.py and execute on import.
from .hydra_model import HydraForMaskedLMConfig, HydraModel, HydraForMaskedLM  # noqa: F401
