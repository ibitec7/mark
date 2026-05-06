import os
import logging

import torch
from deepeval.models.base_model import DeepEvalBaseLLM
from transformers import BertTokenizerFast

from .hydra_model import HydraForMaskedLM
from .mamba_guider import MambaGuiderScorer
from .utils import load_config, load_hydra_state_dict, load_compatible_state_dict, log_setup

LOG_FILE = os.path.join("logs", "eval.log")
LOG_LEVEL = logging.INFO
os.makedirs("logs", exist_ok=True)
logger = log_setup("EvalLogger", LOG_FILE, LOG_LEVEL)


class EvalModel(DeepEvalBaseLLM):
    """DeepEval wrapper for HydraForMaskedLM with MaRK adapters."""

    def __init__(
        self,
        model: HydraForMaskedLM | None = None,
        weights_path: str | None = None,
        config_path: str = "configs/hydra.yaml",
        kernel: str = "chebyshev",
        device: str = "cuda",
        guider_scorer: MambaGuiderScorer | None = None,
    ):
        """
        Initialize the evaluation model.

        Args:
            model (HydraForMaskedLM | None): Pre-loaded model instance.
            weights_path (str | None): Path to ``.pt`` or Lightning ``.ckpt`` if model is None.
            config_path (str): Path to model config YAML.
            kernel (str): MaRK kernel variant ("hypernet", "chebyshev", "dct").
            device (str): Target device.
            guider_scorer: Optional Mamba2 guider for remasking during ``inference()``.
        """
        if model is not None:
            self.model = model
        elif weights_path is not None:
            config = load_config(config_path, pretrained_config=False)
            config.mark_kernel = kernel
            self.model = HydraForMaskedLM(config=config)
            weights = load_hydra_state_dict(weights_path)
            missing, unexpected, mismatched = load_compatible_state_dict(self.model, weights)
            if missing:
                logger.warning(f"Missing keys: {missing[:5]}")
            if unexpected:
                logger.warning(f"Unexpected keys: {unexpected[:5]}")
            if mismatched:
                logger.warning(f"Skipped mismatched tensors: {mismatched[:5]}")
            del weights
        else:
            raise ValueError("Provide either model or weights_path")

        self.model = self.model.to(device)
        self.model.eval()
        self._device = device
        self._guider_scorer = guider_scorer

        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self._name = f"HydraMaRK-{kernel}"

    def load_model(self) -> HydraForMaskedLM:
        """
        Return the loaded model (required by DeepEvalBaseLLM).

        Returns:
            HydraForMaskedLM: The loaded model.
        """
        return self.model

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response for a given prompt using iterative decoding.

        Args:
            prompt (str): The input text prompt.
            **kwargs: Forwarded to model.inference() (seq_len, sampling_steps).

        Returns:
            str: Decoded response text.
        """
        prompt_tokens: list = self.tokenizer.encode(prompt, add_special_tokens=True)

        seq_len = kwargs.pop("seq_len", 128)
        sampling_steps = kwargs.pop("sampling_steps", 30)
        guider_scorer = kwargs.pop("guider_scorer", self._guider_scorer)

        response: torch.Tensor = self.model.inference(
            prompt=prompt_tokens,
            seq_len=seq_len,
            sampling_steps=sampling_steps,
            guider_scorer=guider_scorer,
        )

        response_str: str = self.tokenizer.decode(
            response.squeeze().tolist(),
            skip_special_tokens=True,
        )

        return response_str

    async def a_generate(self, prompt: str, **kwargs) -> str:
        """
        Async generation (required by DeepEvalBaseLLM).

        Args:
            prompt (str): The input text prompt.
            **kwargs: Forwarded to generate().

        Returns:
            str: Decoded response text.
        """
        return self.generate(prompt, **kwargs)

    def batch_generate(self, prompts: list[str], **kwargs) -> list[str]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts (list[str]): List of input text prompts.
            **kwargs: Forwarded to generate().

        Returns:
            list[str]: List of decoded response texts.
        """
        return [self.generate(p, **kwargs) for p in prompts]

    def get_model_name(self) -> str:
        """
        Return the model name (required by DeepEvalBaseLLM).

        Returns:
            str: Model identifier string.
        """
        return self._name