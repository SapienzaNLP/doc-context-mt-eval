"""Response generation using vLLM — supports Qwen3 and Gemma 4."""

import logging
from typing import List, Tuple
from vllm import SamplingParams

from config_evaluation import ModelConfig, SeedConfig
from model_adapter import ModelProfile, parse_model_response

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generate responses using vLLM for any supported model family."""

    def __init__(
        self,
        model_config: ModelConfig,
        seed_config: SeedConfig,
        profile: ModelProfile,
    ):
        self.model_config = model_config
        self.seed_config = seed_config
        self.profile = profile

    def generate(
        self,
        prompts: List[str],
        llm,
        enable_thinking: bool,
    ) -> List[Tuple[str, str]]:
        """
        Generate responses for a list of prompts using vLLM.

        Args:
            prompts:         List of fully-formatted prompt strings.
            llm:             vLLM LLM instance.
            enable_thinking: Whether thinking mode is active.

        Returns:
            List of (thinking_content, final_response) tuples.
        """
        if not prompts:
            logger.warning("Empty prompts list provided.")
            return []

        assert all(isinstance(p, str) and p.strip() for p in prompts), \
            "All prompts must be non-empty strings."

        sampling_params = self._get_sampling_params(enable_thinking)

        logger.info(
            "Generating %d responses | model_family=%s | thinking=%s",
            len(prompts), self.profile.family, enable_thinking,
        )
        outputs = llm.generate(prompts, sampling_params)

        return [
            parse_model_response(
                output.outputs[0].text.strip(),
                enable_thinking,
                self.profile,
            )
            for output in outputs
        ]

    def _get_sampling_params(self, enable_thinking: bool) -> SamplingParams:
        """
        Build SamplingParams.

        Thinking mode:
          Uses the per-family recommended values stored in ModelProfile,
          which can still be overridden by the user via ModelConfig fields.

        Non-thinking mode:
          Greedy decoding (temperature=0) — same for all families.
        """
        if enable_thinking:
            # ModelConfig fields take priority; profile values are the defaults.
            temperature = getattr(
                self.model_config, "thinking_temperature",
                self.profile.thinking_temperature,
            )
            top_p = getattr(
                self.model_config, "thinking_top_p",
                self.profile.thinking_top_p,
            )
            top_k = getattr(
                self.model_config, "thinking_top_k",
                self.profile.thinking_top_k,
            )
            min_p = getattr(
                self.model_config, "thinking_min_p",
                self.profile.thinking_min_p,
            )
            return SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=self.model_config.max_new_tokens,
                seed=self.seed_config.seed,
            )
        else:
            return SamplingParams(
                temperature=self.model_config.normal_temperature,
                max_tokens=self.model_config.max_new_tokens,
                seed=self.seed_config.seed,
            )