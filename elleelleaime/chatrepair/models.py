from importlib import import_module
from typing import Any


MODEL_SPECS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "codellama-instruct": (
        "elleelleaime.generate.strategies.models.huggingface.codellama.codellama_instruct",
        "CodeLLaMAIntruct",
        ("model_name",),
    ),
    "codellama-infilling": (
        "elleelleaime.generate.strategies.models.huggingface.codellama.codellama_infilling",
        "CodeLLaMAInfilling",
        ("model_name",),
    ),
    "repairllama-infilling": (
        "elleelleaime.generate.strategies.models.huggingface.repairllama.repairllama_infilling",
        "RepairLLaMAInfilling",
        ("model_name",),
    ),
    "codet5-small_infilling": (
        "elleelleaime.generate.strategies.models.huggingface.codet5.codet5small_infilling",
        "CodeT5PatchGeneration",
        ("model_name",),
    ),
    "openai-chatcompletion": (
        "elleelleaime.generate.strategies.models.openai.openai",
        "OpenAIChatCompletionModels",
        ("model_name",),
    ),
    "openrouter": (
        "elleelleaime.generate.strategies.models.openrouter.openrouter",
        "OpenRouterModels",
        ("model_name",),
    ),
    "google": (
        "elleelleaime.generate.strategies.models.google.google",
        "GoogleModels",
        ("model_name",),
    ),
    "anthropic": (
        "elleelleaime.generate.strategies.models.anthropic.anthropic",
        "AnthropicModels",
        ("model_name", "max_tokens"),
    ),
    "mistral": (
        "elleelleaime.generate.strategies.models.mistral.mistral",
        "MistralModels",
        ("model_name",),
    ),
    "litellm-chatcompletion": (
        "elleelleaime.generate.strategies.models.litellm.litellm",
        "LiteLLMChatCompletionModels",
        tuple(),
    ),
}


def _normalize_model_kwargs(strategy_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(kwargs)

    if (
        "num_return_sequences" in normalized
        and "n_samples" not in normalized
        and strategy_name in {
            "openai-chatcompletion",
            "openrouter",
            "google",
            "anthropic",
            "mistral",
        }
    ):
        normalized["n_samples"] = normalized["num_return_sequences"]

    if strategy_name == "litellm-chatcompletion":
        if "model" not in normalized and "model_name" in normalized:
            normalized["model"] = normalized["model_name"]

    return normalized


class FrameworkChatModel:
    def __init__(self, strategy_name: str, **kwargs) -> None:
        self.strategy_name = strategy_name.lower().strip()
        if self.strategy_name not in MODEL_SPECS:
            raise ValueError(f"Unknown ChatRepair strategy {strategy_name}")

        module_name, class_name, required_args = MODEL_SPECS[self.strategy_name]
        kwargs = _normalize_model_kwargs(self.strategy_name, kwargs)
        for required_arg in required_args:
            if required_arg not in kwargs:
                raise ValueError(
                    f"Missing argument {required_arg} for strategy {strategy_name}"
                )

        strategy_class = getattr(import_module(module_name), class_name)
        self.strategy = strategy_class(**kwargs)

    def complete(self, messages: list[dict[str, str]]) -> tuple[str, Any]:
        prompt = self._flatten_messages(messages)
        raw_generation = self.strategy.generate([prompt])[0]
        return self._extract_text(raw_generation), raw_generation

    def _flatten_messages(self, messages: list[dict[str, str]]) -> str:
        transcript = []
        for message in messages:
            role = message["role"].capitalize()
            transcript.append(f"{role}:\n{message['content'].strip()}\n")
        transcript.append("Assistant:\n")
        return "\n".join(transcript)

    def _extract_text(self, generation: Any) -> str:
        if generation is None:
            return ""

        if isinstance(generation, list):
            first = generation[0] if generation else ""
            return first if isinstance(first, str) else self._extract_text(first)

        if isinstance(generation, str):
            return generation

        if self.strategy_name in {
            "openai-chatcompletion",
            "openrouter",
            "mistral",
            "litellm-chatcompletion",
        }:
            return generation["choices"][0]["message"]["content"]

        if self.strategy_name == "google":
            return generation["candidates"][0]["content"]["parts"][0]["text"]

        if self.strategy_name == "anthropic":
            return generation["content"][0]["text"]

        raise ValueError(f"Unsupported generation payload for {self.strategy_name}")
