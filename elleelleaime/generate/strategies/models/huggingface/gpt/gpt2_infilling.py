from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional
import logging
import threading

import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from elleelleaime.generate.strategies.strategy import PatchGenerationStrategy


@dataclass
class GenerateSettings:
    name: str
    do_sample: bool = True
    temperature: float = 0.8
    num_beams: int = 1
    num_return_sequences: int = 1
    max_length: int = 1024       # GPT-2 max context
    top_p: float = 0.95
    max_new_tokens: int = 64
    early_stopping: bool = True
    seed: int = 0


class GPT2PatchGeneration(PatchGenerationStrategy):
    """
    GPT-2-large generation strategy for APR.

    IMPORTANT:
    - GPT-2 is NOT a true infilling model.
    - If prompt contains: prefix <FILL_ME> suffix
      generation only uses `prefix`, then inserts completion into <FILL_ME>.
    - The suffix is ignored during generation and only matters later in evaluation.
    """

    __GENERATION_STRATEGIES = {
        "beam_search": GenerateSettings(name="beam_search", early_stopping=True),
        "sampling": GenerateSettings(name="sampling", do_sample=True),
    }

    __MODEL: Optional[torch.nn.Module] = None
    __TOKENIZER: Optional[PreTrainedTokenizerBase] = None
    __MODELS_LOADED: bool = False
    __MODELS_LOCK: threading.Lock = threading.Lock()

    # __MODEL_ID = "openai-community/gpt2-large"

    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name: str = model_name

        generation_strategy = kwargs.get("generation_strategy", "beam_search")
        if generation_strategy not in self.__GENERATION_STRATEGIES:
            raise ValueError(f"Unknown generation_strategy: {generation_strategy}")

        # copy settings so instances don't mutate shared dataclass object
        base = self.__GENERATION_STRATEGIES[generation_strategy]
        self.generate_settings = GenerateSettings(**base.__dict__)

        self.generate_settings.num_return_sequences = int(
            kwargs.get("num_return_sequences", self.generate_settings.num_return_sequences)
        )
        self.generate_settings.num_beams = int(
            kwargs.get("num_beams", self.generate_settings.num_beams)
        )
        self.generate_settings.temperature = float(
            kwargs.get("temperature", self.generate_settings.temperature)
        )
        self.generate_settings.max_length = int(
            kwargs.get("max_length", self.generate_settings.max_length)
        )
        self.generate_settings.max_new_tokens = int(
            kwargs.get("max_new_tokens", self.generate_settings.max_new_tokens)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_size = self.generate_settings.max_length
        self.generate_settings.top_p = float(
                kwargs.get("top_p", self.generate_settings.top_p)
            )
        self.generate_settings.seed = int(
                kwargs.get("seed", self.generate_settings.seed)
            )

        self.__load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def __load_model(self) -> None:
        with self.__MODELS_LOCK:
            if self.__MODELS_LOADED:
                return

            self.__TOKENIZER = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
            )

            # GPT-2 has no pad token by default
            if self.__TOKENIZER.pad_token is None:
                self.__TOKENIZER.pad_token = self.__TOKENIZER.eos_token

            # GPT-2 docs generally recommend right padding
            self.__TOKENIZER.padding_side = "right"
            self.__TOKENIZER.truncation_side = "left"

            self.__MODEL = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            ).to(self.device)

            self.__MODEL.eval()
            set_seed(self.generate_settings.seed)

            # Clamp context size to model maximum
            self.context_size = min(
                self.generate_settings.max_length,
                int(self.__MODEL.config.n_positions),
            )

            self.__MODELS_LOADED = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def __build_generation_prompt(self, prompt: str) -> tuple[str, bool]:
        prompt = prompt.replace("\r\n", "\n").replace("\r", "\n")

        if "<FILL_ME>" not in prompt:
            return prompt, False

        prefix, _, _ = prompt.partition("<FILL_ME>")

        return prefix.rstrip() + "\n", True

    def __safe_generate(self, inputs: dict, max_new_tokens: int) -> Optional[torch.Tensor]:
        assert self.__MODEL is not None and self.__TOKENIZER is not None

        try:
            with torch.no_grad():
                return self.__MODEL.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=1,  
                    num_beams=self.generate_settings.num_beams,
                    num_return_sequences=self.generate_settings.num_return_sequences,
                    do_sample=self.generate_settings.do_sample,
                    temperature=self.generate_settings.temperature,
                    early_stopping=(
                        self.generate_settings.early_stopping
                        if self.generate_settings.num_beams > 1
                        else False
                    ),
                    use_cache=True,
                    pad_token_id=self.__TOKENIZER.pad_token_id,
                    eos_token_id=self.__TOKENIZER.eos_token_id,
                    # top_p=0.95,
                )
        except torch.cuda.OutOfMemoryError:
            logging.warning("CUDA OOM during generation. Skipping this sample.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.warning("RuntimeError OOM during generation. Skipping this sample.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return None
            raise


    def _clean_first_line(self, text: str) -> str:
        text = text.replace("\t", "    ").strip()
        if not text:
            return ""
        line = text.splitlines()[0].strip()
        return line

    # ------------------------------------------------------------------
    # Single-prompt generation
    # ------------------------------------------------------------------
    def __generate_patch(self, prompt: str) -> Optional[List[str]]:
        assert self.__TOKENIZER is not None and self.__MODEL is not None

        generation_prompt, had_fill_marker = self.__build_generation_prompt(prompt)

        inputs = self.__TOKENIZER(
            generation_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_size,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]
        allowed_new_tokens = self.context_size - input_len

        if allowed_new_tokens <= 0:
            logging.warning(
                "Input already fills GPT-2 context window (%d). Skipping sample.",
                self.context_size,
            )
            return None

        allowed_new_tokens = min(
            allowed_new_tokens,
            self.generate_settings.max_new_tokens,
        )

        generated_ids = self.__safe_generate(inputs, allowed_new_tokens)
        if generated_ids is None:
            return None

        completions_ids = generated_ids[:, input_len:]
        completions = self.__TOKENIZER.batch_decode(
            completions_ids,
            skip_special_tokens=True,

        )
        print("-----------------------------------------------")

        # Keep only the first non-empty line
        cleaned = []
        for c in completions:
            line = self._clean_first_line(c)
            if line and line not in cleaned:
                cleaned.append(line)

        # Reconstruct full patched method for inspection
        completions = []
        if had_fill_marker:
            completions = [prompt.replace("<FILL_ME>", c, 1) for c in cleaned]
            print("######################################################")
            print(f"reconstructed: {completions}")
        else:
            completions = cleaned

        return list(completions)

    # ------------------------------------------------------------------
    # Framework hook
    # ------------------------------------------------------------------
    def _generate_impl(self, prompts: List[str]) -> Any:
        results: List[Optional[List[str]]] = []
        for p in tqdm.tqdm(prompts, desc="Generating patches (GPT-2-large)...", total=len(prompts)):
            results.append(self.__generate_patch(p))
        return results