from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import logging
import threading
import re

import torch
import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from elleelleaime.generate.strategies.strategy import PatchGenerationStrategy


@dataclass
class GenerateSettings:
    name: str
    do_sample: bool = False
    temperature: float = 1.0
    num_beams: int = 1
    num_return_sequences: int = 1
    max_length: int = 512          # max input tokens (source)
    max_new_tokens: int = 128      # generated tokens
    early_stopping: bool = True
    top_p: float = 0.95            # used only when do_sample=True
    top_k: int = 50                # used only when do_sample=True


class CodeT5PatchGeneration(PatchGenerationStrategy):
    """
    CodeT5 patch generation strategy (Seq2Seq).

    - model_name: HF model id or local folder containing full seq2seq weights.
    - generation_strategy: "beam_search" or "sampling"
    - prompt format: you can pass your existing buggy+prompt string.
    - <FILL_ME> behavior: if prompt contains <FILL_ME>, model output is treated as
      the completion and inserted into all <FILL_ME> occurrences (same as your current).
    """

    __GENERATION_STRATEGIES = {
        "beam_search": GenerateSettings(name="beam_search", early_stopping=True),
        "sampling": GenerateSettings(name="sampling", do_sample=True),
    }

    __MODEL: Optional[torch.nn.Module] = None
    __TOKENIZER: Optional[PreTrainedTokenizerBase] = None
    __MODELS_LOADED: bool = False
    __MODELS_LOCK: threading.Lock = threading.Lock()

    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name: str = model_name

        generation_strategy = kwargs.get("generation_strategy", "beam_search")
        if generation_strategy not in self.__GENERATION_STRATEGIES:
            raise ValueError(f"Unknown generation_strategy: {generation_strategy}")

        # copy base settings
        base = self.__GENERATION_STRATEGIES[generation_strategy]
        self.generate_settings = GenerateSettings(**base.__dict__)

        # overrides (same knobs you already use)
        self.generate_settings.num_return_sequences = int(
            kwargs.get("num_return_sequences", self.generate_settings.num_return_sequences)
        )
        self.generate_settings.num_beams = int(kwargs.get("num_beams", self.generate_settings.num_beams))
        self.generate_settings.temperature = float(kwargs.get("temperature", self.generate_settings.temperature))
        self.generate_settings.max_length = int(kwargs.get("max_length", self.generate_settings.max_length))
        self.generate_settings.max_new_tokens = int(kwargs.get("max_new_tokens", self.generate_settings.max_new_tokens))

        # optional sampling knobs
        if "top_p" in kwargs:
            self.generate_settings.top_p = float(kwargs["top_p"])
        if "top_k" in kwargs:
            self.generate_settings.top_k = int(kwargs["top_k"])

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_size = self.generate_settings.max_length

        # If user wants fill-only behavior improvement, we can wrap prompt slightly.
        # Keep default True because CodeT5 is not an "infilling" model by default.
        self.wrap_fill_prompt = bool(kwargs.get("wrap_fill_prompt", True))

        self.__load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def __load_model(self) -> None:
        with self.__MODELS_LOCK:
            if self.__MODELS_LOADED:
                return

            self.__TOKENIZER = AutoTokenizer.from_pretrained(self.model_name)

            # For seq2seq, right padding is standard
            self.__TOKENIZER.padding_side = "right"
            self.__TOKENIZER.truncation_side = "right"

            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.__MODEL = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
            ).to(self.device)

            self.__MODEL.eval()
            self.__MODELS_LOADED = True

            # Optional: warn if user set max_length beyond tokenizer recommendation
            try:
                tmax = int(getattr(self.__TOKENIZER, "model_max_length", 0) or 0)
                if 0 < tmax < 100000 and self.context_size > tmax:
                    logging.warning(
                        "Configured max_length=%d is greater than tokenizer.model_max_length=%d. "
                        "Consider lowering max_length to avoid truncation/instability.",
                        self.context_size,
                        tmax,
                    )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def __safe_generate(self, inputs: dict) -> Optional[torch.Tensor]:
        """
        Run generate() with OOM handling.
        If OOM, free cache and return None (skip this sample, continue run).
        """
        assert self.__MODEL is not None and self.__TOKENIZER is not None

        try:
            with torch.no_grad():
                # For beam search: num_return_sequences must be <= num_beams
                num_beams = self.generate_settings.num_beams
                if not self.generate_settings.do_sample:
                    num_beams = max(num_beams, self.generate_settings.num_return_sequences)

                return self.__MODEL.generate(
                    **inputs,
                    max_new_tokens=self.generate_settings.max_new_tokens,
                    num_beams=num_beams,
                    num_return_sequences=self.generate_settings.num_return_sequences,
                    do_sample=self.generate_settings.do_sample,
                    temperature=self.generate_settings.temperature if self.generate_settings.do_sample else None,
                    top_p=self.generate_settings.top_p if self.generate_settings.do_sample else None,
                    top_k=self.generate_settings.top_k if self.generate_settings.do_sample else None,
                    early_stopping=self.generate_settings.early_stopping if num_beams > 1 else False,
                    use_cache=False,
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

    def __build_input(self, prompt: str) -> str:
        """
        If prompt contains <FILL_ME>, optionally wrap with a strict instruction so CodeT5
        outputs ONLY the missing code (acts like your infill behavior).
        """
        if "<FILL_ME>" in prompt and self.wrap_fill_prompt:
            return (
                f"{prompt}"
            )
        return prompt

    # ------------------------------------------------------------------
    # Single-prompt generation
    # ------------------------------------------------------------------
    def __generate_patch(self, prompt: str) -> Optional[List[str]]:
        """
        Generates patches for a single prompt.

        - Truncates input to max_length (source length).
        - Uses max_new_tokens to cap output.
        - Skips sample safely on OOM instead of crashing the whole run.
        - If prompt contains <FILL_ME>, treat model output as completion and replace.
        """
        assert self.__TOKENIZER is not None and self.__MODEL is not None

        # fill_count = prompt.count("<FILL_ME>")
        # if fill_count > 1:
        #     logging.warning(
        #         "Prompt contains %d <FILL_ME> tags; using the same completion for all.",
        #         fill_count,
        #     )

        # model_input = self.__build_input(prompt)

        inputs = self.__TOKENIZER(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_size,
            padding=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generated_ids = self.__safe_generate(inputs)
        if generated_ids is None:
            return None

        completions = self.__TOKENIZER.batch_decode(generated_ids, skip_special_tokens=False)
        print("-----------------------------------------------")
        print(f"prompt: {prompt}")
        print("######################################################")
        print(f"completions: {completions}")
        
        pattern = re.compile(r"<extra_id_0>(.*?)<extra_id_1>", re.DOTALL)


        # Basic cleanup / de-dup
        cleaned: List[str] = []
        seen = set()

        for c in completions:
            c = c.strip()
            if c.startswith("```"):
                c = c.strip("`").strip()

            m = pattern.search(c)
            extracted = m.group(1).strip() if m else c  # if no match, keep original string
            print(f"extracted: {extracted}")

            if extracted and extracted not in seen:
                cleaned.append(extracted)
                seen.add(extracted)

        if not cleaned:
            return None

        # If prompt uses <extra_id_0>, insert completion(s) into the prompt like before
        if "<extra_id_0>" in prompt:
            print(f"Final cleaned completions to insert: {cleaned}")
            return [prompt.replace("<extra_id_0>", c) for c in cleaned]

        # Otherwise return raw generations
        # print(f"cleaned: {cleaned}")
        return cleaned

    # ------------------------------------------------------------------
    # Framework hook
    # ------------------------------------------------------------------
    def _generate_impl(self, prompts: List[str]) -> Any:
        results: List[Optional[List[str]]] = []
        for p in tqdm.tqdm(prompts, desc="Generating patches (CodeT5)...", total=len(prompts)):
            results.append(self.__generate_patch(p))
        return results