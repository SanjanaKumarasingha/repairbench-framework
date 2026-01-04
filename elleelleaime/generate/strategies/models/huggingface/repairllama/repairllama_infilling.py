from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Union

import logging
import threading
from pathlib import Path

import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from peft import PeftModel
from elleelleaime.generate.strategies.strategy import PatchGenerationStrategy


@dataclass
class GenerateSettings:
    name: str
    do_sample: bool = False
    temperature: float = 1.0
    num_beams: int = 1
    num_return_sequences: int = 1
    max_length: int = 4096          # context window cap (prompt tokens)
    max_new_tokens: int = 128       # how many tokens to generate
    early_stopping: bool = True


class RepairLLaMAInfilling(PatchGenerationStrategy):
    """
    RepairLLaMA infilling strategy.

    Supports:
      - Official HF adapters (ASSERT-KTH/RepairLLaMA-IR3-OR2 etc.)
      - Local fine-tuned LoRA adapter directories (your "fine-tune/v3", "final", etc.)

    IMPORTANT:
      - For local adapters, we *ignore* adapter_config.json base_model_name_or_path
        (it may point to a cluster-local path) and load a known base model instead.
    """

    __SUPPORTED_MODELS = {
        "ASSERT-KTH/RepairLLaMA-IR1-OR1",
        "ASSERT-KTH/RepairLLaMA-IR1-OR3",
        "ASSERT-KTH/RepairLLaMA-IR1-OR4",
        "ASSERT-KTH/RepairLLaMA-IR2-OR2",
        "ASSERT-KTH/RepairLLaMA-IR3-OR2",
    }

    __GENERATION_STRATEGIES = {
        "beam_search": GenerateSettings(name="beam_search", early_stopping=True),
        "sampling": GenerateSettings(name="sampling", do_sample=True),
    }

    __MODEL: Optional[torch.nn.Module] = None
    __TOKENIZER: Optional[PreTrainedTokenizerBase] = None
    __MODELS_LOADED: bool = False
    __MODELS_LOCK: threading.Lock = threading.Lock()
    __DEFAULT_BASE_MODEL_ID = "codellama/CodeLlama-7b-hf"

    def __init__(self, model_name: str, **kwargs) -> None:
        # Accept HF id OR local directory
        if model_name in self.__SUPPORTED_MODELS:
            self.model_name: str = model_name
            self.is_local_adapter = False
        else:
            p = Path(model_name)
            if not p.is_dir():
                raise ValueError(
                    f"Model '{model_name}' not supported. "
                    f"Expected one of {sorted(self.__SUPPORTED_MODELS)} or a local adapter directory."
                )
            self.model_name = str(p)
            self.is_local_adapter = True

        generation_strategy = kwargs.get("generation_strategy", "beam_search")
        if generation_strategy not in self.__GENERATION_STRATEGIES:
            raise ValueError(f"Unknown generation_strategy: {generation_strategy}")

        self.generate_settings = self.__GENERATION_STRATEGIES[generation_strategy]

        # overrides
        self.generate_settings.num_return_sequences = int(
            kwargs.get("num_return_sequences", self.generate_settings.num_return_sequences)
        )
        self.generate_settings.num_beams = int(kwargs.get("num_beams", self.generate_settings.num_beams))
        self.generate_settings.temperature = float(
            kwargs.get("temperature", self.generate_settings.temperature)
        )
        self.generate_settings.max_length = int(kwargs.get("max_length", self.generate_settings.max_length))
        self.generate_settings.max_new_tokens = int(
            kwargs.get("max_new_tokens", self.generate_settings.max_new_tokens)
        )

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_size = self.generate_settings.max_length

        self.__load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def __load_model(self) -> None:
        """
        Loads tokenizer + model.

        - HF RepairLLaMA adapters: can load tokenizer from adapter repo.
        - Local adapter: load tokenizer from base model (or adapter folder if it contains tokenizer files),
          load base model from a known HF id, then attach adapter via PeftModel.from_pretrained.
        """
        with self.__MODELS_LOCK:
            if self.__MODELS_LOADED:
                return

            # Tokenizer:
            # - If local adapter folder has tokenizer files, you can still point to it;
            #   otherwise fall back to base model tokenizer.
            tokenizer_source = self.model_name
            if self.is_local_adapter:
                # if no tokenizer files in adapter dir, use base tokenizer
                adapter_dir = Path(self.model_name)
                has_tokenizer = any((adapter_dir / f).exists() for f in [
                    "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"
                ])
                if not has_tokenizer:
                    tokenizer_source = self.__DEFAULT_BASE_MODEL_ID

            self.__TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_source)
            if self.__TOKENIZER.pad_token is None:
                self.__TOKENIZER.pad_token = self.__TOKENIZER.eos_token
            self.__TOKENIZER.padding_side = "left"
            self.__TOKENIZER.truncation_side = "left"

            # Model:
            # HF adapters sometimes embed base info correctly → AutoPeft works,
            # BUT local adapters may embed invalid base paths → must manual-load.
            if self.is_local_adapter:
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.__DEFAULT_BASE_MODEL_ID,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                ).to(self.device)

                self.__MODEL = PeftModel.from_pretrained(base_model, self.model_name)
                # DEBUG: verify adapter really loaded
                m = self.__MODEL
                self.__MODEL.to(self.device)
                self.__MODEL.eval()
            else:
                # Official HF adapters: simplest reliable way is still manual attach,
                # because it avoids weird device_map/offload surprises.
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.__DEFAULT_BASE_MODEL_ID,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                ).to(self.device)

                self.__MODEL = PeftModel.from_pretrained(base_model, self.model_name)
                m = self.__MODEL
                print("Loaded adapter path/name:", self.model_name)
                print("Base model id:", self.__DEFAULT_BASE_MODEL_ID)
                self.__MODEL.to(self.device)
                self.__MODEL.eval()

            self.__MODELS_LOADED = True

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
                return self.__MODEL.generate(
                    **inputs,
                    max_new_tokens=self.generate_settings.max_new_tokens,
                    num_beams=self.generate_settings.num_beams,
                    num_return_sequences=self.generate_settings.num_return_sequences,
                    do_sample=self.generate_settings.do_sample,
                    temperature=self.generate_settings.temperature,
                    # early_stopping only meaningful for beam search
                    early_stopping=self.generate_settings.early_stopping if self.generate_settings.num_beams > 1 else False,
                    use_cache=False,
                    pad_token_id=self.__TOKENIZER.pad_token_id,
                )
        except torch.cuda.OutOfMemoryError:
            logging.warning("CUDA OOM during generation. Skipping this sample.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
        except RuntimeError as e:
            # Sometimes OOM comes as RuntimeError("CUDA out of memory")
            if "out of memory" in str(e).lower():
                logging.warning("RuntimeError OOM during generation. Skipping this sample.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return None
            raise

    # ------------------------------------------------------------------
    # Single-prompt generation
    # ------------------------------------------------------------------
    def __generate_patch(self, prompt: str) -> Optional[List[str]]:
        """
        Generates patches for a single prompt.

        - Allows multiple <FILL_ME>; the same completion is inserted into all.
        - Truncates prompt to fit context window.
        - Uses max_new_tokens to cap output.
        - Skips sample safely on OOM instead of crashing the whole run.
        """
        assert self.__TOKENIZER is not None and self.__MODEL is not None

        fill_count = prompt.count("<FILL_ME>")
        if fill_count > 1:
            logging.warning(
                "Prompt contains %d <FILL_ME> tags; using the same completion for all.",
                fill_count,
            )

        # Make sure prompt fits in context window:
        # reserve room for generation tokens
        max_prompt_tokens = max(32, self.context_size - self.generate_settings.max_new_tokens - 8)

        inputs = self.__TOKENIZER(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]
        if input_len >= self.context_size:
            logging.warning(
                "warning: input_len (%d) >= context window (%d). Skipping sample.",
                input_len,
                self.context_size,
            )
            return None

        generated_ids = self.__safe_generate(inputs)
        if generated_ids is None:
            return None

        # Decode only new tokens
        completions_ids = generated_ids[:, input_len:]
        completions = self.__TOKENIZER.batch_decode(completions_ids, skip_special_tokens=True)
        print(f"Completions: {completions}")

        if "<FILL_ME>" in prompt:
            return [prompt.replace("<FILL_ME>", c) for c in completions]
        return list(completions)

    # ------------------------------------------------------------------
    # Framework hook
    # ------------------------------------------------------------------
    def _generate_impl(self, prompts: List[str]) -> Any:
        results: List[Optional[List[str]]] = []
        for p in tqdm.tqdm(prompts, desc="Generating patches (RepairLLaMA)...", total=len(prompts)):
            results.append(self.__generate_patch(p))
        return results
