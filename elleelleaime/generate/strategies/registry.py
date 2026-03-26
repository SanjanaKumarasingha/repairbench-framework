from elleelleaime.generate.strategies.strategy import PatchGenerationStrategy
from elleelleaime.generate.strategies.models.huggingface.codellama.codellama_infilling import (
    CodeLLaMAInfilling,
)
from elleelleaime.generate.strategies.models.huggingface.codellama.codellama_instruct import (
    CodeLLaMAIntruct,
)
from elleelleaime.generate.strategies.models.huggingface.repairllama.repairllama_infilling import (
    RepairLLaMAInfilling,
)
from elleelleaime.generate.strategies.models.huggingface.codet5.codet5small_infilling import (
    CodeT5PatchGeneration,
)
# from elleelleaime.generate.strategies.models.huggingface.gpt.gpt2_infilling import (
#     GPT2PatchGeneration,
# )


from typing import Tuple


class PatchGenerationStrategyRegistry:
    """
    Class for storing and retrieving models based on their name.
    """

    # The registry is a dict of strategy names to a tuple of class and mandatory arguments to init the class
    # NOTE: Do not instantiate the model here, as we should only instanciate the class to be used
    __MODELS: dict[str, Tuple[type, Tuple]] = {
        "codellama-infilling": (CodeLLaMAInfilling, ("model_name",)),
        "codellama-instruct": (CodeLLaMAIntruct, ("model_name",)),
        "repairllama-infilling": (RepairLLaMAInfilling, ("model_name",)),
        "codet5-small_infilling": (CodeT5PatchGeneration, ("model_name",)),
        # "gpt2_infilling": (GPT2PatchGeneration, ("model_name",)),
    }

    @classmethod
    def get_generation(cls, name: str, **kwargs) -> PatchGenerationStrategy:
        if name.lower().strip() not in cls.__MODELS:
            raise ValueError(f"Unknown strategy {name}")

        strategy_class, strategy_args = cls.__MODELS[name.lower().strip()]
        for strategy_arg in strategy_args:
            print(f"checking for arg: {strategy_arg}")
            if strategy_arg not in kwargs:
                raise ValueError(f"Missing argument {strategy_arg} for strategy {name}")
        return strategy_class(**kwargs)
