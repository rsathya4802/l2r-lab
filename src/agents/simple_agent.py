import json
import time
import numpy as np
from src.agents.base import BaseAgent
from src.config.yamlize import yamlize, create_configurable, NameToSourcePath
from src.utils.utils import ActionSample


@yamlize
class randomAgent(BaseAgent):
    """Randomly pick actions in the space."""
    
    def __init__(
        self,
        steps_to_sample_randomly: int,
        gamma: float,
        alpha: float,
        polyak: float,
        lr: float,
        actor_critic_cfg_path: str,
        load_checkpoint_from: str = "",
    ):
        super(randomAgent, self).__init__()

    def select_action(self, obs) -> np.array:
        ac_sample = ActionSample()
        random_action = self.action_space.sample()
        ac_sample.action = random_action

        return ac_sample
