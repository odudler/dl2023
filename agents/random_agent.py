from .agent_interface import Agent
from env import Env

class RandomAgent(Agent):
    """
    Implementation of the agent following random policy, not learning at all. 
    """
    def __init__(self, env: Env):
        super(RandomAgent, self).__init__(learning=False)
        self.env = env

    def load_model(self, loadpath):
        pass

    def save_model(self, savepath):
        pass

    def optimize_model(self):
        pass

    def reset(self):
        pass

    def act(self, state: list, **kwargs):
        return self.env.random_valid_action()