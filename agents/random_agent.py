from .agent_interface import Agent
from env import Env

class RandomAgent(Agent):
    """
    Implementation of the agent following random policy, not learning at all. 
    """
    def __init__(self):
        super(RandomAgent, self).__init__(learning=False)

    def load_model(self, loadpath):
        pass

    def save_model(self, savepath):
        pass

    def optimize_model(self):
        pass

    def reset(self):
        pass

    def act(self, env: Env):
        return env.random_valid_action()