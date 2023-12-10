"Implementation of Agent following Random Policy, not learning at all"
from .agent_interface import Agent

class RandomAgent(Agent):
    def __init__(self, env):
        self.env = env

    def load_model(self, loadpath):
        pass

    def save_model(self, savepath):
        pass

    def optimize_model(self):
        pass

    def reset(self):
        pass

    def act(self, state):
        return self.env.random_valid_action()