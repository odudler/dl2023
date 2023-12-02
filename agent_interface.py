from abc import ABC, abstractmethod

'''
Interface for various kinds of Agents
NOTE: __init__ method takes different parameters depending on Agent
Normally, it takes at least: env, mem (buffer), policy/target network, potentially some hyperparameters
'''

class Agent(ABC):
    @abstractmethod
    def load_model(self, loadpath):
        pass

    @abstractmethod
    def save_model(self, savepath):
        pass

    @abstractmethod
    def optimize_model(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def act(self, state):
        pass
