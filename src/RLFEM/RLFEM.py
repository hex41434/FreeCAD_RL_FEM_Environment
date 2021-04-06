import os
import sys

import numpy as np
import gym

from .FreeCADWrapper import FreeCADWrapper

class RLFEM(gym.Env):
    """Environment allowing to interact with the FreeCad simulation.

    TODO: more description
    """

    def __init__(self, path=''):
        """
        :param path: path to the .ply file.
        """
        self.path = path

        self.interface = FreeCADWrapper(self.path)

        self.state = self.reset()

        super().__init__()
    
    def reset(self):
        """Resets the environment and starts from an initial state.
        
        :returns: the initial state.
        """
        
        # Initial state
        # self.state = self.interface.reset()
        
        return self.state
    
    def step(self, action):
        """Takes an action in the environment and returns a new state, a reward, a boolean (True for terminal states) 
        and a dictionary with additional info (optional).

        :param action: the action to be taken.

        :returns: (next_state, reward, done, info)
        """
        
        self.state = self.interface.apply(action) # Random transition to another state
        self.reward = np.random.uniform(0, 1, 1)[0] # Random reward
        self.done = False # Continuing task
        self.info = {} # No info
        
        return self.state, self.reward, self.done, self.info

    def render(self, mode='human'):
        """Displays the current state of the environment. 
        
        Can be text or video frames.
        """
        
        print(self.state)
    
    def close(self):
        "To be called before exiting, to free resources."
        pass