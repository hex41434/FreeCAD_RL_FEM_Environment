import os
import sys

import numpy as np
import gym

from .FreeCADWrapper import FreeCADWrapper


class RLFEM(gym.Env):
    """Environment allowing to interact with the FreeCad simulation.

    TODO: more description
    """

    def __init__(self,
                 path=None,
                 force_value=None,
                 force_factor=None,
                 num_actions=None,
                 flag_save=None,
                 action_type=None,
                 max_triangles=None,
                 try_crash_num=5):
        """
        :param path: path to the .ply file.
        """

        self.interface = FreeCADWrapper(path=path,
                                        force_value=force_value,
                                        force_factor=force_factor,
                                        num_actions=num_actions,
                                        flag_save=flag_save,
                                        action_type=action_type,
                                        max_triangles=max_triangles)

        super().__init__()  # TODO: could this line be called before interface super(RLFEM, self).__init__()

        self.try_crash_num = try_crash_num

        self.state = self.reset()

    def reset(self):
        """Resets the environment and starts from an initial state.

        :returns: the initial state.
        """
        self.interface.clear_doc()
        self.interface.initialize_doc()
        # Initial state
        # self.state = self.interface.reset()

        return 'self.state'

    def step(self, action):
        """Takes an action in the environment and returns a new state, a reward, a boolean (True for terminal states)
        and a dictionary with additional info (optional).

        :param action: the action to be taken.

        :returns: (next_state, reward, done, info)
        """
        region_values, force_dir_str = action
        self.state = self.interface.run_step(self, region_values=region_values,
                                             force_dir_str=force_dir_str)  # Random transition to another state
        self.reward = np.random.uniform(0, 1, 1)[0]  # Random reward
        self.done = False  # Continuing task
        self.info = {}  # No info

        return self.state, self.reward, self.done, self.info

    def render(self, mode='human'):
        """Displays the current state of the environment.

        Can be text or video frames.
        """

        print(self.state)

    def close(self):
        "To be called before exiting, to free resources."
        pass

    def random_action(self):
        region_values, force_dir_str = self.interface.get_random_action()
        return region_values, force_dir_str