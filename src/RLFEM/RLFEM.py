import os
import sys

import numpy as np
import gym
from gym import spaces

# import gymnasium as gym
# from gymnasium import spaces

from .FreeCADWrapper import FreeCADWrapper


class RLFEM(gym.Env):
    """Environment allowing to interact with the FreeCad simulation
    TODO: more description
    """
    #metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,
                 render_mode=None,
                 save_path,
                 loaded_mesh_path,
                 loaded_mesh_filename,
                 view_meshes,
                 xls_pth,
                 xls_filename,
                 load_3d=True):
        """
        :param path: path to the .obj file.
        """
        super(RLFEM, self).__init__()

        self.interface = FreeCADWrapper(
                 save_path,
                 loaded_mesh_path,
                 loaded_mesh_filename,
                 view_meshes,
                 xls_pth,
                 xls_filename,
                 load_3d=True)

        self.state = self.reset()
        self.action_space = spaces.Box(np.array([2,250000]),np.array([8,300000]))

    def reset(self):
        """Resets the environment and starts from an initial state.

        :returns: the initial state.
        """
        return self.reset_env()

    def step(self, action):
        """Takes an action in the environment and returns a new state, a reward, a boolean (True for terminal states)
        and a dictionary with additional info (optional).

        :param action: the action to be taken.

        :returns: (next_state, reward, done, info)
        """
        #region_values, force_dir_str = action
        #(self.force_position, self.force_val) = self.generate_action()
        
        self.create_fem_analysis()
        self.fem_step()
        
        self.reward = np.random.uniform(0, 1, 1)[0]  # Random reward
        #chamf = compute_trimesh_chamfer(gt_points, msh, offset=0, scale=1, num_mesh_samples=300000)
        
        self.done = False  # Continuing task
        self.info = {}  # No info

        return self.result_trimesh, self.reward, self.done

    def render(self, mode='human'):
        """Displays the current state of the environment.

        Can be text or video frames.
        """

        print(self.result_trimesh)# ?! current state?! #########

    def close(self):
        "To be called before exiting, to free resources."
        pass

    def random_action(self):
        (self.force_position, self.force_val) = self.generate_action()
        return self.force_position, self.force_val
