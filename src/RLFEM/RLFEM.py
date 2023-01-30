import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import sys

import numpy as np
import gym
from gym import spaces
import trimesh
from termcolor import colored
import random

from FreeCADWrapper import FreeCADWrapper


class RLFEM(gym.Env):
    """Environment allowing to interact with the FreeCad simulation
    TODO: more description
    """
    #metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,
                 render_modes,
                 save_path,
                 loaded_mesh_path,
                 loaded_mesh_filename,
                 gt_mesh_path,
                 view_meshes,
                 xls_pth,
                 xls_filename,
                 load_3d):
        """
        :param path: path to the .obj file.
        """
        super(RLFEM, self).__init__()

        self.interface = FreeCADWrapper(
                 render_modes,
                 save_path,
                 loaded_mesh_path,
                 loaded_mesh_filename,
                 gt_mesh_path,
                 view_meshes,
                 xls_pth,
                 xls_filename,
                 load_3d)

        self.state = self.reset()
        # self.action_space = spaces.Box(np.array([2,450000]),np.array([8,800000]))
        self.action_space = spaces.Box(low=20, high=80, shape=(1, 3), dtype=np.uint8)
        # self.observation_space = spaces.Box(low=0, high=255, shape=(55, 23, 3), dtype=np.uint8)
        self.observation_space = spaces.Box(low=0, high=255, shape=(23, 55, 3), dtype=np.uint8)
        
        self.gt_mesh = trimesh.load_mesh(gt_mesh_path) #TODO
        self.max_steps = 3

    def reset(self):
        """Resets the environment and starts from an initial state.

        :returns: the initial state.
        """
        return self.interface.reset_env()

    def step(self, action):
        """Takes an action in the environment and returns a new state, a reward, a boolean (True for terminal states)
        and a dictionary with additional info (optional).

        :param action: the action to be taken.

        :returns: (next_state, reward, done, info)
        """

        self.interface.create_fem_analysis(action)
        self.doc, self.fea = self.interface.run_ccx_updated_inp()
        
        self.state = self.interface.prepare_for_next_fem_step()
        
        self.interface.step_no +=1
        print(colored(f'step_no- th:{self.interface.step_no}','green'))
        
        self.reward = np.random.uniform(0, 1, 1)[0]  # Random reward
        
        # msh = self.interface.result_trimesh
        # gt_points = self.gt_mesh
        # # self.reward = self.interface.compute_trimesh_chamfer(gt_points, msh)
        
        print(colored(f'reward:{self.reward}','cyan'))
        if self.interface.step_no>=self.max_steps: 
            self.done = True
        else:
            self.done = False  # Continuing task
        
        self.info = {"info" : str(self.state)}
  

        print(f"+ . + . + . + . observation_image_size:{self.interface.im_observation.size}")
        return self.interface.im_observation, self.reward, self.done, self.info

    def render(self, mode='rgb_array'):
        """Displays the current state of the environment.
        Can be text or video frames.
        """
        # self.state = self.interface.state0_trimesh.vertices[0][2]
        # print(colored(f'state_render:{self.state}', 'cyan'))
        return self.interface.im_observation

    def close(self):
        "To be called before exiting, to free resources."
        pass

        