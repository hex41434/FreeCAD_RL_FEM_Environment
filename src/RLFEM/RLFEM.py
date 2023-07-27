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

        
        self.state = self.interface.state0_trimesh
        
        # self.action_space = spaces.Box(low=20, high=80, shape=(1, 3), dtype=np.uint8)
        # self.observation_space = spaces.Box(low=0, high=255, shape=(23, 55, 3), dtype=np.uint8)
        
        # self.gt_mesh = trimesh.load_mesh(gt_mesh_path) #TODO
        
        self.max_steps = 15

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
        self.interface.doc, self.interface.fea,self.interface.fem_ok = self.interface.run_ccx_updated_inp()
        if not self.interface.fem_ok: # jump out... 
            print(colored('X - X - X - X -X - X -X - X - FEM not OK','red'))
            return [], 0, True, ''
        
        self.interface.step_no +=1
        self.interface.new_Nodes = self.interface.compute_node_displacements()
        self.interface.result_trimesh,self.interface.result_FCmesh = self.interface.resultmesh_to_trimesh()
        self.new_state = self.interface.save_state() # TODO return?!
        self.interface.mesh_OK = self.interface.checkMesh(self.interface.result_FCmesh) #jump out
        if not self.interface.mesh_OK:
            return [], 0, True, ''
        
        print(colored(f'step_no- th:{self.interface.step_no}','green'))
        self.reward = 0 #np.random.uniform(0, 1, 1)[0]  # Random reward
        
        if self.interface.step_no>=self.max_steps:
            self.done = True
            print(f'mesh quality: {self.interface.mesh_OK}')
        else:
            self.done = False  # Continuing task
        
        self.state = self.interface.result_trimesh
        self.info = {"info" : str(self.state)}
  
        return self.state, self.reward, self.done, self.info

    def render(self, mode='rgb_array'):
        """Displays the current state of the environment.
        Can be text or video frames.
        """
        return self.interface.im_observation

    def close(self):
        "To be called before exiting, to free resources."
        pass

        