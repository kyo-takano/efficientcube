"""
EfficientCube Package

This package supports inference with the models trained in the original paper.
For training, please refer to the notebooks listed at https://github.com/kyo-takano/efficientcube/blob/main/notebooks.
"""

import os
import torch
from .environments import load_environment
from .model import Model, ScalableModel
from . import search

class EfficientCube:
    def __init__(
        self,
        env="Rubik's Cube",
        model_path="auto",
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        """
        Initialize EfficientCube object.

        Parameters:
            env (str): The name of the Rubik's Cube environment.
            model_path (str): Path to the trained model file, or "auto" to use default paths.
            device (torch.device): The device to run the model on (GPU if available, otherwise CPU).
        """

        # Set up Rubik's Cube environment
        self.env = load_environment(env)
        
        # If model_path is set to "auto", use default paths based on the environment
        if model_path.lower().strip()=="auto":
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = {
                "Rubik's Cube": "./models/cube3.pth",
                "15 Puzzle": "./models/puzzle15.pth",
                "Lights Out": "./models/lightsout7.pth"
            }[env]
            model_path = os.path.normpath(os.path.join(script_dir, model_path))
        assert os.path.exists(model_path), f"Model file not found at `{model_path}`"

        # Load a trained model from the specified path
        try:
            self.model = torch.jit.load(model_path).to(device)
        except:
            try:
                self.model = torch.load(model_path).to(device)
            except:
                raise ValueError(f"Model could not be loaded from `{model_path}`")

        self.model.eval()  # Set the model to evaluation mode (no training)

    """ Methods defined below are mere routers """

    def solve(self, beam_width):
        # Execute a beam search to find the solution
        return search.beam_search(self.env, self.model, beam_width)

    def env_is_solved(self):
        return self.env.is_solved()
    
    def reset_env(self):
        self.env.reset()
    
    def apply_moves_to_env(self, moves):
        self.env.apply_scramble(moves)
