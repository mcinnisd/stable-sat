#!/usr/bin/env python3
"""
Train an RL Agent for Satellite Attitude Control using PPO (Stable Baselines3)

This script sets up the RL environment, either trains a PPO agent with a custom callback that logs training progress,
plots a reward curve, and saves the model, or loads a pretrained model to simulate its performance.
"""

import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

from dynamics import SatelliteRLEnv

# Custom callback to log training progress and store average rewards
class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, print_freq=5000, verbose=1, model_path='ppo_satellite_agent'):
        super(ProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self.pbar = None
        self.model_path = model_path
        self.next_save = 10000

    def _on_training_start(self) -> None:
        # Initialize a progress bar with total timesteps
        self.pbar = tqdm(total=self.total_timesteps, desc='Training Progress')

    def _on_step(self) -> bool:
        # Update progress bar by 1 step for each call to _on_step
        if self.pbar is not None:
            self.pbar.update(1)

        # Save model at each order of magnitude of timesteps
        if hasattr(self, 'num_timesteps') and self.num_timesteps >= self.next_save:
            self.model.save(f"{self.model_path}_{self.next_save}")
            print(f"Saved model at {self.next_save} timesteps to {self.model_path}_{self.next_save}")
            self.next_save *= 10

        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()

def train_model(args):
    # Create the RL environment with controlled_axes
    env = SatelliteRLEnv(sim_time=args.sim_time, dt=args.dt, inertia=[1.0, 1.0, 1.0],
                          I_w=args.I_w, max_torque=args.max_torque, controlled_axes=args.controlled_axes)
    # Wrap the environment in a DummyVecEnv (required by SB3)
    vec_env = DummyVecEnv([lambda: env])
    # Create the PPO agent with provided hyperparameters
    model = PPO("MlpPolicy", vec_env, verbose=0, learning_rate=args.learning_rate,
                n_steps=args.n_steps, tensorboard_log="./ppo_tensorboard/")
    # Create a progress callback
    progress_callback = ProgressCallback(total_timesteps=args.total_timesteps, print_freq=args.print_freq, verbose=0, model_path=args.model_path)
    # Train the agent
    model.learn(total_timesteps=args.total_timesteps, callback=progress_callback)
    # Save the trained model
    model.save(args.model_path)
    print(f"Model saved as {args.model_path}")
    # (Static plotting via the callback is disabled; see below in simulation mode.)
    return model

def main():
    parser = argparse.ArgumentParser(description="Train or simulate an RL agent for Satellite Attitude Control using PPO")
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--model_path', type=str, default='ppo_satellite_agent', help='Path to save/load the model')
    parser.add_argument('--total_timesteps', type=int, default=100000, help='Total timesteps for training')
    parser.add_argument('--print_freq', type=int, default=5000, help='Frequency of printing training progress')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for PPO')
    parser.add_argument('--n_steps', type=int, default=4096, help='Number of steps per rollout')
    parser.add_argument('--sim_time', type=float, default=50.0, help='Simulation time for environment')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step for environment')
    parser.add_argument('--I_w', type=float, default=0.05, help='Reaction wheel inertia')
    parser.add_argument('--max_torque', type=float, default=2.0, help='Maximum torque')
    parser.add_argument('--plot_static', action='store_true', help='Use static plotting instead of interactive simulation')
    parser.add_argument('--controlled_axes', type=str, default='0', help='Comma-separated list of axis indices to control (e.g., "0" for roll, "0,1" for roll and pitch, "0,1,2" for all axes)')
    args = parser.parse_args()

    # Parse controlled_axes from string to list of ints
    args.controlled_axes = [int(x.strip()) for x in args.controlled_axes.split(",")]

    if args.train:
        model = train_model(args)
    else:
        print("Please specify --train to train a new model.")
        return

if __name__ == '__main__':
    main()