#!/usr/bin/env python3
"""
RL Controller for Satellite Attitude Control

This module defines RLController, a controller class that wraps an RL policy network.
It conforms to the BaseController interface, allowing it to be swapped in with classical controllers.
Additionally, it supports periodic logging of the agent's version.
"""

import numpy as np
import logging
from controllers import BaseController  # Assumes BaseController is defined in your controllers module

class RLController(BaseController):
    def __init__(self, policy_network, log_interval=100.0, **kwargs):
        """
        policy_network: A callable that accepts an observation and returns an action vector.
        log_interval: Time interval (in seconds) at which to log the agent's version.
        Additional kwargs are passed to BaseController.
        """
        super().__init__(**kwargs)
        self.policy_network = policy_network
        self.log_interval = log_interval
        self.last_log_time = 0.0

    def __call__(self, env):
        # Obtain the current observation.
        obs = env._get_obs()
        # Use the policy network to compute the action.
        action = self.policy_network(obs)
        
        # Log the agent's version periodically.
        if env.current_time - self.last_log_time >= self.log_interval:
            logging.info(f"Logging agent version at time {env.current_time:.2f}s")
            # (Insert code here to save or log agent parameters as needed.)
            self.last_log_time = env.current_time
        
        return action