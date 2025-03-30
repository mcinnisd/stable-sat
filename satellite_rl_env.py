#!/usr/bin/env python3
"""
RL-Ready Satellite Environment

This module defines SatelliteRLEnv, an RL-ready environment for satellite attitude control.
It inherits dynamics from SatelliteEnv (core_sat_dynamics.py) and overrides the step function
to accept actions from an RL agent. The state is the satellite's attitude (expressed as Euler angles)
and angular velocities. The action is a lower-dimensional vector matching the number of controlled axes.
This environment now penalizes overshoot and oscillations, uses a larger range for random perturbations,
and, during inference, requires a hold period before "teleporting" to a new target.
"""

import numpy as np
from gym import spaces
from scipy.spatial.transform import Rotation as R
from dynamics import SatelliteEnv, normalize_quaternion, update_state

class SatelliteRLEnv(SatelliteEnv):
    def __init__(self, sim_time=100.0, dt=0.1, inertia=[1.0, 1.0, 1.0],
                 I_w=0.05, max_torque=2.0, controlled_axes=[0, 1, 2],
                 hold_time_threshold=1.0, inference_hold_time=0.5,
                 overshoot_factor=10.0, oscillation_factor=50.0):
        """
        controlled_axes: list of indices (0: roll, 1: pitch, 2: yaw) to be actively controlled.
        hold_time_threshold: time (in seconds) the agent must hold stable to receive a bonus.
        inference_hold_time: minimal hold time during inference before teleportation.
        overshoot_factor: penalty factor for overshoot.
        oscillation_factor: penalty factor for oscillation.
        """
        # Set controlled_axes first
        self.controlled_axes = controlled_axes
        # Initialize desired target as zero (origin) for all axes (in degrees)
        self.desired_euler = np.zeros(3)
        self.hold_time_threshold = hold_time_threshold
        self.inference_hold_time = inference_hold_time
        self.target_hold_time = 0.0
        self.stabilized = False
        # For overshoot/oscillation tracking
        self.prev_error_vector = None
        # Call parent constructor
        super().__init__(sim_time=sim_time, dt=dt, inertia=inertia, I_w=I_w, control_policy=None)
        
        # Number of controlled axes
        self.num_controlled_axes = len(self.controlled_axes)
        # Define action space to match controlled axes
        self.max_torque = max_torque
        self.action_space = spaces.Box(
            low=-max_torque, high=max_torque,
            shape=(self.num_controlled_axes,), dtype=np.float32
        )
        # Observation space remains 6D (Euler angles + angular velocity)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.error_threshold = 5.0     # degrees threshold
        self.omega_threshold = 0.5     # rad/s threshold
        self.time_penalty = 0.1
        
        # Overshoot and oscillation penalty factors
        self.overshoot_factor = overshoot_factor
        self.oscillation_factor = oscillation_factor

    def reset(self, initial_q=None, initial_omega=None):
        # If no initial orientation is provided, sample random Euler angles for controlled axes from a larger range.
        if initial_q is None:
            euler = np.zeros(3)
            for ax in self.controlled_axes:
                # Sample from -90 to 90 degrees
                angle = np.random.uniform(-90, 90)
                # Ensure a minimum perturbation of 5°
                if abs(angle) < 5:
                    angle = 5 if angle >= 0 else -5
                euler[ax] = angle
            initial_q = R.from_euler('xyz', euler, degrees=True).as_quat()
        self.q = normalize_quaternion(initial_q)
        self.omega = np.zeros(3) if initial_omega is None else initial_omega
        self.current_time = 0.0
        self.omega_w = np.zeros(3)
        self.stabilized = False
        self.target_hold_time = 0.0
        self.prev_error_vector = None
        # Desired target remains as previously set (initially zero)
        return self._get_obs()

    def _compute_attitude_error(self):
        """
        Compute error between current attitude and desired attitude (in Euler angles)
        but only for the controlled axes.
        """
        current_euler = R.from_quat(self.q).as_euler('xyz', degrees=True)
        error_vector = np.zeros(3)
        for ax in self.controlled_axes:
            error_vector[ax] = current_euler[ax] - self.desired_euler[ax]
        return np.linalg.norm(error_vector), error_vector

    def step(self, action):
        """
        The RL agent outputs an action vector for the controlled axes.
        We embed it into a 3D torque command (zeros for uncontrolled axes) and update dynamics.
        The reward is based on squared error, angular velocity, control effort, and time penalty.
        Additional penalties are applied for overshoot and oscillations.
        When stable for the required hold time, a bonus is given and the satellite is "teleported"
        (perturbed) to a new target.
        """
        # Embed the lower-dim action into a full 3D command.
        full_action = np.zeros(3, dtype=np.float32)
        for i, ax in enumerate(self.controlled_axes):
            full_action[ax] = action[i]
        full_action = np.clip(full_action, -self.max_torque, self.max_torque)
        
        # Update dynamics
        self.q, self.omega = update_state(self.q, self.omega, self.dt, -full_action, self.I_sat)
        self.omega_w = self.omega_w + (full_action / self.I_w) * self.dt
        self.current_time += self.dt
        
        obs = self._get_obs()
        
        # Compute error (and get per-axis error vector)
        error, curr_error_vector = self._compute_attitude_error()
        error_squared = error ** 2
        
        # Angular velocity and control effort penalties
        angular_penalty = np.linalg.norm(self.omega) ** 2
        control_penalty = np.linalg.norm(action) ** 2
        
        # Base reward coefficients
        C0 = 1000.0
        C1 = 0.7
        C2 = 0.2
        C3 = 0.1
        
        reward = C0 - C1 * error_squared - C2 * angular_penalty - C3 * control_penalty
        reward -= self.time_penalty * self.current_time

        # Overshoot and oscillation penalties (only applied if previous error exists)
        if self.prev_error_vector is not None:
            overshoot_penalty = 0.0
            oscillation_penalty = 0.0
            for ax in self.controlled_axes:
                curr_err = curr_error_vector[ax]
                prev_err = self.prev_error_vector[ax]
                # Overshoot: if absolute error increased
                if abs(curr_err) > abs(prev_err):
                    overshoot_penalty += self.overshoot_factor * (abs(curr_err) - abs(prev_err))
                # Oscillation: if sign change occurs
                if curr_err * prev_err < 0:
                    oscillation_penalty += self.oscillation_factor
            reward -= (overshoot_penalty + oscillation_penalty)
        self.prev_error_vector = curr_error_vector.copy()
        
        # Check stability: if error and angular velocity are below thresholds
        if error < self.error_threshold and np.linalg.norm(self.omega) < self.omega_threshold:
            self.target_hold_time += self.dt
            # In inference, require at least inference_hold_time; otherwise use hold_time_threshold.
            required_hold = self.inference_hold_time if hasattr(self, 'inference_hold_time') else self.hold_time_threshold
            if self.target_hold_time >= required_hold:
                reward += 200.0  # Bonus for holding target
                if not self.stabilized:
                    print(f"Successfully stabilized on axes {self.controlled_axes} at time {self.current_time:.2f}s!")
                    self.stabilized = True
                # Sample new target for controlled axes
                for ax in self.controlled_axes:
                    angle = np.random.uniform(-30, 30)
                    if abs(angle) < 5:
                        angle = 5 if angle >= 0 else -5
                    self.desired_euler[ax] = angle
                print(f"New target for axes {self.controlled_axes}: {self.desired_euler[self.controlled_axes]}")
                # Teleport: perturb the current state to a new random orientation around the new target.
                current_euler = R.from_quat(self.q).as_euler('xyz', degrees=True)
                for ax in self.controlled_axes:
                    perturb = np.random.uniform(-15, 15)
                    current_euler[ax] = self.desired_euler[ax] + perturb
                self.q = normalize_quaternion(R.from_euler('xyz', current_euler, degrees=True).as_quat())
                self.target_hold_time = 0.0
        else:
            self.target_hold_time = 0.0

        # Termination condition: run until simulation time is reached
        done = (self.current_time >= self.sim_time)

        return obs, reward, done, {}

    def render(self, mode='human'):
        # Use parent's rendering method
        super().render(mode)