#!/usr/bin/env python3
"""
RL-Ready Satellite Environment

This module defines SatelliteRLEnv, an RL-ready environment for satellite attitude control.
It inherits dynamics from SatelliteEnv (core_sat_dynamics.py) and overrides the step function
to accept actions from an RL agent. The state is the satellite's attitude (expressed as Euler angles)
and angular velocities. The reward function now penalizes deviations from the ideal quaternion [0, 0, 0, 1],
with separate considerations for the attitude (controlled Euler angles) and the scalar component.
The episode terminates when the error is below a defined threshold for a sustained period.
"""

import numpy as np
from gym import spaces
from scipy.spatial.transform import Rotation as R
from dynamics import SatelliteEnv, normalize_quaternion, update_state

class SatelliteRLEnv(SatelliteEnv):
    def __init__(self, sim_time=100.0, dt=0.1, inertia=[1.0, 1.0, 1.0],
                 I_w=0.05, max_torque=.2, controlled_axes=[0, 1, 2],
                 hold_time_threshold=1.0):
        """
        controlled_axes: list of indices (0: roll, 1: pitch, 2: yaw) to be actively controlled.
        hold_time_threshold: time (in seconds) the agent must hold stable to end the episode.
        """
        self.controlled_axes = controlled_axes
        self.hold_time_threshold = hold_time_threshold
        self.target_hold_time = 0.0
        
        # Call parent constructor (no control policy since RL agent provides actions)
        super().__init__(sim_time=sim_time, dt=dt, inertia=inertia, I_w=I_w, control_policy=None)
        
        self.num_controlled_axes = len(self.controlled_axes)
        self.max_torque = max_torque
        self.action_space = spaces.Box(
            low=-max_torque, high=max_torque,
            shape=(self.num_controlled_axes,), dtype=np.float32
        )
        # Observation remains 6D: Euler angles (attitude) and angular velocity.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        # Thresholds for stabilization
        self.error_threshold = 0.5     # degrees (attitude error for controlled axes)
        self.omega_threshold = 0.25     # rad/s (angular velocity threshold)
        self.time_penalty = 0.05

    def reset(self, initial_q=None, initial_omega=None):
        # If no initial orientation is provided, sample a random quaternion uniformly.
        if initial_q is None:
            if set(self.controlled_axes) == {0, 1, 2}:
                # If all axes are controlled, sample a uniformly random quaternion.
                initial_q = np.random.randn(4)
                initial_q = initial_q / np.linalg.norm(initial_q)
            else:
                # If only a subset of axes is controlled, sample a random rotation only around those axes.
                euler = np.zeros(3)
                for ax in self.controlled_axes:
                    angle = np.random.uniform(-90, 90)
                    if abs(angle) < 5:
                        angle = 5 if angle >= 0 else -5
                    euler[ax] = angle
                initial_q = R.from_euler('xyz', euler, degrees=True).as_quat()
        self.q = normalize_quaternion(initial_q)
        self.omega = np.zeros(3) if initial_omega is None else initial_omega
        self.current_time = 0.0
        self.omega_w = np.zeros(3)
        self.target_hold_time = 0.0
        self.prev_error = self._compute_attitude_error()
        return self._get_obs()

    def _compute_attitude_error(self):
        q_target = np.array([0, 0, 0, 1])
        error_rot = R.from_quat(self.q) * R.from_quat(q_target).inv()
        rotvec = error_rot.as_rotvec()
        error = np.degrees(np.linalg.norm(rotvec))
        return error

    def step(self, action):
        """
        The RL agent outputs an action vector for the controlled axes.
        The action is embedded into a full 3D torque command (with zeros for uncontrolled axes).
        The reward is based on:
          - Squared attitude error (for controlled axes),
          - A penalty for deviation of the quaternion scalar component from 1,
          - Angular velocity and control effort penalties,
          - Time penalty, and additional overshoot/oscillation penalties.
        The episode terminates when the attitude error is below a threshold for a sustained period.
        """
        # Embed lower-dimensional action into full 3D command.
        full_action = np.zeros(3, dtype=np.float32)
        for i, ax in enumerate(self.controlled_axes):
            full_action[ax] = action[i]
        full_action = np.clip(full_action, -self.max_torque, self.max_torque)
        
        # Update dynamics.
        self.q, self.omega = update_state(self.q, self.omega, self.dt, -full_action, self.I_sat)
        self.current_time += self.dt
        
        self.omega_w = self.omega_w + (full_action / self.I_w) * self.dt
        
        obs = self._get_obs()
        
        # Compute current error
        error = self._compute_attitude_error()
        
        # Incremental reward: reward improvement if error decreases
        if hasattr(self, 'prev_error'):
            delta_error = self.prev_error - error
        else:
            delta_error = 0.0
        self.prev_error = error
        
        # Base reward: use the improvement delta and apply a time penalty
        reward = delta_error - self.time_penalty * self.current_time
        # reward = -error**2 - self.time_penalty * self.current_time

        ang_pen_coef = 1
        angular_penalty = -ang_pen_coef * np.linalg.norm(self.omega)**2
        reward += angular_penalty

        ctrl_pen_coef = 1
        control_penalty = -ctrl_pen_coef * np.linalg.norm(action)**2
        reward += control_penalty

        # if hasattr(self, 'prev_error'):
        #     if np.sign(self.prev_error - self.error_threshold) != np.sign(error - self.error_threshold):
        #         reward -= 5.0  # Adjust penalty magnitude as needed

        # Check stabilization
        done = False
        if error < self.error_threshold and np.linalg.norm(self.omega) < self.omega_threshold:
            self.target_hold_time += self.dt
            if self.target_hold_time >= self.hold_time_threshold:
                reward += 1000.0  # Bonus for holding the target
                done = True
        else:
            self.target_hold_time = 0.0
        if self.current_time >= self.sim_time:
            done = True

        return obs, reward, done, {}

    def render(self, mode='human'):
        # Use the parent's rendering method.
        super().render(mode)

    def draw_satellite(ax, q, inertia, current_time, q_desired=None, error=None):
        pass  # Existing function body remains unchanged.