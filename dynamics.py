#!/usr/bin/env python3
"""
Core Satellite Dynamics and Environment Classes

This module contains the core satellite dynamics including the reaction wheel dynamics,
state update functions, and a base environment class for satellite attitude control.
"""

import gym
from gym import spaces
import numpy as np
import scipy.linalg as la
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import visualization
# import logging for dynamics module
import logging
import warnings
warnings.filterwarnings("ignore", message="Gimbal lock detected")

# Set up logging for dynamics module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def update_state(q: np.ndarray, omega: np.ndarray, dt: float, torque: np.ndarray, I_sat: np.ndarray, I_inv: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Updates the satellite state given the current state, time step, applied torque, and inertia matrix.
    
    Parameters:
      q (np.ndarray): Current quaternion [x, y, z, w].
      omega (np.ndarray): Current angular velocity (rad/s).
      dt (float): Time step.
      torque (np.ndarray): Applied external torque.
      I_sat (np.ndarray): Satellite inertia matrix.
      I_inv (np.ndarray, optional): Precomputed inverse of I_sat. If not provided, it will be computed.
    
    Returns:
      q_new (np.ndarray): Updated (and normalized) quaternion.
      omega_new (np.ndarray): Updated angular velocity.
    """
    if I_inv is None:
        I_inv = np.linalg.inv(I_sat)
    
    # Calculate angular acceleration: ω̇ = I⁻¹ (τ - ω × (I * ω))
    omega_dot = I_inv @ (torque - np.cross(omega, I_sat @ omega))
    
    # Update angular velocity using Euler integration
    omega_new = omega + omega_dot * dt
    
    # Update quaternion: use the new angular velocity to compute the rotation vector
    delta_rot = R.from_rotvec(omega_new * dt)
    q_new = (R.from_quat(q) * delta_rot).as_quat()
    
    # Normalize the updated quaternion to ensure unit length
    q_new = normalize_quaternion(q_new)
    
    return q_new, omega_new


def compute_lqr_gain(I_i: float, Q: np.ndarray, R_mat: np.ndarray) -> np.ndarray:
    """Compute LQR gain for a double integrator model for a given axis."""
    A = np.array([[0, 1],
                  [0, 0]])
    B = np.array([[0],
                  [1.0 / I_i]])
    P = la.solve_continuous_are(A, B, Q, R_mat)
    K = np.linalg.inv(R_mat) @ (B.T @ P)
    return K


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalizes a quaternion and logs a warning if the norm is zero."""
    norm = np.linalg.norm(q)
    if norm == 0:
        logging.warning("Attempted to normalize a zero-norm quaternion. Returning the original quaternion.")
        return q
    return q / norm


class SatelliteEnv(gym.Env):
    """
    Base Satellite Environment with Reaction Wheel Dynamics.
    The control policy is injected via the control_policy callable.
    The environment applies the control policy torque as reaction wheel torque.
    """
    def __init__(self, sim_time=100.0, dt=0.1, inertia=[1.0, 1.0, 1.0], I_w=0.05, control_policy=None):
        super(SatelliteEnv, self).__init__()
        self.sim_time = sim_time
        self.dt = dt
        self.inertia = inertia
        self.I_sat = np.diag(inertia)
        self.I_w = I_w
        # Precompute the inverse inertia matrix since I_sat is constant
        self.I_inv = np.linalg.inv(self.I_sat)
        # Inject control policy; if None, default to zero control.
        self.control_policy = control_policy if control_policy is not None else self.zero_control
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.fig = None
        self.ax = None
        self.reset()
    
    def zero_control(self, env):
        # Default control policy: no torque.
        return np.zeros(3)
    
    def reset(self, initial_q=None, initial_omega=None):
        self.q = np.array([0, 0, 0, 1]) if initial_q is None else normalize_quaternion(initial_q)
        self.omega = np.zeros(3) if initial_omega is None else initial_omega
        self.current_time = 0.0
        self.omega_w = np.zeros(3)
        if hasattr(self, 'control_policy') and self.control_policy is not None and hasattr(self.control_policy, 'desired_points'):
            from scipy.spatial.transform import Rotation as R
            self.desired_euler_points = [R.from_quat(q).as_euler('xyz', degrees=True) for q in self.control_policy.desired_points]
        else:
            self.desired_euler_points = None
        return self._get_obs()
    
    def _get_obs(self):
        euler = R.from_quat(self.q).as_euler('xyz', degrees=True)
        return np.concatenate([euler, self.omega])
    
    def _init_plot(self):
        if self.fig is None or self.ax is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
    
    def render(self, mode='human'):
        self._init_plot()
        q_desired = None
        error = None
        # If the control policy has a desired_points attribute, compute the target orientation and error
        if hasattr(self, 'control_policy') and self.control_policy is not None and hasattr(self.control_policy, 'desired_points'):
            q_desired = self.control_policy.desired_points[self.control_policy.current_target_idx]
            from scipy.spatial.transform import Rotation as R
            error_rot = R.from_quat(q_desired) * R.from_quat(self.q).inv()
            error = error_rot.as_rotvec()
        visualization.draw_satellite(self.ax, self.q, self.inertia, self.current_time, q_desired=q_desired)
        visualization.annotate_wheel_speeds(self.ax, self.omega_w, error=error)
        plt.draw()
        plt.pause(0.1)
    
    def step(self, action=None):
        # Retrieve the control torque from the injected policy.
        u = self.control_policy(self)
        # The satellite experiences -u due to the reaction wheels.
        self.q, self.omega = update_state(self.q, self.omega, self.dt, -u, self.I_sat, self.I_inv)
        # Update reaction wheel speeds.
        self.omega_w = self.omega_w + (u / self.I_w) * self.dt
        self.current_time += self.dt
        done = self.current_time >= self.sim_time
        return self._get_obs(), 0.0, done, {'torque': u.copy()}

class SatelliteRLEnv(SatelliteEnv):
    """
    RL-Ready Satellite Environment

    Inherits dynamics from SatelliteEnv and accepts actions from an RL agent.
    State is [roll, pitch, yaw, ω_x, ω_y, ω_z], reward penalizes deviation from [0,0,0,1],
    and episode ends when error is below threshold for hold_time_threshold.
    """
    def __init__(self, sim_time=100.0, dt=0.1, inertia=[1.0, 1.0, 1.0],
                 I_w=0.05, max_torque=.2, controlled_axes=[0, 1, 2],
                 hold_time_threshold=1.0):
        # Flag to indicate we're in initialization, so reset uses parent logic
        self._during_init = True
        # Normalize controlled_axes: accept either a comma-separated string or a list of indices
        if isinstance(controlled_axes, str):
            axes_list = controlled_axes.split(',')
        else:
            axes_list = controlled_axes
        self.controlled_axes = [int(ax) for ax in axes_list]
        self.hold_time_threshold = hold_time_threshold
        self.target_hold_time = 0.0
        # Call parent constructor (no control_policy for RL)
        super().__init__(sim_time=sim_time, dt=dt, inertia=inertia, I_w=I_w, control_policy=None)
        # End of init-phase; subsequent resets will use RL-specific logic
        self._during_init = False
        self.num_controlled_axes = len(self.controlled_axes)
        self.max_torque = max_torque
        self.action_space = spaces.Box(
            low=-max_torque, high=max_torque,
            shape=(self.num_controlled_axes,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.error_threshold = 0.5     # degrees
        self.omega_threshold = 0.25    # rad/s
        self.time_penalty = 0.05

    def reset(self, initial_q=None, initial_omega=None):
        # If called during base init, defer to parent reset to avoid sampling logic
        if getattr(self, '_during_init', False):
            return super().reset(initial_q, initial_omega)

        # Sample initial quaternion if none provided
        if initial_q is None:
            if set(self.controlled_axes) == {0, 1, 2}:
                initial_q = np.random.randn(4)
                initial_q /= np.linalg.norm(initial_q)
            else:
                euler = np.zeros(3)
                for ax in self.controlled_axes:
                    angle = np.random.uniform(-90, 90)
                    euler[ax] = angle if abs(angle) >= 5 else (5 if angle >= 0 else -5)
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
        return np.degrees(np.linalg.norm(error_rot.as_rotvec()))

    def step(self, action):
        full_action = np.zeros(3, dtype=np.float32)
        for i, ax in enumerate(self.controlled_axes):
            full_action[ax] = action[i]
        full_action = np.clip(full_action, -self.max_torque, self.max_torque)
        # Update dynamics
        self.q, self.omega = update_state(self.q, self.omega, self.dt, -full_action, self.I_sat)
        self.current_time += self.dt
        self.omega_w += (full_action / self.I_w) * self.dt
        obs = self._get_obs()
        # Reward computation
        error = self._compute_attitude_error()
        delta_error = getattr(self, 'prev_error', 0.0) - error
        self.prev_error = error
        reward = delta_error - self.time_penalty * self.current_time
        reward -= np.linalg.norm(self.omega)**2
        reward -= np.linalg.norm(action)**2
        done = False
        if error < self.error_threshold and np.linalg.norm(self.omega) < self.omega_threshold:
            self.target_hold_time += self.dt
            if self.target_hold_time >= self.hold_time_threshold:
                reward += 1000.0
                done = True
        else:
            self.target_hold_time = 0.0
        if self.current_time >= self.sim_time:
            done = True
        info = {'torque': full_action.copy()}
        return obs, reward, done, info

    def render(self, mode='human'):
        super().render(mode)
