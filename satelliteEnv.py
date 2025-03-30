#!/usr/bin/env python3
"""
Satellite Environment Module

This module contains Gym environment classes for satellite simulations.
"""

import numpy as np
import gym
from gym import spaces
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import visualization
import torqueFunctions
from dynamics import update_state, compute_lqr_gain

def normalize_quaternion(q):
	norm = np.linalg.norm(q)
	if norm == 0:
		return q
	return q / norm

class SatelliteAttitudeEnv(gym.Env):
	"""
	A basic satellite environment without reaction wheels.
	"""
	def __init__(self, torque_mode='impulse', sim_time=10.0, dt=0.1, inertia=[1.0, 1.0, 1.0]):
		super(SatelliteAttitudeEnv, self).__init__()
		self.sim_time = sim_time
		self.dt = dt
		self.torque_mode = torque_mode
		self.inertia = inertia
		self.I_sat = np.diag(inertia)
		self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
		self.fig = None
		self.ax = None
		self.reset()
		
	def reset(self):
		self.q = np.array([0, 0, 0, 1])
		self.omega = np.zeros(3)
		self.current_time = 0.0
		return self._get_obs()

	def _init_plot(self):
		if self.fig is None or self.ax is None:
			self.fig = plt.figure()
			self.ax = self.fig.add_subplot(111, projection='3d')
	
	def step(self, action=None):
		torque_func = torqueFunctions.EXTERNAL_TORQUE_MODES.get(self.torque_mode, torqueFunctions.external_torque_zero)
		tau_ext = torque_func(self.current_time)
		self.q, self.omega = update_state(self.q, self.omega, self.dt, tau_ext, self.I_sat)
		
		self.current_time += self.dt
		done = self.current_time >= self.sim_time
		return self._get_obs(), 0.0, done, {}
	
	def _get_obs(self):
		euler = R.from_quat(self.q).as_euler('xyz', degrees=True)
		return np.concatenate([euler, self.omega])
	
	def render(self, mode='human'):
		self._init_plot()
		visualization.draw_satellite(self.ax, self.q, self.inertia, self.current_time)
		plt.draw()
		plt.pause(0.1)

class SatelliteEnvWithWheels(SatelliteAttitudeEnv):
	"""
	Satellite environment that includes reaction wheel dynamics.
	"""
	def __init__(self, sim_time=10.0, dt=0.1, inertia=[1.0,1.0,1.0], I_w=0.01,
				wheel_torque_mode='impulse'):
		super(SatelliteEnvWithWheels, self).__init__(torque_mode='impulse', sim_time=sim_time, dt=dt, inertia=inertia)
		self.I_w = I_w
		self.wheel_torque_mode = wheel_torque_mode
		self.omega_w = np.zeros(3)
	
	def reset(self):
		super().reset()
		self.omega_w = np.zeros(3)
		return self._get_obs()
	
	def step(self, action=None):
		# Reaction wheel torque
		torque_func = torqueFunctions.WHEEL_TORQUE_MODES.get(
			self.wheel_torque_mode, torqueFunctions.wheel_torque_zero)
		tau_rw = torque_func(self.current_time)
		# Satellite dynamics with reaction wheel torque (note the sign)
		self.q, self.omega = update_state(self.q, self.omega, self.dt, -tau_rw, self.I_sat)
		# Update reaction wheel speeds (using simple Euler integration)
		self.omega_w = self.omega_w + (tau_rw / self.I_w) * self.dt
		
		self.current_time += self.dt
		done = self.current_time >= self.sim_time
		return self._get_obs(), 0.0, done, {}
	
	
	def render(self, mode='human'):
		self._init_plot()
		visualization.draw_satellite(self.ax, self.q, self.inertia, self.current_time)
		visualization.annotate_wheel_speeds(self.ax, self.omega_w)
		plt.draw()
		plt.pause(0.1)
		

class SatelliteEnvWithPID(SatelliteEnvWithWheels):
	"""
	Satellite environment using PID for attitude stabilization.
	Inherits from SatelliteEnvWithWheels.
	"""
	def __init__(self, sim_time=100.0, dt=0.1, inertia=[1.0, 1.0, 1.0], I_w=0.05,
				k_R=1.5, k_omega=10, k_I = .0,
				disturbance_time=2.0, disturbance_torque_mode='impulse',
				desired_points=None, hold_time_threshold=1.0, orientation_error_threshold=0.15):
		# Set default desired orientations if not provided.
		if desired_points is None:
			desired_points = [
				np.array([0, 0, 0, 1]),
				R.from_euler('z', 90, degrees=True).as_quat(),
				R.from_euler('y', 90, degrees=True).as_quat(),
				R.from_euler('x', 90, degrees=True).as_quat()
			]
		self.desired_points = [p / np.linalg.norm(p) for p in desired_points]
		self.current_target_idx = 0
		self.target_hold_time = 0.0
		self.hold_time_threshold = hold_time_threshold
		self.orientation_error_threshold = orientation_error_threshold

		# Initialize parent class.
		super().__init__(sim_time=sim_time, dt=dt, inertia=inertia, I_w=I_w)

		# Remove LQR gains; use PD gains instead
		self.k_R = k_R
		self.k_omega = k_omega
		self.k_I = k_I
		
		self.disturbance_time = disturbance_time
		self.disturbance_torque_mode = disturbance_torque_mode
		self.q_desired = self.desired_points[self.current_target_idx]

		self.error_integral = np.zeros(3)

	def reset(self, initial_q=None, initial_omega=None):
		if initial_q is not None:
			self.q = normalize_quaternion(initial_q)
		else:
			self.q = np.array([0, 0, 0, 1])
		self.omega = initial_omega if initial_omega is not None else np.zeros(3)
		self.current_time = 0.0
		self.omega_w = np.zeros(3)
		self.current_target_idx = 0
		self.target_hold_time = 0.0
		self.q_desired = self.desired_points[self.current_target_idx]
		return self._get_obs()
	
	def get_disturbance_torque(self):
		"""Return external disturbance torque during the disturbance phase."""
		torque_func = torqueFunctions.EXTERNAL_TORQUE_MODES.get(
			self.disturbance_torque_mode, torqueFunctions.external_torque_step)
		return torque_func(self.current_time)
	
	def compute_pid_torque(self):
		# Convert desired and current quaternions to rotation matrices.
		R_d = R.from_quat(self.q_desired).as_matrix()
		R_current = R.from_quat(self.q).as_matrix()

		# Compute the skew-symmetric error matrix.
		R_err = R_d.T @ R_current - R_current.T @ R_d
		# The attitude error vector is half the vee-map of R_err.
		e_R = 0.5 * np.array([R_err[2, 1], R_err[0, 2], R_err[1, 0]])
		
		# Debug print
		# print(f"DEBUG: e_R (rotation matrix error) = {e_R}")
		
		# Setpoint update logic: if error is below threshold, accumulate hold time
		if np.linalg.norm(e_R) < self.orientation_error_threshold:
			self.target_hold_time += self.dt
			if self.target_hold_time >= self.hold_time_threshold:
				print(f"Time {self.current_time:.2f}: success: Reached control point {self.current_target_idx}")
				if self.current_target_idx < len(self.desired_points) - 1:
					self.current_target_idx += 1
					self.q_desired = self.desired_points[self.current_target_idx]
					print(f"Time {self.current_time:.2f}: now heading to control point {self.current_target_idx}")
				self.target_hold_time = 0.0
		else:
			self.target_hold_time = 0.0
		
		# Update the integral term and clamp it.
		self.error_integral += e_R * self.dt
		self.error_integral = np.clip(self.error_integral, -5.0, 5.0)

		# PID control law
		u = -self.k_R * e_R - self.k_omega * self.omega - self.k_I * self.error_integral

		max_torque = 2.0
		u = np.clip(u, -max_torque, max_torque)
		# print(f"DEBUG: control torque u = {u}")
		# Note: the calling function applies -u, so return -u
		return -u
		
	def step(self, action=None):
		if self.current_time < self.disturbance_time:
			tau = self.get_disturbance_torque()
		else:
			tau = self.compute_pid_torque()
		
		# Update satellite dynamics using the common update_state function.
		# Note: The satellite experiences -tau from the reaction wheels.
		self.q, self.omega = update_state(self.q, self.omega, self.dt, -tau, self.I_sat)
		self.q = normalize_quaternion(self.q)
		
		# Update reaction wheel speeds.
		self.omega_w = self.omega_w + (tau / self.I_w) * self.dt
		
		self.current_time += self.dt
		done = self.current_time >= self.sim_time
		return self._get_obs(), 0.0, done, {}

class SatelliteEnvWithLQR(SatelliteEnvWithWheels):
	"""
	Satellite environment using LQR for attitude stabilization.
	Inherits from SatelliteEnvWithWheels.
	This implementation uses a rotation matrix error (via the vee-map) and decoupled LQR gains
	computed for each axis.
	"""
	def __init__(self, sim_time=100.0, dt=0.1, inertia=[1.0, 1.0, 1.0], I_w=0.05,
                 Q=np.diag([10, 1]), R_mat=np.array([[10]]),
                 disturbance_time=2.0, disturbance_torque_mode='impulse',
                 desired_points=None, hold_time_threshold=1.0, orientation_error_threshold=0.15):
		if desired_points is None:
			desired_points = [
				np.array([0, 0, 0, 1]),
				R.from_euler('z', 90, degrees=True).as_quat(),
				R.from_euler('y', 90, degrees=True).as_quat(),
				R.from_euler('x', 90, degrees=True).as_quat()
			]
		self.desired_points = [p / np.linalg.norm(p) for p in desired_points]
		self.current_target_idx = 0
		self.target_hold_time = 0.0
		self.hold_time_threshold = hold_time_threshold
		self.orientation_error_threshold = orientation_error_threshold
		
		super().__init__(sim_time=sim_time, dt=dt, inertia=inertia, I_w=I_w)
		
		# Pre-compute LQR gains for each axis (assuming decoupled dynamics).
		# Here we assume a function compute_lqr_gain is available.
		self.K = np.zeros((3, 2))
		for i in range(3):
			self.K[i, :] = compute_lqr_gain(inertia[i], Q, R_mat)
		
		self.disturbance_time = disturbance_time
		self.disturbance_torque_mode = disturbance_torque_mode
		self.q_desired = self.desired_points[self.current_target_idx]
		
	def reset(self, initial_q=None, initial_omega=None):
		if initial_q is not None:
			self.q = normalize_quaternion(initial_q)
		else:
			self.q = np.array([0, 0, 0, 1])
		self.omega = initial_omega if initial_omega is not None else np.zeros(3)
		self.current_time = 0.0
		self.omega_w = np.zeros(3)
		self.current_target_idx = 0
		self.target_hold_time = 0.0
		self.q_desired = self.desired_points[self.current_target_idx]
		return self._get_obs()
	
	def get_disturbance_torque(self):
		torque_func = torqueFunctions.EXTERNAL_TORQUE_MODES.get(
			self.disturbance_torque_mode, torqueFunctions.external_torque_step)
		return torque_func(self.current_time)
	
	def compute_lqr_torque(self):
		# Compute rotation matrix error using desired and current quaternions
		R_d = R.from_quat(self.q_desired).as_matrix()
		R_current = R.from_quat(self.q).as_matrix()
		R_err = R_d.T @ R_current - R_current.T @ R_d
		# Attitude error vector using vee-map (half the skew symmetric part)
		e_R = 0.5 * np.array([R_err[2, 1], R_err[0, 2], R_err[1, 0]])
		print(f"DEBUG: e_R (LQR error) = {e_R}")
		
		# Check if target reached (optional logic based on orientation_error_threshold)
		if np.linalg.norm(e_R) < self.orientation_error_threshold:
			self.target_hold_time += self.dt
			if self.target_hold_time >= self.hold_time_threshold:
				print(f"Time {self.current_time:.2f}: success: Reached control point {self.current_target_idx}")
				if self.current_target_idx < len(self.desired_points) - 1:
					self.current_target_idx += 1
					self.q_desired = self.desired_points[self.current_target_idx]
					print(f"Time {self.current_time:.2f}: now heading to control point {self.current_target_idx}")
				self.target_hold_time = 0.0
		else:
			self.target_hold_time = 0.0
		
		# Compute control torque using decoupled LQR for each axis.
		u = np.zeros(3)
		for i in range(3):
			# Construct state vector: [error_i, omega_i]
			x_i = np.array([e_R[i], self.omega[i]])
			u[i] = -self.K[i, :].dot(x_i)
		return -u
	
	def step(self, action=None):
		if self.current_time < self.disturbance_time:
			tau = self.get_disturbance_torque()
		else:
			tau = self.compute_lqr_torque()
		
		# Update satellite dynamics (note the sign convention for reaction wheel torque).
		self.q, self.omega = update_state(self.q, self.omega, self.dt, -tau, self.I_sat)
		self.q = normalize_quaternion(self.q)
		
		# Update reaction wheel speeds.
		self.omega_w = self.omega_w + (tau / self.I_w) * self.dt
		
		self.current_time += self.dt
		done = self.current_time >= self.sim_time
		return self._get_obs(), 0.0, done, {}