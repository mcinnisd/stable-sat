#!/usr/bin/env python3
"""
Satellite Attitude Simulation (Step 1)

This script simulates the 3D satellite attitude dynamics with an external torque input.
The external torque input can be chosen among different modes:
- impulse: a brief impulse at the beginning of the simulation.
- sinusoid: a sinusoidal torque input.
- step: a step function torque input.
- zero: no external torque.

The simulation integrates the satellite's attitude using quaternion representation.
Euler angles (roll, pitch, yaw) and angular velocity are recorded and plotted over time.
An optional Gym environment stub for interactive visualization is also included.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import argparse
import gym
from gym import spaces
import visualization
from torqueFunctions import external_torque_zero, EXTERNAL_TORQUE_MODES
from satelliteEnv import SatelliteAttitudeEnv

def simulate_satellite(torque_mode='impulse', sim_time=10.0, dt=0.1, inertia=[1.0, 1.0, 1.0]):
	"""
	Simulates the satellite attitude dynamics over a specified time.
	
	Parameters:
		torque_mode (str): Mode for the external torque ('impulse', 'sinusoid', 'step', 'zero').
		sim_time (float): Total simulation time in seconds.
		dt (float): Time step for simulation.
		inertia (list of float): Inertia values [Ix, Iy, Iz].
	
	Returns:
		times (np.array): Array of time stamps.
		euler_angles (np.array): Array of Euler angles (roll, pitch, yaw in degrees) over time.
		omega_history (np.array): Array of angular velocity components over time.
	"""
	I_sat = np.diag(inertia)
	I_inv = np.linalg.inv(I_sat)
	
	# Initial state: identity quaternion and zero angular velocity
	q = np.array([0, 0, 0, 1])  # [x, y, z, w]
	omega = np.zeros(3)
	
	# History storage
	times = []
	euler_angles = []  # Each entry: [roll, pitch, yaw] in degrees
	omega_history = []  # Each entry: [omega_x, omega_y, omega_z]
	
	# Select external torque function based on chosen mode
	torque_func = EXTERNAL_TORQUE_MODES.get(torque_mode, external_torque_zero)
	
	t = 0.0
	while t <= sim_time:
		# Record current state
		times.append(t)
		euler = R.from_quat(q).as_euler('xyz', degrees=True)
		euler_angles.append(euler)
		omega_history.append(omega.copy())
		
		# Compute external torque
		tau_ext = torque_func(t)
		
		# Compute angular acceleration: ω̇ = I⁻¹ * (τ_ext - ω × (I * ω))
		omega_dot = I_inv @ (tau_ext - np.cross(omega, I_sat @ omega))
		
		# Update angular velocity (Euler integration)
		omega = omega + omega_dot * dt
		
		# Update orientation (quaternion) using the rotation vector method
		delta_rot = R.from_rotvec(omega * dt)
		q = (R.from_quat(q) * delta_rot).as_quat()
		
		t += dt
		
	return np.array(times), np.array(euler_angles), np.array(omega_history)


		

def main():
	parser = argparse.ArgumentParser(description="Satellite Attitude Simulation")
	parser.add_argument('--torque_mode', type=str, default='impulse', choices=list(EXTERNAL_TORQUE_MODES.keys()),
						help="Mode for external torque: impulse, sinusoid, step, zero")
	parser.add_argument('--sim_time', type=float, default=10.0, help="Total simulation time in seconds")
	parser.add_argument('--dt', type=float, default=0.1, help="Time step for simulation")
	parser.add_argument('--inertia', type=float, nargs=3, default=[1.0, 1.0, 1.0],
						help="Satellite inertia values (Ix, Iy, Iz)")
	parser.add_argument('--interactive', action='store_true', help="Run interactive simulation using Gym environment stub")
	parser.add_argument('--save_animation', action='store_true',
						help="Save the simulation as a gif or video instead of running interactively")
	parser.add_argument('--output', type=str, default='satellite_simulation.gif',
						help="Output filename for saved animation (e.g., satellite_simulation.gif or .mp4)")
	args = parser.parse_args()
	
	if args.save_animation:
		# Save the simulation as an animation (gif or video)
		import matplotlib.animation as animation
		env = SatelliteAttitudeEnv(torque_mode=args.torque_mode, sim_time=args.sim_time, dt=args.dt, inertia=args.inertia)
		obs = env.reset()
		frames = int(env.sim_time / env.dt)
		
		def update(frame):
			obs, reward, done, _ = env.step()
			env.render()
			return []  # not using blitting
		
		ani = animation.FuncAnimation(env.fig, update, frames=frames, interval=100, blit=False)
		# For a gif, use the Pillow writer; for video, you can change writer to "ffmpeg" and adjust fps.
		ani.save(args.output, writer="pillow", fps=10)
		plt.close('all')
	elif args.interactive:
		# Run the Gym environment for interactive visualization
		env = SatelliteAttitudeEnv(torque_mode=args.torque_mode, sim_time=args.sim_time, dt=args.dt, inertia=args.inertia)
		obs = env.reset()
		done = False
		while not done:
			obs, reward, done, _ = env.step()
			env.render()
		plt.close('all')
	else:
		# Run simulation and generate static plots
		times, euler_angles, omega_history = simulate_satellite(
			torque_mode=args.torque_mode,
			sim_time=args.sim_time,
			dt=args.dt,
			inertia=args.inertia
		)
		visualization.plot_static_simulation(times, euler_angles, omega_history)

if __name__ == '__main__':
	main()