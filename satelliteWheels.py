#!/usr/bin/env python3
"""
Satellite Attitude Simulation with Reaction Wheels

Uses the refactored SatelliteEnvWithWheels to run the simulation.
"""

import argparse
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import visualization
from satelliteEnv import SatelliteEnvWithWheels

def simulate_satellite_with_wheels(env):
    """
    Runs the simulation loop using the environment to collect state data.
    """
    times = []
    euler_angles = []
    omega_history = []
    wheel_speeds_history = []
    
    done = False
    while not done:
        times.append(env.current_time)
        obs = env._get_obs()  # Observation: first 3 are Euler angles, next 3 are angular velocities
        euler_angles.append(obs[:3])
        omega_history.append(obs[3:])
        wheel_speeds_history.append(env.omega_w.copy())
        _, _, done, _ = env.step()
        
    return (np.array(times), np.array(euler_angles),
            np.array(omega_history), np.array(wheel_speeds_history))

def main():
    parser = argparse.ArgumentParser(description="Satellite Simulation with Reaction Wheels")
    parser.add_argument('--wheel_torque_mode', type=str, default='impulse', choices=['impulse','sinusoid','step','zero'],
                        help="Reaction wheel torque mode")
    parser.add_argument('--sim_time', type=float, default=10.0, help="Total simulation time (s)")
    parser.add_argument('--dt', type=float, default=0.1, help="Time step (s)")
    parser.add_argument('--inertia', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Satellite inertia values (Ix, Iy, Iz)")
    parser.add_argument('--I_w', type=float, default=0.01, help="Reaction wheel inertia")
    parser.add_argument('--interactive', action='store_true', help="Run interactive simulation")
    parser.add_argument('--save_animation', action='store_true', help="Save animation as gif/video")
    parser.add_argument('--output', type=str, default='satellite_simulation_wheels.gif', help="Output filename")
    args = parser.parse_args()
    
    # Initialize the environment
    env = SatelliteEnvWithWheels(sim_time=args.sim_time, dt=args.dt, inertia=args.inertia, I_w=args.I_w,
                                 wheel_torque_mode=args.wheel_torque_mode)
    
    if args.save_animation:
        env.reset()
        frames = int(env.sim_time / env.dt)
        def update(frame):
            env.step()
            env.render()
            return []
        ani = animation.FuncAnimation(env.fig, update, frames=frames, interval=100, blit=False)
        ani.save(args.output, writer="pillow", fps=10)
        plt.close('all')
    elif args.interactive:
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step()
            env.render()
        plt.close('all')
    else:
        env.reset()
        times, euler_angles, omega_history, wheel_speeds = simulate_satellite_with_wheels(env)
        visualization.plot_static_simulation(times, euler_angles, omega_history)
        visualization.plot_wheel_speeds(times, wheel_speeds)

if __name__ == '__main__':
    main()