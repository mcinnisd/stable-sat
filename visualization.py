#!/usr/bin/env python3
"""
Visualization Module

This module contains functions for drawing the satellite in a 3D matplotlib axis,
plotting static simulation data, and additional plotting functions such as for reaction wheel speeds.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def draw_satellite(ax, q, inertia, current_time):
    """
    Draws the satellite on the provided 3D axis.
    
    Parameters:
        ax: matplotlib 3D axis.
        q: Current quaternion (array-like) for the satellite orientation.
        inertia: Inertia values [Ix, Iy, Iz] used to scale the satellite's shape.
        current_time: Simulation time (displayed in the title).
    """
    ax.cla()  # Clear dynamic content
    origin = np.array([0, 0, 0])
    # Draw inertial frame axes
    ax.quiver(*origin, 1, 0, 0, color='r', length=1, normalize=True)
    ax.quiver(*origin, 0, 1, 0, color='g', length=1, normalize=True)
    ax.quiver(*origin, 0, 0, 1, color='b', length=1, normalize=True)
    
    # Draw satellite as a rectangular prism scaled by inertia
    avg_inertia = np.mean(inertia)
    l_x = 0.5 * (inertia[0] / avg_inertia)
    l_y = 0.5 * (inertia[1] / avg_inertia)
    l_z = 0.5 * (inertia[2] / avg_inertia)
    
    cube_points = np.array([
        [-l_x, -l_y, -l_z],
        [-l_x, -l_y,  l_z],
        [-l_x,  l_y,  l_z],
        [-l_x,  l_y, -l_z],
        [ l_x, -l_y, -l_z],
        [ l_x, -l_y,  l_z],
        [ l_x,  l_y,  l_z],
        [ l_x,  l_y, -l_z],
    ])
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    R_body = R.from_quat(q).as_matrix()
    cube_points_world = (R_body @ cube_points.T).T
    for edge in edges:
        p1, p2 = cube_points_world[edge[0]], cube_points_world[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='k')
    
    # Draw satellite body axes
    body_x = R_body @ np.array([1, 0, 0])
    body_y = R_body @ np.array([0, 1, 0])
    body_z = R_body @ np.array([0, 0, 1])
    ax.quiver(*origin, *body_x, color='m', length=0.8, normalize=True)
    ax.quiver(*origin, *body_y, color='c', length=0.8, normalize=True)
    ax.quiver(*origin, *body_z, color='y', length=0.8, normalize=True)
    
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_title(f"Satellite Orientation at t = {current_time:.2f}s")

def plot_static_simulation(times, euler_angles, omega_history):
    """
    Plots static graphs for the satellite simulation.
    
    Plots:
      - Euler angles (roll, pitch, yaw) over time.
      - Angular velocity components over time.
    """
    roll = euler_angles[:, 0]
    pitch = euler_angles[:, 1]
    yaw = euler_angles[:, 2]
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].plot(times, roll, label="Roll")
    axs[0].plot(times, pitch, label="Pitch")
    axs[0].plot(times, yaw, label="Yaw")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Angle (deg)")
    axs[0].set_title("Satellite Euler Angles Over Time")
    axs[0].legend()
    
    omega_history = np.array(omega_history)
    axs[1].plot(times, omega_history[:, 0], label="$\omega_x$")
    axs[1].plot(times, omega_history[:, 1], label="$\omega_y$")
    axs[1].plot(times, omega_history[:, 2], label="$\omega_z$")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Angular Velocity (rad/s)")
    axs[1].set_title("Satellite Angular Velocity Over Time")
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()
    
def annotate_wheel_speeds(ax, wheel_speeds):
    """Annotates the reaction wheel speeds as text on the given axis."""
    ax.text2D(0.05, 0.95, f"Wheel speeds: {np.round(wheel_speeds, 3)}", transform=ax.transAxes)

def plot_wheel_speeds(times, wheel_speeds):
    """
    Plots the reaction wheel speeds over time.
    """
    plt.figure()
    wheel_speeds = np.array(wheel_speeds)
    plt.plot(times, wheel_speeds[:, 0], label="Wheel x")
    plt.plot(times, wheel_speeds[:, 1], label="Wheel y")
    plt.plot(times, wheel_speeds[:, 2], label="Wheel z")
    plt.xlabel("Time (s)")
    plt.ylabel("Wheel Speed (rad/s)")
    plt.title("Reaction Wheel Speeds Over Time")
    plt.legend()
    plt.show()