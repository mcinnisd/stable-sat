#!/usr/bin/env python3
"""
Control Policies for Satellite Attitude Control

This module defines different control strategies (PID, LQR, etc.) as callable classes.
Each controller is designed to be injected into the SatelliteEnv.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import torqueFunctions  # assumes you have a module with your torque functions

class PIDController:
    """
    PID Controller for Satellite Attitude Control.
    Implements a PID control law with proportional, derivative, and integral terms.
    """
    def __init__(self, k_R=1.5, k_omega=10.0, k_I=0.0,
                 hold_time_threshold=1.0, orientation_error_threshold=0.15,
                 disturbance_time=2.0, desired_points=None, max_torque=2.0):
        self.k_R = k_R
        self.k_omega = k_omega
        self.k_I = k_I
        self.hold_time_threshold = hold_time_threshold
        self.orientation_error_threshold = orientation_error_threshold
        self.disturbance_time = disturbance_time
        self.max_torque = max_torque
        
        # Define desired orientations as quaternions.
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
        self.error_integral = np.zeros(3)
    
    def __call__(self, env):
        # During disturbance phase, use the external disturbance torque.
        if env.current_time < self.disturbance_time:
            return torqueFunctions.external_torque_step(env.current_time)
        # Compute control torque using PID law.
        q_desired = self.desired_points[self.current_target_idx]
        R_d = R.from_quat(q_desired).as_matrix()
        R_current = R.from_quat(env.q).as_matrix()
        R_err = R_d.T @ R_current - R_current.T @ R_d
        e_R = 0.5 * np.array([R_err[2, 1], R_err[0, 2], R_err[1, 0]])
        
        # Update hold time logic for switching target points.
        if np.linalg.norm(e_R) < self.orientation_error_threshold:
            self.target_hold_time += env.dt
            if self.target_hold_time >= self.hold_time_threshold:
                print(f"Time {env.current_time:.2f}: success: Reached control point {self.current_target_idx}")
                if self.current_target_idx < len(self.desired_points) - 1:
                    self.current_target_idx += 1
                    print(f"Time {env.current_time:.2f}: now heading to control point {self.current_target_idx}")
                self.target_hold_time = 0.0
        else:
            self.target_hold_time = 0.0
        
        # Update the integral term.
        self.error_integral += e_R * env.dt
        self.error_integral = np.clip(self.error_integral, -5.0, 5.0)
        
        # Compute PID control torque.
        u = -self.k_R * e_R - self.k_omega * env.omega - self.k_I * self.error_integral
        u = np.clip(u, -self.max_torque, self.max_torque)
        # Return computed control torque (environment applies -u).
        return -u

class LQRController:
    """
    LQR Controller for Satellite Attitude Control.
    Computes control torque using decoupled LQR gains.
    """
    def __init__(self, Q=np.diag([10, 1]), R_mat=np.array([[10]]),
                 hold_time_threshold=1.0, orientation_error_threshold=0.15,
                 disturbance_time=2.0, desired_points=None, max_torque=2.0):
        self.Q = Q
        self.R_mat = R_mat
        self.hold_time_threshold = hold_time_threshold
        self.orientation_error_threshold = orientation_error_threshold
        self.disturbance_time = disturbance_time
        
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
        self.K = None  # LQR gains will be computed on first call.
        self.max_torque = max_torque
    
    def __call__(self, env):
        # During disturbance phase, use external disturbance torque.
        if env.current_time < self.disturbance_time:
            return torqueFunctions.external_torque_step(env.current_time)
        q_desired = self.desired_points[self.current_target_idx]
        R_d = R.from_quat(q_desired).as_matrix()
        R_current = R.from_quat(env.q).as_matrix()
        R_err = R_d.T @ R_current - R_current.T @ R_d
        e_R = 0.5 * np.array([R_err[2, 1], R_err[0, 2], R_err[1, 0]])
        
        if np.linalg.norm(e_R) < self.orientation_error_threshold:
            self.target_hold_time += env.dt
            if self.target_hold_time >= self.hold_time_threshold:
                print(f"Time {env.current_time:.2f}: success: Reached control point {self.current_target_idx}")
                if self.current_target_idx < len(self.desired_points) - 1:
                    self.current_target_idx += 1
                    print(f"Time {env.current_time:.2f}: now heading to control point {self.current_target_idx}")
                self.target_hold_time = 0.0
        else:
            self.target_hold_time = 0.0
        
        # Compute LQR control torque for each axis.
        u = np.zeros(3)
        # Compute LQR gains on first call.
        if self.K is None:
            from dynamics import compute_lqr_gain
            self.K = np.zeros((3, 2))
            for i in range(3):
                self.K[i, :] = compute_lqr_gain(env.inertia[i], self.Q, self.R_mat)
        for i in range(3):
            x_i = np.array([e_R[i], env.omega[i]])
            u[i] = -self.K[i, :].dot(x_i)
        
        u = np.clip(u, -self.max_torque, self.max_torque)
        return -u