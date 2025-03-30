#!/usr/bin/env python3
"""
Torque Functions Module

This module contains all custom torque functions for both external disturbances and reaction wheels.
"""

import numpy as np

# Helper functions for torque calculations

def impulse_torque(t, amplitude, duration, pattern):
    """Return impulse torque based on pattern if t is less than duration, else zeros."""
    return np.array(pattern) * amplitude if t < duration else np.zeros(3)


def sinusoid_torque(t, amplitude, frequency, phase_offsets):
    """Return sinusoidal torque with given phase offsets."""
    return amplitude * np.array([np.sin(2 * np.pi * frequency * t + phase) for phase in phase_offsets])


def step_torque(t, amplitude, step_time, pattern):
    """Return step torque based on pattern if t is greater or equal to step_time, else zeros."""
    return np.array(pattern) * amplitude if t >= step_time else np.zeros(3)


# External disturbance torque functions

def external_torque_impulse(t, amplitude=0.1, duration=0.5):
    return impulse_torque(t, amplitude, duration, [1, 0, 0])


def external_torque_sinusoid(t, amplitude=0.1, frequency=1.0):
    return sinusoid_torque(t, amplitude, frequency, [0, np.pi/3, 2*np.pi/3])


def external_torque_step(t, amplitude=0.1, step_time=1.0):
    return step_torque(t, amplitude, step_time, [1, 1, 1])


def external_torque_zero(t):
    return np.zeros(3)


# Reaction wheel torque functions

def wheel_torque_impulse(t, amplitude=0.1, duration=0.5):
    return impulse_torque(t, amplitude, duration, [1, 1, 1])


def wheel_torque_sinusoid(t, amplitude=0.1, frequency=1.0):
    return sinusoid_torque(t, amplitude, frequency, [0, np.pi/4, np.pi/2])


def wheel_torque_step(t, amplitude=0.1, step_time=1.0):
    return step_torque(t, amplitude, step_time, [1, 1, 1])


def wheel_torque_zero(t):
    return np.zeros(3)


# Dictionary mappings for easy selection

EXTERNAL_TORQUE_MODES = {
    'impulse': external_torque_impulse,
    'sinusoid': external_torque_sinusoid,
    'step': external_torque_step,
    'zero': external_torque_zero
}


WHEEL_TORQUE_MODES = {
    'impulse': wheel_torque_impulse,
    'sinusoid': wheel_torque_sinusoid,
    'step': wheel_torque_step,
    'zero': wheel_torque_zero
}