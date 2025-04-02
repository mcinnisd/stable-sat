#!/usr/bin/env python3
"""
Torque Functions Module

This module contains all custom torque functions for external disturbances.
"""

import numpy as np


# External disturbance torque functions

def external_torque_impulse(t, amplitude=0.1, duration=0.5):
    return np.array([1, 0, 0]) * amplitude if t < duration else np.zeros(3)


def external_torque_sinusoid(t, amplitude=0.1, frequency=1.0):
    return amplitude * np.array([np.sin(2 * np.pi * frequency * t + phase) for phase in [0, np.pi/3, 2*np.pi/3]])


def external_torque_step(t, amplitude=0.1, step_time=1.0):
    return np.array([1, 1, 1]) * amplitude if t >= step_time else np.zeros(3)


def external_torque_zero(t):
    return np.zeros(3)

# Dictionary mappings for easy selection

EXTERNAL_TORQUE_MODES = {
    'impulse': external_torque_impulse,
    'sinusoid': external_torque_sinusoid,
    'step': external_torque_step,
    'zero': external_torque_zero
}
