"""
policy_inference_py.py -- ctypes wrapper around libpolicy_inference.so.

Compile the C side first:
    gcc -O3 -ffast-math -march=native -fPIC -shared \
        src/policy_inference.c -o src/libpolicy_inference.so -lm

The DistilledPolicy class is a drop-in for an MPC solver's .solve(x, x_ref):
it builds the 20-D observation, calls policy_forward(), and rescales the
[-1, 1] action to physical controls via the env's box bounds.
"""
import ctypes
import time
from pathlib import Path

import numpy as np

_SO = Path(__file__).resolve().parent / "libpolicy_inference.so"


# ---- C library binding --------------------------------------------------

def _lib():
    if not _SO.exists():
        raise FileNotFoundError(
            f"{_SO} missing -- see the module docstring for the build command."
        )
    lib = ctypes.CDLL(str(_SO))
    lib.policy_forward.argtypes = [ctypes.POINTER(ctypes.c_float),
                                   ctypes.POINTER(ctypes.c_float)]
    lib.policy_forward.restype = None
    return lib


# ---- Drop-in solver wrapper ---------------------------------------------

class DistilledPolicy:
    """20-D obs -> 4-D action in [-1, 1] -> physical control via box rescale.

    Obs layout matches what dagger.py produces during training:
        [0:12]  full state  [pos, vel, euler, omega]
        [12:15] tracking error  state[0:3] - ref[0:3]
        [15:18] reference velocity
        [18:20] reference attitude (roll_ref, pitch_ref)
    """

    def __init__(self, u_min, u_max):
        self._lib = _lib()
        self.u_mid = 0.5 * (np.asarray(u_max) + np.asarray(u_min))
        self.u_half = 0.5 * (np.asarray(u_max) - np.asarray(u_min))
        self._obs = np.zeros(20, dtype=np.float32)
        self._act = np.zeros(4, dtype=np.float32)

    def solve(self, x, x_ref):
        o = self._obs
        o[0:12] = x
        o[12:15] = x[0:3] - x_ref[0:3, 0]
        o[15:18] = x_ref[3:6, 0]
        o[18:20] = x_ref[6:8, 0]

        t0 = time.perf_counter()
        self._lib.policy_forward(
            o.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        solve_us = (time.perf_counter() - t0) * 1e6

        u = self.u_mid + self.u_half * self._act.astype(np.float64)
        return u, {"solve_time_us": solve_us, "status": "ok"}
