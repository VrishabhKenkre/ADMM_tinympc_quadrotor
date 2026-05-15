"""ctypes wrapper around src/libpolicy_inference.so.

The 5,764-parameter distilled MLP runs as a single C call: 20-D float obs in,
4-D float action in [-1,1] out. The action is mapped to physical Crazyflie
controls (thrust + 3 torques) using the env's box-bound midpoint/half-range.

The obs layout matches what dagger.py produces during training:
    [0:12]  full state  [px,py,pz, vx,vy,vz, roll,pitch,yaw, wx,wy,wz]
    [12:15] tracking error  state[0:3] - ref[0:3]
    [15:18] reference velocity [vrx, vry, vrz]
    [18:20] reference attitude [roll_ref, pitch_ref]
"""
from __future__ import annotations

import ctypes
import time
from pathlib import Path

import numpy as np

_SO_PATH = Path(__file__).resolve().parent / "libpolicy_inference.so"


def _load_lib():
    if not _SO_PATH.exists():
        raise FileNotFoundError(
            f"{_SO_PATH} missing. Compile with:\n"
            "  gcc -O3 -ffast-math -march=native -fPIC -shared "
            "src/policy_inference.c -o src/libpolicy_inference.so -lm"
        )
    lib = ctypes.CDLL(str(_SO_PATH))
    lib.policy_forward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.policy_forward.restype = None
    return lib


class DistilledPolicy:
    """Drop-in replacement for an MPC solver's ``.solve(x, x_ref)``.

    Calls into ``libpolicy_inference.so`` (5,764-param MLP, 20-D obs, 4-D
    action in [-1, 1]), then maps the normalized action to physical controls
    via ``u = u_mid + u_half * action``.
    """

    def __init__(self, u_min: np.ndarray, u_max: np.ndarray):
        self._lib = _load_lib()
        self.u_min = np.asarray(u_min, dtype=np.float64)
        self.u_max = np.asarray(u_max, dtype=np.float64)
        self.u_mid = 0.5 * (self.u_max + self.u_min)
        self.u_half = 0.5 * (self.u_max - self.u_min)
        self._obs_buf = np.zeros(20, dtype=np.float32)
        self._act_buf = np.zeros(4, dtype=np.float32)

    def _build_obs(self, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        o = self._obs_buf
        o[0:12] = x
        o[12:15] = x[0:3] - x_ref[0:3, 0]
        o[15:18] = x_ref[3:6, 0]
        o[18:20] = x_ref[6:8, 0]
        return o

    def solve(self, x: np.ndarray, x_ref: np.ndarray):
        obs = self._build_obs(x, x_ref)
        t0 = time.perf_counter()
        self._lib.policy_forward(
            obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._act_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        solve_us = (time.perf_counter() - t0) * 1e6
        u = self.u_mid + self.u_half * self._act_buf.astype(np.float64)
        return u, {"solve_time_us": solve_us, "status": "ok"}
