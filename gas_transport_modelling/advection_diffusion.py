"""Simple 2-D advection–diffusion forward model.

This module provides a minimal solver that can be used to explore the
transport of a tracer in a horizontal domain.  It assumes a regular grid
with a uniform spacing of 1 m.  Wind components (``u`` and ``v``) may be
passed in as scalars or 2-D arrays matching the concentration field.

The solver is intentionally lightweight and is meant for experimentation or
educational purposes rather than production use.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def initialize_domain(nx: int, ny: int, value: float = 0.0) -> np.ndarray:
    """Return a ``ny`` by ``nx`` array initialised to ``value``."""

    return np.full((ny, nx), value, dtype=float)

    # pad with edge values to approximate zero-gradient boundaries
    padded = np.pad(c, 1, mode="edge")

    laplacian = (
        padded[2:, 1:-1] - 2 * padded[1:-1, 1:-1] + padded[:-2, 1:-1]
    ) / dy**2 + (
        padded[1:-1, 2:] - 2 * padded[1:-1, 1:-1] + padded[1:-1, :-2]
    ) / dx**2

    adv_x = -u_field * (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2 * dx)
    adv_y = -v_field * (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2 * dy)

    c_new = c + dt * (diff_coeff * laplacian + adv_x + adv_y)
    return c_new


def run_simulation(
    nx: int = 100,
    ny: int = 100,
    dt: float = 0.1,
    total_time: float = 20.0,
    u: float | np.ndarray = 1.0,
    v: float | np.ndarray = 0.5,
    diff_coeff: float = 0.1,
    source_x: int = 50,
    source_y: int = 50

    return c

    plt.imshow(c, origin="lower", cmap="viridis")
    plt.colorbar(label="Concentration (ppm)")
    plt.title("Final concentration field")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()


if __name__ == "__main__":
    _example()
