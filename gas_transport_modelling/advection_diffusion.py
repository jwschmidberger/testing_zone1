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


def apply_boundary_conditions(c: np.ndarray) -> None:
    """Apply simple zero-value Dirichlet boundaries in-place."""

    c[0, :] = 0.0
    c[-1, :] = 0.0
    c[:, 0] = 0.0
    c[:, -1] = 0.0


def add_point_source(c: np.ndarray, x: int, y: int, rate: float, dt: float) -> None:
    """Add a point source emission into the concentration field in-place."""

    if 0 <= x < c.shape[1] and 0 <= y < c.shape[0]:
        c[y, x] += rate * dt


def step(
    c: np.ndarray,
    dt: float,
    u: float | np.ndarray,
    v: float | np.ndarray,
    diff_coeff: float,
) -> np.ndarray:
    """Advance ``c`` one time step using a simple finite-difference scheme."""

    dx = dy = 1.0

    if np.isscalar(u):
        u_field = np.full_like(c, float(u))
    else:
        u_field = u

    if np.isscalar(v):
        v_field = np.full_like(c, float(v))
    else:
        v_field = v

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
    source_y: int = 50,
    emission_rate: float = 1.0,
) -> np.ndarray:
    """Run a forward model and return the final concentration field."""

    c = initialize_domain(nx, ny, value=2.0)  # 2 ppm initial concentration

    nsteps = int(total_time / dt)
    for _ in range(nsteps):
        add_point_source(c, source_x, source_y, emission_rate, dt)
        c = step(c, dt, u, v, diff_coeff)
        apply_boundary_conditions(c)

    return c


def _example():
    """Run a simple example and plot the result."""

    c = run_simulation()
    plt.imshow(c, origin="lower", cmap="viridis")
    plt.colorbar(label="Concentration (ppm)")
    plt.title("Final concentration field")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()


if __name__ == "__main__":
    _example()
