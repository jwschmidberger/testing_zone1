"""
Project: Methane Source Localization
Author: Jason Schmidberger
See agent.md for full context.
Model: pyELQ, RJMCMC, Gaussian plume
"""

import numpy as np
import matplotlib.pyplot as plt

def initialize_domain(nx: int, ny: int, value: float = 0.0) -> np.ndarray:
    """Return a ``ny`` by ``nx`` array initialised to ``value``."""
    return np.full((ny, nx), value, dtype=float)

def apply_boundary_conditions(c: np.ndarray, value: float = 0.0) -> None:
    """Apply constant-value Dirichlet boundaries in-place."""
    c[0, :] = value
    c[-1, :] = value
    c[:, 0] = value
    c[:, -1] = value    

def add_point_source(
    c: np.ndarray,
    x: int,
    y: int,
    rate_kg_per_h: float,
    dt: float,
    cell_volume_m3: float = 1.0,
) -> None:
    """Inject a point emission source given in kilograms per hour."""
    if not (0 <= x < c.shape[1] and 0 <= y < c.shape[0]):
        return
    kg_per_s = rate_kg_per_h / 3600.0
    mg_per_s = kg_per_s * 1_000_000.0
    ppm_per_s = mg_per_s / (0.714 * cell_volume_m3)
    c[y, x] += ppm_per_s * dt

def step(
    c: np.ndarray,
    dt: float,
    u: float | np.ndarray,
    v: float | np.ndarray,
    diff_coeff: float,
) -> np.ndarray:
    """Advance ``c`` one time step using a simple finite-difference scheme."""
    dx = dy = 1.0
    u_field = np.full_like(c, float(u)) if np.isscalar(u) else u
    v_field = np.full_like(c, float(v)) if np.isscalar(v) else v
    padded = np.pad(c, 1, mode="edge")
    laplacian = (
        (padded[2:, 1:-1] - 2 * padded[1:-1, 1:-1] + padded[:-2, 1:-1]) / dy**2 +
        (padded[1:-1, 2:] - 2 * padded[1:-1, 1:-1] + padded[1:-1, :-2]) / dx**2
    )
    adv_x = -u_field * (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2 * dx)
    adv_y = -v_field * (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2 * dy)
    return c + dt * (diff_coeff * laplacian + adv_x + adv_y)

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
    emission_rate_kg_per_h: float = 1.0,
    background_conc: float = 2.0,
    cell_volume_m3: float = 1.0,
) -> np.ndarray:
    """Run a forward model and return the final concentration field."""
    c = initialize_domain(nx, ny, value=background_conc)
    nsteps = int(total_time / dt)
    for _ in range(nsteps):
        add_point_source(
            c, source_x, source_y, emission_rate_kg_per_h, dt, cell_volume_m3=cell_volume_m3
        )
        c = step(c, dt, u, v, diff_coeff)
        apply_boundary_conditions(c, value=background_conc)
    return c

def _example() -> None:
    """Run a simple example using a 2 ppm background and a 5 kg/h source."""
    c = run_simulation(total_time=20.0, emission_rate_kg_per_h=5.0, background_conc=2.0)
    plt.imshow(c, origin="lower", cmap="viridis")
    plt.colorbar(label="Concentration (ppm)")
    plt.title("Final concentration field")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()

if __name__ == "__main__":
    _example()
