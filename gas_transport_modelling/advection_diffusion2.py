#%%

"""
Project: Methane Source Localization
Author: Jason Schmidberger
See agent.md for full context.
Model: pyELQ, RJMCMC, Gaussian plume
"""


import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


#%%
def initialize_domain(nx: int, ny: int, value: float = 0.0) -> np.ndarray:
    """Return a ``ny`` by ``nx`` array initialised to ``value``."""
    return np.full((ny, nx), value, dtype=float)


def apply_boundary_conditions(c: np.ndarray, value: float = 0.0) -> None:
    """Apply constant-value Dirichlet boundaries in-place.

    Parameters
    ----------
    c
        Concentration field to update in-place.
    value
        Boundary concentration to impose on all edges.
    """
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
    """Inject a point emission source given in kilograms per hour.

    Parameters
    ----------
    c
        Concentration field in ``ppm``.
    x, y
        Grid coordinates where the source is applied.
    rate_kg_per_h
        Emission rate in kilograms per hour.
    dt
        Time-step size in seconds.
    cell_volume_m3
        Assumed volume represented by a single grid cell.  The conversion
        from kilograms to ``ppm`` uses a simple approximation of
        ``0.714 mg`` of methane per ``ppm`` in one cubic metre at standard
        conditions.
    """

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
    emission_rate_kg_per_h: float = 1.0,
    background_conc: float = 2.0,
    cell_volume_m3: float = 1.0,
) -> np.ndarray:
    """Run a forward model and return the final concentration field.

    Parameters
    ----------
    nx, ny
        Domain dimensions in grid cells.
    dt
        Time-step size in seconds.
    total_time
        Total simulation time in seconds.
    u, v
        Wind components in m/s; can be scalars or 2-D arrays.
    diff_coeff
        Diffusion coefficient in m^2/s.
    source_x, source_y
        Grid indices of the point source location.
    emission_rate_kg_per_h
        Emission rate in kilograms per hour. This value is converted to an
        approximate change in ``ppm`` using ``cell_volume_m3``.
    background_conc
        Uniform initial concentration (e.g., 2 ppm).
    cell_volume_m3
        Volume represented by each grid cell in cubic metres. This is used
        for the simple conversion from ``kg/h`` to ``ppm`` when injecting
        the point source.

    Notes
    -----
    The solver enforces Dirichlet boundary conditions with the same value
    as ``background_conc`` at every time step.
    """

    c = initialize_domain(nx, ny, value=background_conc)

    nsteps = int(total_time / dt)
    for _ in range(nsteps):
        add_point_source(
            c,
            source_x,
            source_y,
            emission_rate_kg_per_h,
            dt,
            cell_volume_m3=cell_volume_m3,
        )
        c = step(c, dt, u, v, diff_coeff)
        apply_boundary_conditions(c, value=background_conc)

    return c


def _example() -> None:
    """Run a simple example using a 2 ppm background and a 5 kg/h source."""

    c = run_simulation(
        total_time=20.0,
        emission_rate_kg_per_h=5.0,
        background_conc=2.0,
    )
    plt.imshow(c, origin="lower", cmap="viridis")
    plt.colorbar(label="Concentration (ppm)")
    plt.title("Final concentration field")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()


def interactive_wind_example() -> None:
    """Demonstrate a Plotly slider for changing wind direction."""

    angles = np.linspace(0.0, 360.0, 13)
    frames: list[go.Frame] = []
    for angle in angles:
        u = np.cos(np.deg2rad(angle))
        v = np.sin(np.deg2rad(angle))
        c = run_simulation(u=u, v=v)
        frames.append(
            go.Frame(
                data=[go.Heatmap(z=c, colorscale="Viridis")],
                name=f"{angle:.0f}",
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames)
    steps = [
        dict(
            method="animate",
            args=[[f.name], {"mode": "immediate"}],
            label=f.name,
        )
        for f in frames
    ]
    fig.update_layout(
        title="Wind direction demo",
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        sliders=[{"steps": steps, "active": 0, "currentvalue": {"prefix": "Angle: "}}],
    )
    fig.show()


if __name__ == "__main__":
    _example()


