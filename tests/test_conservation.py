import numpy as np


def run_simulation(num_steps=20, num_snapshots=5):
    """Run a very small linear advection simulation."""
    L = 1.0
    nx = 100
    dx = L / nx
    dt = 0.4 * dx  # CFL number 0.4

    x = np.linspace(0, L, nx, endpoint=False)
    n = np.sin(2 * np.pi * x)
    u = np.ones(nx)
    T = np.ones(nx)

    snapshot_interval = max(1, num_steps // num_snapshots)
    nint = []
    uint = []
    Tint = []
    n_snap = []
    u_snap = []
    T_snap = []

    for step in range(num_steps + 1):
        if step % snapshot_interval == 0:
            n_snap.append(n.copy())
            u_snap.append(u.copy())
            T_snap.append(T.copy())
            nint.append(np.sum(n) * dx)
            uint.append(np.sum(u) * dx)
            Tint.append(np.sum(T) * dx)

        # Upwind update for n with constant velocity u=1
        n = n - dt / dx * (n - np.roll(n, 1))
        # u and T remain constant in this simple model

    return (
        np.array(nint),
        np.array(uint),
        np.array(Tint),
        np.array(n_snap),
        np.array(u_snap),
        np.array(T_snap),
    )


def test_conservation():
    nint, uint, Tint, *_ = run_simulation()
    tol = 1e-6
    for arr in (nint, uint, Tint):
        baseline = arr[0]
        for val in arr:
            assert abs(val - baseline) < tol


def test_stability():
    _, _, _, n_snap, u_snap, T_snap = run_simulation()
    for arr in (n_snap, u_snap, T_snap):
        assert np.isfinite(arr).all()
