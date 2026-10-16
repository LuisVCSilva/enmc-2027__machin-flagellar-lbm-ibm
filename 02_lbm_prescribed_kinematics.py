#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MODEL II — LBM PRESCRIBED KINEMATICS
Machin waveform + D2Q9 LBM + direct-forcing IBM

The flagellar deformation is prescribed from the converged finite-length
Machin boundary-value problem.

This model does NOT solve fluid-structure interaction:

    Model I   -> prescribed flagellar waveform
    Model II  -> hydrodynamic response of the prescribed waveform

The configuration file is supplied explicitly:

    python3 02_lbm_prescribed_kinematics.py exp_1.json

This script DOES NOT generate figures.

It exports raw numerical data to CSV files. A separate script,
03_figures_article.py, is responsible for generating all article figures.

Outputs
-------

results_model_II_machin_lbm/
    machin_reference.csv
    flagellum_kinematics.csv
    tip_history.csv
    lbm_history.csv
    final_field.csv
    parameters.csv
"""


from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp


# ============================================================================
# COMMAND LINE
# ============================================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Model II — D2Q9 LBM with prescribed Machin flagellar "
            "kinematics and direct-forcing IBM."
        )
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to the JSON configuration file.",
    )

    return parser.parse_args()


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_configuration(
    config_file: Path,
) -> tuple[dict, Path]:

    config_file = config_file.resolve()

    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found:\n{config_file}"
        )

    if config_file.suffix.lower() != ".json":
        raise ValueError(
            f"Configuration file must be JSON:\n{config_file}"
        )

    with config_file.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)

    if "model_I" not in config:
        raise KeyError(
            "Missing 'model_I' section."
        )

    if "model_II" not in config:
        raise KeyError(
            "Missing 'model_II' section."
        )

    return config, config_file.parent


# ============================================================================
# MODEL I — MACHIN BVP
# ============================================================================

def machin_ode(
    chi: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Real representation of

        F'''' + i F = 0.

    With

        F = a + i b,

    therefore

        a'''' = b
        b'''' = -a.
    """

    dydchi = np.zeros_like(y)

    # Real part
    dydchi[0] = y[1]
    dydchi[1] = y[2]
    dydchi[2] = y[3]
    dydchi[3] = y[4]

    # Imaginary part
    dydchi[4] = y[5]
    dydchi[5] = y[6]
    dydchi[6] = y[7]
    dydchi[7] = -y[0]

    return dydchi


def machin_bc(
    ya: np.ndarray,
    yb: np.ndarray,
) -> np.ndarray:
    """
    Boundary conditions:

        F(0)    = 1
        F''(0)  = 0
        F''(Λ)  = 0
        F'''(Λ) = 0
    """

    return np.array(
        [
            ya[0] - 1.0,
            ya[4],

            ya[2],
            ya[6],

            yb[2],
            yb[6],

            yb[3],
            yb[7],
        ]
    )


def solve_machin(
    Lambda: float,
    n_mesh: int,
    tol: float,
    max_nodes: int,
):

    chi = np.linspace(
        0.0,
        Lambda,
        n_mesh,
    )

    # Simple initial guess.
    #
    # The BVP solver determines the final solution. The initial
    # condition only provides a starting approximation.

    y_guess = np.zeros(
        (
            8,
            chi.size,
        )
    )

    y_guess[0] = 1.0

    solution = solve_bvp(
        machin_ode,
        machin_bc,
        chi,
        y_guess,
        tol=tol,
        max_nodes=max_nodes,
    )

    if not solution.success:

        raise RuntimeError(
            "Machin BVP did not converge:\n"
            f"{solution.message}"
        )

    return solution


# ============================================================================
# PESKIN FOUR-POINT DELTA
# ============================================================================

def delta_kernel(
    r: np.ndarray,
) -> np.ndarray:
    """
    Peskin four-point regularized delta kernel.
    """

    q = np.abs(r)

    result = np.zeros_like(q)

    mask1 = q < 1.0

    result[mask1] = (
        1.0 / 8.0
        * (
            3.0
            - 2.0 * q[mask1]
            + np.sqrt(
                1.0
                + 4.0 * q[mask1]
                - 4.0 * q[mask1] ** 2
            )
        )
    )

    mask2 = (
        (q >= 1.0)
        & (q < 2.0)
    )

    result[mask2] = (
        1.0 / 8.0
        * (
            5.0
            - 2.0 * q[mask2]
            - np.sqrt(
                -7.0
                + 12.0 * q[mask2]
                - 4.0 * q[mask2] ** 2
            )
        )
    )

    return result


# ============================================================================
# EQUILIBRIUM
# ============================================================================

def equilibrium(
    rho: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    c: np.ndarray,
    w: np.ndarray,
    cs2: float,
) -> np.ndarray:

    Ny, Nx = rho.shape

    feq = np.zeros(
        (
            9,
            Ny,
            Nx,
        )
    )

    u2 = (
        ux ** 2
        + uy ** 2
    )

    for q in range(9):

        cu = (
            c[q, 0] * ux
            + c[q, 1] * uy
        )

        feq[q] = (
            w[q]
            * rho
            * (
                1.0
                + cu / cs2
                + 0.5
                * cu ** 2
                / cs2 ** 2
                - 0.5
                * u2
                / cs2
            )
        )

    return feq


# ============================================================================
# GUO FORCE
# ============================================================================

def guo_force(
    ux: np.ndarray,
    uy: np.ndarray,
    Fx: np.ndarray,
    Fy: np.ndarray,
    c: np.ndarray,
    w: np.ndarray,
    cs2: float,
    tau: float,
) -> np.ndarray:

    Ny, Nx = ux.shape

    force = np.zeros(
        (
            9,
            Ny,
            Nx,
        )
    )

    for q in range(9):

        cx = c[q, 0]
        cy = c[q, 1]

        cu = (
            cx * ux
            + cy * uy
        )

        term_x = (
            (cx - ux) / cs2
            + cu * cx / cs2 ** 2
        )

        term_y = (
            (cy - uy) / cs2
            + cu * cy / cs2 ** 2
        )

        force[q] = (
            w[q]
            * (
                term_x * Fx
                + term_y * Fy
            )
        )

    return (
        1.0
        - 0.5 / tau
    ) * force


# ============================================================================
# IBM VELOCITY INTERPOLATION
# ============================================================================

def interpolate_velocity(
    ux: np.ndarray,
    uy: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Nx: int,
    Ny: int,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:

    Nflag = len(X)

    Ubx = np.zeros(Nflag)
    Uby = np.zeros(Nflag)

    for j in range(Nflag):

        xj = X[j] / dx
        yj = Y[j] / dx

        ix_min = max(
            0,
            int(np.floor(xj)) - 2,
        )

        ix_max = min(
            Nx - 1,
            int(np.floor(xj)) + 2,
        )

        iy_min = max(
            0,
            int(np.floor(yj)) - 2,
        )

        iy_max = min(
            Ny - 1,
            int(np.floor(yj)) + 2,
        )

        for ix in range(
            ix_min,
            ix_max + 1,
        ):

            delta_x = delta_kernel(
                np.array(
                    [ix - xj]
                )
            )[0]

            if delta_x == 0.0:
                continue

            for iy in range(
                iy_min,
                iy_max + 1,
            ):

                delta_y = delta_kernel(
                    np.array(
                        [iy - yj]
                    )
                )[0]

                if delta_y == 0.0:
                    continue

                delta = (
                    delta_x
                    * delta_y
                )

                Ubx[j] += (
                    ux[iy, ix]
                    * delta
                    * dx ** 2
                )

                Uby[j] += (
                    uy[iy, ix]
                    * delta
                    * dx ** 2
                )

    return Ubx, Uby


# ============================================================================
# IBM FORCE SPREADING
# ============================================================================

def spread_force(
    fx_lag: np.ndarray,
    fy_lag: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Nx: int,
    Ny: int,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:

    Fx = np.zeros(
        (
            Ny,
            Nx,
        )
    )

    Fy = np.zeros(
        (
            Ny,
            Nx,
        )
    )

    Nflag = len(X)

    for j in range(Nflag):

        xj = X[j] / dx
        yj = Y[j] / dx

        ix_min = max(
            0,
            int(np.floor(xj)) - 2,
        )

        ix_max = min(
            Nx - 1,
            int(np.floor(xj)) + 2,
        )

        iy_min = max(
            0,
            int(np.floor(yj)) - 2,
        )

        iy_max = min(
            Ny - 1,
            int(np.floor(yj)) + 2,
        )

        for ix in range(
            ix_min,
            ix_max + 1,
        ):

            delta_x = delta_kernel(
                np.array(
                    [ix - xj]
                )
            )[0]

            if delta_x == 0.0:
                continue

            for iy in range(
                iy_min,
                iy_max + 1,
            ):

                delta_y = delta_kernel(
                    np.array(
                        [iy - yj]
                    )
                )[0]

                if delta_y == 0.0:
                    continue

                delta = (
                    delta_x
                    * delta_y
                )

                Fx[iy, ix] += (
                    fx_lag[j]
                    * delta
                )

                Fy[iy, ix] += (
                    fy_lag[j]
                    * delta
                )

    return Fx, Fy


# ============================================================================
# FLAGELLUM KINEMATICS
# ============================================================================

def flagellum_state(
    t: float,
    F_flag: np.ndarray,
    xf0: np.ndarray,
    h0: float,
    omega: float,
    y_center: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    temporal = np.exp(
        1j * omega * t
    )

    h = np.real(
        h0
        * F_flag
        * temporal
    )

    vy = np.real(
        h0
        * 1j
        * omega
        * F_flag
        * temporal
    )

    x = xf0.copy()

    y = (
        y_center
        + h
    )

    vx = np.zeros_like(x)

    return x, y, vx, vy


# ============================================================================
# VORTICITY
# ============================================================================

def calculate_vorticity(
    ux: np.ndarray,
    uy: np.ndarray,
) -> np.ndarray:

    return (
        np.gradient(
            uy,
            axis=1,
        )
        -
        np.gradient(
            ux,
            axis=0,
        )
    )


# ============================================================================
# CSV HELPERS
# ============================================================================

def write_csv(
    filename: Path,
    header: list[str],
    rows,
) -> None:

    with filename.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(header)

        writer.writerows(rows)


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    args = parse_arguments()

    config_file = args.config

    if not config_file.is_absolute():
        config_file = (
            Path.cwd()
            / config_file
        )

    config, config_dir = (
        load_configuration(
            config_file
        )
    )

    M1 = config["model_I"]
    M2 = config["model_II"]

    # ------------------------------------------------------------------------
    # Model I
    # ------------------------------------------------------------------------

    Lambda = float(
        M1["Lambda"]
    )

    BVP = M1["bvp"]

    bvp_n_mesh = int(
        BVP["n_mesh"]
    )

    bvp_tol = float(
        BVP["tol"]
    )

    bvp_max_nodes = int(
        BVP["max_nodes"]
    )

    # ------------------------------------------------------------------------
    # Model II
    # ------------------------------------------------------------------------

    GRID = M2["grid"]
    FLUID = M2["fluid"]
    FLAG = M2["flagellum"]

    DIAG = M2.get(
        "diagnostics",
        {}
    )

    OUTPUT = M2.get(
        "output",
        {}
    )

    # Grid
    Nx = int(GRID["Nx"])
    Ny = int(GRID["Ny"])
    Nt = int(GRID["Nt"])

    dx = float(
        GRID.get(
            "dx",
            1.0,
        )
    )

    dt = float(
        GRID.get(
            "dt",
            1.0,
        )
    )

    # Fluid
    rho0 = float(
        FLUID["rho0"]
    )

    nu = float(
        FLUID["nu"]
    )

    cs2 = float(
        FLUID["cs2"]
    )

    tau = (
        0.5
        + nu / cs2
    )

    # Flagellum
    x0 = float(
        FLAG["x0"]
    )

    Lf = float(
        FLAG["Lf"]
    )

    Nflag = int(
        FLAG["Nflag"]
    )

    h0 = float(
        FLAG["h0"]
    )

    period = float(
        FLAG["period"]
    )

    omega = (
        2.0
        * np.pi
        / period
    )

    y_center = float(
        FLAG.get(
            "y_center",
            Ny / 2.0,
        )
    )

    # Diagnostics
    diagnostic_interval = int(
        DIAG.get(
            "interval",
            100,
        )
    )

    snapshot_interval = int(
        DIAG.get(
            "snapshot_interval",
            400,
        )
    )

    # Output
    results_dir = Path(
        OUTPUT.get(
            "directory",
            "results_model_II_machin_lbm",
        )
    )

    if not results_dir.is_absolute():
        results_dir = (
            config_dir
            / results_dir
        )

    snapshot_dir = Path(
        OUTPUT.get(
            "snapshot_directory",
            "snapshots",
        )
    )

    if not snapshot_dir.is_absolute():
        snapshot_dir = (
            results_dir
            / snapshot_dir
        )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # D2Q9
    # ------------------------------------------------------------------------

    c = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
            [1, 1],
            [-1, 1],
            [-1, -1],
            [1, -1],
        ],
        dtype=int,
    )

    w = np.array(
        [
            4.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
        ]
    )

    # ------------------------------------------------------------------------
    # Console header
    # ------------------------------------------------------------------------

    print("=" * 72)
    print(
        "MODEL II — LBM PRESCRIBED KINEMATICS"
    )
    print("=" * 72)

    print()
    print("Configuration")
    print(
        f"    file       = {config_file.resolve()}"
    )

    print()
    print("Model I")
    print(
        f"    Lambda     = {Lambda:.8f}"
    )
    print(
        f"    BVP mesh   = {bvp_n_mesh}"
    )
    print(
        f"    BVP tol    = {bvp_tol:.3e}"
    )

    print()
    print("Grid")
    print(
        f"    Nx         = {Nx}"
    )
    print(
        f"    Ny         = {Ny}"
    )
    print(
        f"    Nt         = {Nt}"
    )
    print(
        f"    dx         = {dx:.6f}"
    )
    print(
        f"    dt         = {dt:.6f}"
    )

    print()
    print("Fluid")
    print(
        f"    rho0       = {rho0:.8f}"
    )
    print(
        f"    nu         = {nu:.8f}"
    )
    print(
        f"    cs2        = {cs2:.8f}"
    )
    print(
        f"    tau        = {tau:.8f}"
    )

    print()
    print("Flagellum")
    print(
        f"    x0         = {x0:.8f}"
    )
    print(
        f"    Lf         = {Lf:.8f}"
    )
    print(
        f"    Nflag      = {Nflag}"
    )
    print(
        f"    h0         = {h0:.8f}"
    )
    print(
        f"    period     = {period:.8f}"
    )
    print(
        f"    omega      = {omega:.8e}"
    )
    print(
        f"    y_center   = {y_center:.8f}"
    )

    # ------------------------------------------------------------------------
    # Solve Machin
    # ------------------------------------------------------------------------

    print()
    print("Solving Machin BVP...")

    machin = solve_machin(
        Lambda=Lambda,
        n_mesh=bvp_n_mesh,
        tol=bvp_tol,
        max_nodes=bvp_max_nodes,
    )

    print(
        f"Machin BVP converged with "
        f"{machin.x.size} nodes."
    )

    # ------------------------------------------------------------------------
    # Interpolate Machin solution to IBM markers
    # ------------------------------------------------------------------------

    chi_flag = np.linspace(
        0.0,
        Lambda,
        Nflag,
    )

    machin_values = machin.sol(
        chi_flag
    )

    F_flag = (
        machin_values[0]
        + 1j * machin_values[4]
    )

    amplitude = np.abs(
        F_flag
    )

    phase = np.unwrap(
        np.angle(F_flag)
    )

    R_A = (
        amplitude[-1]
        / amplitude[0]
    )

    attenuation = (
        100.0
        * (1.0 - R_A)
    )

    ds = (
        Lf
        / (Nflag - 1)
    )

    xf0 = (
        x0
        + Lf
        * chi_flag
        / Lambda
    )

    # ------------------------------------------------------------------------
    # Save Machin reference
    # ------------------------------------------------------------------------

    machin_file = (
        results_dir
        / "machin_reference.csv"
    )

    machin_rows = []

    for j in range(Nflag):

        machin_rows.append(
            [
                j,
                chi_flag[j],
                xf0[j],
                F_flag[j].real,
                F_flag[j].imag,
                amplitude[j],
                phase[j],
            ]
        )

    write_csv(
        machin_file,
        [
            "marker",
            "chi",
            "x0",
            "F_real",
            "F_imag",
            "amplitude",
            "phase",
        ],
        machin_rows,
    )

    # ------------------------------------------------------------------------
    # Initial LBM fields
    # ------------------------------------------------------------------------

    rho = (
        rho0
        * np.ones(
            (
                Ny,
                Nx,
            )
        )
    )

    ux = np.zeros(
        (
            Ny,
            Nx,
        )
    )

    uy = np.zeros(
        (
            Ny,
            Nx,
        )
    )

    f = equilibrium(
        rho,
        ux,
        uy,
        c,
        w,
        cs2,
    )

    # ------------------------------------------------------------------------
    # Histories
    # ------------------------------------------------------------------------

    lbm_history = []

    tip_history = []

    kinematics_rows = []

    # ------------------------------------------------------------------------
    # Final-state placeholders
    # ------------------------------------------------------------------------

    last_rho = rho.copy()
    last_ux = ux.copy()
    last_uy = uy.copy()

    last_vorticity = np.zeros(
        (
            Ny,
            Nx,
        )
    )

    last_X = None
    last_Y = None
    last_Fx = np.zeros(
        (
            Ny,
            Nx,
        )
    )
    last_Fy = np.zeros(
        (
            Ny,
            Nx,
        )
    )

    # ------------------------------------------------------------------------
    # LBM LOOP
    # ------------------------------------------------------------------------

    print()
    print("Starting LBM simulation...")
    print()

    for it in range(Nt):

        time = it * dt

        # --------------------------------------------------------------------
        # Macroscopic variables
        # --------------------------------------------------------------------

        rho = np.sum(
            f,
            axis=0,
        )

        rho = np.maximum(
            rho,
            1.0e-14,
        )

        ux = (
            np.sum(
                f
                * c[:, 0, None, None],
                axis=0,
            )
            / rho
        )

        uy = (
            np.sum(
                f
                * c[:, 1, None, None],
                axis=0,
            )
            / rho
        )

        # --------------------------------------------------------------------
        # Prescribed flagellum
        # --------------------------------------------------------------------

        X, Y, vx_b, vy_b = (
            flagellum_state(
                time,
                F_flag,
                xf0,
                h0,
                omega,
                y_center,
            )
        )

        # --------------------------------------------------------------------
        # Store raw flagellum kinematics
        #
        # We save every time step. This is intentionally independent of
        # diagnostic_interval because the third script should have the
        # complete prescribed trajectory.
        # --------------------------------------------------------------------

        phase_time = (
            omega * time
        )

        current_period = (
            time / period
        )

        for j in range(Nflag):

            h_j = (
                Y[j]
                - y_center
            )

            kinematics_rows.append(
                [
                    it,
                    time,
                    current_period,
                    phase_time,
                    j,
                    chi_flag[j],
                    X[j],
                    Y[j],
                    h_j,
                    vx_b[j],
                    vy_b[j],
                ]
            )

        # --------------------------------------------------------------------
        # Interpolate velocity to IBM markers
        # --------------------------------------------------------------------

        Ubx, Uby = (
            interpolate_velocity(
                ux,
                uy,
                X,
                Y,
                Nx,
                Ny,
                dx,
            )
        )

        # --------------------------------------------------------------------
        # Direct forcing
        # --------------------------------------------------------------------

        Fx_lag = (
            rho0
            * (
                vx_b
                - Ubx
            )
            / dt
        )

        Fy_lag = (
            rho0
            * (
                vy_b
                - Uby
            )
            / dt
        )

        # --------------------------------------------------------------------
        # Spread force
        # --------------------------------------------------------------------

        Fx, Fy = spread_force(
            Fx_lag,
            Fy_lag,
            X,
            Y,
            Nx,
            Ny,
            dx,
        )

        # --------------------------------------------------------------------
        # Force-corrected velocity
        # --------------------------------------------------------------------

        ux_d = (
            ux
            + 0.5
            * Fx
            / rho
        )

        uy_d = (
            uy
            + 0.5
            * Fy
            / rho
        )

        # --------------------------------------------------------------------
        # Collision
        # --------------------------------------------------------------------

        feq = equilibrium(
            rho,
            ux_d,
            uy_d,
            c,
            w,
            cs2,
        )

        Fq = guo_force(
            ux_d,
            uy_d,
            Fx,
            Fy,
            c,
            w,
            cs2,
            tau,
        )

        f = (
            f
            - (
                f - feq
            )
            / tau
            + Fq
        )

        # --------------------------------------------------------------------
        # Streaming
        # --------------------------------------------------------------------

        for q in range(9):

            f[q] = np.roll(
                f[q],
                c[q, 0],
                axis=1,
            )

            f[q] = np.roll(
                f[q],
                c[q, 1],
                axis=0,
            )

        # --------------------------------------------------------------------
        # Horizontal bounce-back
        # --------------------------------------------------------------------

        f[2, 0, :] = f[4, 0, :]
        f[5, 0, :] = f[7, 0, :]
        f[6, 0, :] = f[8, 0, :]

        f[4, -1, :] = f[2, -1, :]
        f[7, -1, :] = f[5, -1, :]
        f[8, -1, :] = f[6, -1, :]

        # --------------------------------------------------------------------
        # Open/copy x boundaries
        # --------------------------------------------------------------------

        f[:, :, 0] = f[:, :, 1]
        f[:, :, -1] = f[:, :, -2]

        # --------------------------------------------------------------------
        # Tip history
        # --------------------------------------------------------------------

        tip_history.append(
            [
                it,
                time,
                current_period,
                X[-1],
                Y[-1],
                Y[-1] - y_center,
                vx_b[-1],
                vy_b[-1],
            ]
        )

        # --------------------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------------------

        need_diagnostic = (
            it % diagnostic_interval == 0
            or it == Nt - 1
        )

        if need_diagnostic:

            rho_now = np.sum(
                f,
                axis=0,
            )

            rho_now = np.maximum(
                rho_now,
                1.0e-14,
            )

            ux_now = (
                np.sum(
                    f
                    * c[:, 0, None, None],
                    axis=0,
                )
                / rho_now
            )

            uy_now = (
                np.sum(
                    f
                    * c[:, 1, None, None],
                    axis=0,
                )
                / rho_now
            )

            vorticity = (
                np.gradient(
                    uy_now,
                    axis=1,
                )
                -
                np.gradient(
                    ux_now,
                    axis=0,
                )
            )

            rho_min = np.min(
                rho_now
            )

            rho_max = np.max(
                rho_now
            )

            ux_max = np.max(
                np.abs(ux_now)
            )

            uy_max = np.max(
                np.abs(uy_now)
            )

            u_max = max(
                ux_max,
                uy_max,
            )

            F_max = max(
                np.max(
                    np.abs(Fx)
                ),
                np.max(
                    np.abs(Fy)
                ),
            )

            Fy_max = np.max(
                np.abs(Fy)
            )

            lbm_history.append(
                [
                    it,
                    time,
                    current_period,
                    rho_min,
                    rho_max,
                    ux_max,
                    uy_max,
                    u_max,
                    F_max,
                    Fy_max,
                ]
            )

            last_rho = rho_now.copy()
            last_ux = ux_now.copy()
            last_uy = uy_now.copy()

            last_vorticity = (
                vorticity.copy()
            )

            last_X = X.copy()
            last_Y = Y.copy()

            last_Fx = Fx.copy()
            last_Fy = Fy.copy()

            print(
                f"step {it:6d} | "
                f"rho=[{rho_min:.9f}, "
                f"{rho_max:.9f}] | "
                f"|u|max={u_max:.4e} | "
                f"|F|max={F_max:.4e}"
            )

        # --------------------------------------------------------------------
        # Snapshots
        # --------------------------------------------------------------------
        if (
            it % snapshot_interval == 0
            or it == Nt - 1
        ):

            # ---------------------------------------------------------------
            # Snapshot somente se houver diagnóstico disponível
            # ---------------------------------------------------------------

            if last_rho is None:
                continue

            # ---------------------------------------------------------------
            # Coordenadas Eulerianas
            #
            # Os campos LBM possuem shape:
            #     (Ny, Nx)
            #
            # Portanto X e Y também devem possuir:
            #     (Ny, Nx)
            # ---------------------------------------------------------------

            x_grid = np.arange(
                Nx,
                dtype=float,
            ) * dx

            y_grid = np.arange(
                Ny,
                dtype=float,
            ) * dx

            X_grid, Y_grid = np.meshgrid(
                x_grid,
                y_grid,
            )

            # ---------------------------------------------------------------
            # Flatten
            # ---------------------------------------------------------------

            X_flat = X_grid.ravel()
            Y_flat = Y_grid.ravel()

            rho_flat = (
                np.asarray(last_rho)
                .ravel()
            )

            ux_flat = (
                np.asarray(last_ux)
                .ravel()
            )

            uy_flat = (
                np.asarray(last_uy)
                .ravel()
            )

            vorticity_flat = (
                np.asarray(last_vorticity)
                .ravel()
            )

            Fx_flat = (
                np.asarray(last_Fx)
                .ravel()
            )

            Fy_flat = (
                np.asarray(last_Fy)
                .ravel()
            )

            # ---------------------------------------------------------------
            # Sanity check
            # ---------------------------------------------------------------

            n_points = Nx * Ny

            assert X_flat.size == n_points
            assert Y_flat.size == n_points
            assert rho_flat.size == n_points
            assert ux_flat.size == n_points
            assert uy_flat.size == n_points
            assert vorticity_flat.size == n_points
            assert Fx_flat.size == n_points
            assert Fy_flat.size == n_points

            # ---------------------------------------------------------------
            # CSV
            # ---------------------------------------------------------------

            snapshot_df = pd.DataFrame(
                {
                    "step": np.full(
                        n_points,
                        it,
                        dtype=np.int32,
                    ),

                    "time": np.full(
                        n_points,
                        time,
                        dtype=float,
                    ),

                    "X": X_flat,
                    "Y": Y_flat,

                    "rho": rho_flat,

                    "ux": ux_flat,
                    "uy": uy_flat,

                    "vorticity": vorticity_flat,

                    "Fx": Fx_flat,
                    "Fy": Fy_flat,
                }
            )

            snapshot_file = (
                snapshot_dir
                / f"snapshot_{it:06d}.csv"
            )

            snapshot_df.to_csv(
                snapshot_file,
                index=False,
            )

            print(
                f"  snapshot -> "
                f"{snapshot_file}"
            )

    # =========================================================================
    # FINAL FIELD
    # =========================================================================

    rho_final = np.sum(
        f,
        axis=0,
    )

    rho_final = np.maximum(
        rho_final,
        1.0e-14,
    )

    ux_final = (
        np.sum(
            f
            * c[:, 0, None, None],
            axis=0,
        )
        / rho_final
    )

    uy_final = (
        np.sum(
            f
            * c[:, 1, None, None],
            axis=0,
        )
        / rho_final
    )

    vorticity_final = (
        np.gradient(
            uy_final,
            axis=1,
        )
        -
        np.gradient(
            ux_final,
            axis=0,
        )
    )

    # ------------------------------------------------------------------------
    # Final field CSV
    # ------------------------------------------------------------------------

    final_field_file = (
        results_dir
        / "final_field.csv"
    )

    final_field_rows = []

    for iy in range(Ny):

        for ix in range(Nx):

            final_field_rows.append(
                [
                    ix,
                    iy,
                    ix * dx,
                    iy * dx,
                    rho_final[iy, ix],
                    ux_final[iy, ix],
                    uy_final[iy, ix],
                    vorticity_final[iy, ix],
                ]
            )

    write_csv(
        final_field_file,
        [
            "ix",
            "iy",
            "x",
            "y",
            "rho",
            "ux",
            "uy",
            "vorticity",
        ],
        final_field_rows,
    )

    # ------------------------------------------------------------------------
    # Flagellum kinematics CSV
    # ------------------------------------------------------------------------

    kinematics_file = (
        results_dir
        / "flagellum_kinematics.csv"
    )

    write_csv(
        kinematics_file,
        [
            "step",
            "time",
            "period",
            "phase",
            "marker",
            "chi",
            "x",
            "y",
            "h",
            "vx",
            "vy",
        ],
        kinematics_rows,
    )

    # ------------------------------------------------------------------------
    # Tip history CSV
    # ------------------------------------------------------------------------

    tip_file = (
        results_dir
        / "tip_history.csv"
    )

    write_csv(
        tip_file,
        [
            "step",
            "time",
            "period",
            "x_tip",
            "y_tip",
            "h_tip",
            "vx_tip",
            "vy_tip",
        ],
        tip_history,
    )

    # ------------------------------------------------------------------------
    # LBM history CSV
    # ------------------------------------------------------------------------

    history_file = (
        results_dir
        / "lbm_history.csv"
    )

    write_csv(
        history_file,
        [
            "step",
            "time",
            "period",
            "rho_min",
            "rho_max",
            "ux_max",
            "uy_max",
            "u_max",
            "F_max",
            "Fy_max",
        ],
        lbm_history,
    )

    # =========================================================================
    # PARAMETERS CSV
    # =========================================================================

    parameters_file = (
        results_dir
        / "parameters.csv"
    )

    parameters = [
        ["Lambda", Lambda],
        ["BVP_n_mesh", bvp_n_mesh],
        ["BVP_tol", bvp_tol],
        ["BVP_max_nodes", bvp_max_nodes],

        ["Nx", Nx],
        ["Ny", Ny],
        ["Nt", Nt],
        ["dx", dx],
        ["dt", dt],

        ["rho0", rho0],
        ["nu", nu],
        ["cs2", cs2],
        ["tau", tau],

        ["x0", x0],
        ["Lf", Lf],
        ["Nflag", Nflag],
        ["h0", h0],
        ["period", period],
        ["omega", omega],
        ["y_center", y_center],
        ["ds", ds],

        ["diagnostic_interval", diagnostic_interval],
        ["snapshot_interval", snapshot_interval],

        ["Machin_R_A", R_A],
        ["Machin_attenuation_percent", attenuation],

        ["simulated_periods", Nt * dt / period],
    ]

    write_csv(
        parameters_file,
        [
            "parameter",
            "value",
        ],
        parameters,
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    rho_min_final = np.min(
        rho_final
    )

    rho_max_final = np.max(
        rho_final
    )

    ux_max_final = np.max(
        np.abs(ux_final)
    )

    uy_max_final = np.max(
        np.abs(uy_final)
    )

    force_max_final = max(
        np.max(
            np.abs(last_Fx)
        ),
        np.max(
            np.abs(last_Fy)
        ),
    )

    force_y_max_final = np.max(
        np.abs(last_Fy)
    )

    print()
    print("=" * 72)
    print("MODEL II — FINAL SUMMARY")
    print("=" * 72)

    print()
    print("Machin reference")
    print(
        f"    Lambda          = {Lambda:.8f}"
    )
    print(
        f"    R_A             = {R_A:.10f}"
    )
    print(
        f"    attenuation     = {attenuation:.6f}%"
    )

    print()
    print("LBM")
    print(
        f"    Nx x Ny         = {Nx} x {Ny}"
    )
    print(
        f"    Nt              = {Nt}"
    )
    print(
        f"    period          = {period:.6f}"
    )
    print(
        f"    periods         = "
        f"{Nt * dt / period:.6f}"
    )
    print(
        f"    tau             = {tau:.8f}"
    )

    print()
    print("Final density")
    print(
        f"    rho_min         = "
        f"{rho_min_final:.12f}"
    )
    print(
        f"    rho_max         = "
        f"{rho_max_final:.12f}"
    )

    print()
    print("Final velocity")
    print(
        f"    max |ux|        = "
        f"{ux_max_final:.8e}"
    )
    print(
        f"    max |uy|        = "
        f"{uy_max_final:.8e}"
    )

    print()
    print("Immersed-boundary force")
    print(
        f"    max |F|         = "
        f"{force_max_final:.8e}"
    )
    print(
        f"    max |Fy|        = "
        f"{force_y_max_final:.8e}"
    )

    print()
    print("Raw data")
    print(
        f"    {machin_file}"
    )
    print(
        f"    {kinematics_file}"
    )
    print(
        f"    {tip_file}"
    )
    print(
        f"    {history_file}"
    )
    print(
        f"    {final_field_file}"
    )
    print(
        f"    {parameters_file}"
    )

    print()
    print("Snapshots")
    print(
        f"    {snapshot_dir}"
    )

    print()
    print("=" * 72)
    print("MODEL II completed successfully.")
    print("=" * 72)

    return 0


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )
