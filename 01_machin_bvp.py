#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MODEL I — MACHIN FINITE-FLAGELLUM BVP

Solves the finite-length Machin boundary-value problem

    F'''' + i F = 0

with boundary conditions

    F(0)      = 1
    F''(0)    = 0
    F''(Λ)    = 0
    F'''(Λ)   = 0

The solution is written to CSV files. No figures are generated here.

Usage
-----
    python3 01_machin_bvp.py exp_1.json

Input JSON
----------
{
    "model_I": {
        "Lambda": 5.0,
        "bvp": {
            "max_nodes": 50000,
            "n_mesh": 200,
            "tol": 1e-8
        },
        "output": {
            "directory": "results_machin_reference"
        }
    }
}

Outputs
-------
results_machin_reference/
    machin_solution.csv
    machin_summary.csv
    machin_parameters.csv

machin_solution.csv
    Raw spatial solution used by the plotting script.

machin_summary.csv
    Scalar quantities extracted from the solution.

machin_parameters.csv
    Numerical/configuration parameters used in the simulation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_bvp


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_N_EVAL = 5000


# ============================================================================
# COMMAND LINE
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve the finite-length Machin BVP and export raw CSV data."
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to the experiment JSON configuration file."
    )

    return parser.parse_args()


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_configuration(config_file: Path) -> tuple[dict, Path]:
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_file}"
        )

    with config_file.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if "model_I" not in config:
        raise KeyError(
            "Configuration must contain the section 'model_I'."
        )

    model = config["model_I"]

    required = ["Lambda", "bvp", "output"]

    for key in required:
        if key not in model:
            raise KeyError(
                f"Missing key 'model_I.{key}' in configuration."
            )

    for key in ["n_mesh", "tol", "max_nodes"]:
        if key not in model["bvp"]:
            raise KeyError(
                f"Missing key 'model_I.bvp.{key}' in configuration."
            )

    if "directory" not in model["output"]:
        raise KeyError(
            "Missing key 'model_I.output.directory' in configuration."
        )

    config_dir = config_file.resolve().parent

    return config, config_dir


# ============================================================================
# MACHIN ODE
# ============================================================================

def machin_ode(
    chi: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Real-valued representation of

        F'''' + i F = 0

    Let

        F = a + i b

    and define

        y = [Re(F), Re(F'), Re(F''), Re(F'''),
             Im(F), Im(F'), Im(F''), Im(F''')].

    Since

        F'''' = -i F,

    we have

        Re(F'''') = Im(F)
        Im(F'''') = -Re(F).
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


# ============================================================================
# BOUNDARY CONDITIONS
# ============================================================================

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

    return np.array([
        # F(0) = 1
        ya[0] - 1.0,
        ya[4],

        # F''(0) = 0
        ya[2],
        ya[6],

        # F''(Λ) = 0
        yb[2],
        yb[6],

        # F'''(Λ) = 0
        yb[3],
        yb[7],
    ])


# ============================================================================
# INITIAL GUESS
# ============================================================================

def initial_guess(chi: np.ndarray) -> np.ndarray:
    """
    Initial guess based on the decaying semi-infinite Machin mode.

    This is used only to provide a physically appropriate starting point
    for solve_bvp. The final solution is determined by the finite-length
    boundary conditions.
    """

    # Roots of r^4 = -i.
    #
    # A decaying mode can be written using
    #
    #   r = exp(-i*pi/8)
    #
    # whose real part is positive. We use the conjugate-compatible
    # real representation through exp(-alpha*x) * oscillatory terms.

    alpha = 1.0 / np.sqrt(2.0)

    decay = np.exp(-alpha * chi)

    # A smooth decaying initial approximation.
    real = decay * np.cos(alpha * chi)
    imag = -decay * np.sin(alpha * chi)

    # Normalize F(0) approximately to 1.
    real /= real[0]
    imag /= real[0] if real[0] != 0 else 1.0

    # Numerical derivatives using repeated gradients.
    d1_real = np.gradient(real, chi)
    d2_real = np.gradient(d1_real, chi)
    d3_real = np.gradient(d2_real, chi)

    d1_imag = np.gradient(imag, chi)
    d2_imag = np.gradient(d1_imag, chi)
    d3_imag = np.gradient(d2_imag, chi)

    y = np.vstack([
        real,
        d1_real,
        d2_real,
        d3_real,
        imag,
        d1_imag,
        d2_imag,
        d3_imag,
    ])

    return y


# ============================================================================
# SOLVER
# ============================================================================

def solve_machin(
    Lambda: float,
    n_mesh: int,
    tol: float,
    max_nodes: int,
):
    if Lambda <= 0:
        raise ValueError("Lambda must be positive.")

    if n_mesh < 10:
        raise ValueError("n_mesh must be at least 10.")

    if tol <= 0:
        raise ValueError("BVP tolerance must be positive.")

    if max_nodes <= n_mesh:
        raise ValueError(
            "max_nodes must be larger than n_mesh."
        )

    chi_mesh = np.linspace(0.0, Lambda, n_mesh)

    y_guess = initial_guess(chi_mesh)

    solution = solve_bvp(
        machin_ode,
        machin_bc,
        chi_mesh,
        y_guess,
        tol=tol,
        max_nodes=max_nodes,
        verbose=0,
    )

    if not solution.success:
        raise RuntimeError(
            "Machin BVP did not converge.\n"
            f"Message: {solution.message}\n"
            f"Status: {solution.status}\n"
            f"Nodes: {solution.x.size}"
        )

    return solution


# ============================================================================
# RAW DATA
# ============================================================================

def evaluate_solution(
    solution,
    Lambda: float,
    n_eval: int = DEFAULT_N_EVAL,
) -> dict[str, np.ndarray]:

    chi = np.linspace(
        0.0,
        Lambda,
        n_eval,
    )

    y = solution.sol(chi)

    F_real = y[0]
    F_imag = y[4]

    F = F_real + 1j * F_imag

    amplitude = np.abs(F)

    # Unwrapped phase is useful for plotting the spatial phase evolution.
    phase = np.unwrap(np.angle(F))

    # Derivatives of F.
    F1 = y[1] + 1j * y[5]
    F2 = y[2] + 1j * y[6]
    F3 = y[3] + 1j * y[7]

    return {
        "chi": chi,
        "F_real": F_real,
        "F_imag": F_imag,
        "F_abs": amplitude,
        "F_phase": phase,
        "F1_real": np.real(F1),
        "F1_imag": np.imag(F1),
        "F2_real": np.real(F2),
        "F2_imag": np.imag(F2),
        "F3_real": np.real(F3),
        "F3_imag": np.imag(F3),
    }


# ============================================================================
# SUMMARY
# ============================================================================

def calculate_summary(
    solution,
    Lambda: float,
) -> dict[str, float]:

    F_prox = complex(
        solution.sol(0.0)[0],
        solution.sol(0.0)[4],
    )

    F_dist = complex(
        solution.sol(Lambda)[0],
        solution.sol(Lambda)[4],
    )

    amplitude_proximal = abs(F_prox)
    amplitude_distal = abs(F_dist)

    R_A = amplitude_distal / amplitude_proximal

    attenuation_percent = (
        100.0 * (1.0 - R_A)
    )

    phase_proximal = np.angle(F_prox)
    phase_distal = np.angle(F_dist)

    phase_difference = np.angle(
        F_dist / F_prox
    )

    return {
        "Lambda": Lambda,
        "amplitude_proximal": amplitude_proximal,
        "amplitude_distal": amplitude_distal,
        "R_A": R_A,
        "attenuation_percent": attenuation_percent,
        "phase_proximal": phase_proximal,
        "phase_distal": phase_distal,
        "phase_difference": phase_difference,
        "solver_nodes": float(solution.x.size),
        "solver_iterations": float(solution.niter),
    }


# ============================================================================
# CSV WRITERS
# ============================================================================

def write_solution_csv(
    filename: Path,
    data: dict[str, np.ndarray],
) -> None:

    fieldnames = [
        "chi",
        "F_real",
        "F_imag",
        "F_abs",
        "F_phase",
        "F1_real",
        "F1_imag",
        "F2_real",
        "F2_imag",
        "F3_real",
        "F3_imag",
    ]

    with filename.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(fieldnames)

        n = len(data["chi"])

        for i in range(n):
            writer.writerow([
                f"{data['chi'][i]:.16e}",
                f"{data['F_real'][i]:.16e}",
                f"{data['F_imag'][i]:.16e}",
                f"{data['F_abs'][i]:.16e}",
                f"{data['F_phase'][i]:.16e}",
                f"{data['F1_real'][i]:.16e}",
                f"{data['F1_imag'][i]:.16e}",
                f"{data['F2_real'][i]:.16e}",
                f"{data['F2_imag'][i]:.16e}",
                f"{data['F3_real'][i]:.16e}",
                f"{data['F3_imag'][i]:.16e}",
            ])


def write_summary_csv(
    filename: Path,
    summary: dict[str, float],
) -> None:

    with filename.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(summary.keys())

        writer.writerow([
            f"{value:.16e}"
            for value in summary.values()
        ])


def write_parameters_csv(
    filename: Path,
    config_file: Path,
    Lambda: float,
    n_mesh: int,
    tol: float,
    max_nodes: int,
    n_eval: int,
) -> None:

    parameters = {
        "config_file": str(config_file),
        "Lambda": Lambda,
        "n_mesh": n_mesh,
        "tol": tol,
        "max_nodes": max_nodes,
        "n_eval": n_eval,
    }

    with filename.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "parameter",
            "value",
        ])

        for key, value in parameters.items():
            writer.writerow([
                key,
                value,
            ])


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    args = parse_arguments()

    config_file = args.config.resolve()

    try:
        config, config_dir = load_configuration(
            config_file
        )

        model = config["model_I"]

        Lambda = float(
            model["Lambda"]
        )

        bvp_config = model["bvp"]

        n_mesh = int(
            bvp_config["n_mesh"]
        )

        tol = float(
            bvp_config["tol"]
        )

        max_nodes = int(
            bvp_config["max_nodes"]
        )

        output_directory = Path(
            model["output"]["directory"]
        )

        if not output_directory.is_absolute():
            output_directory = (
                config_dir / output_directory
            )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ------------------------------------------------------------
        # Solve BVP
        # ------------------------------------------------------------

        print("=" * 72)
        print("MODEL I — MACHIN FINITE-FLAGELLUM BVP")
        print("=" * 72)

        print()
        print("Configuration")
        print(f"    file       = {config_file}")
        print(f"    Lambda     = {Lambda:.8f}")
        print(f"    n_mesh     = {n_mesh}")
        print(f"    tol        = {tol:.3e}")
        print(f"    max_nodes  = {max_nodes}")
        print()

        print("Solving Machin BVP...")

        solution = solve_machin(
            Lambda=Lambda,
            n_mesh=n_mesh,
            tol=tol,
            max_nodes=max_nodes,
        )

        print(
            "Machin BVP converged with "
            f"{solution.x.size} nodes."
        )

        # ------------------------------------------------------------
        # Evaluate dense raw data
        # ------------------------------------------------------------

        data = evaluate_solution(
            solution,
            Lambda=Lambda,
            n_eval=DEFAULT_N_EVAL,
        )

        summary = calculate_summary(
            solution,
            Lambda=Lambda,
        )

        # ------------------------------------------------------------
        # Write CSV files
        # ------------------------------------------------------------

        solution_file = (
            output_directory /
            "machin_solution.csv"
        )

        summary_file = (
            output_directory /
            "machin_summary.csv"
        )

        parameters_file = (
            output_directory /
            "machin_parameters.csv"
        )

        write_solution_csv(
            solution_file,
            data,
        )

        write_summary_csv(
            summary_file,
            summary,
        )

        write_parameters_csv(
            parameters_file,
            config_file=config_file,
            Lambda=Lambda,
            n_mesh=n_mesh,
            tol=tol,
            max_nodes=max_nodes,
            n_eval=DEFAULT_N_EVAL,
        )

        # ------------------------------------------------------------
        # Report
        # ------------------------------------------------------------

        print()
        print("Machin reference")
        print(
            f"    Lambda       = "
            f"{summary['Lambda']:.8f}"
        )
        print(
            f"    A_prox       = "
            f"{summary['amplitude_proximal']:.10f}"
        )
        print(
            f"    A_dist       = "
            f"{summary['amplitude_distal']:.10f}"
        )
        print(
            f"    R_A          = "
            f"{summary['R_A']:.10f}"
        )
        print(
            f"    attenuation  = "
            f"{summary['attenuation_percent']:.6f}%"
        )
        print(
            f"    phase_prox   = "
            f"{summary['phase_proximal']:.10f}"
        )
        print(
            f"    phase_dist   = "
            f"{summary['phase_distal']:.10f}"
        )
        print(
            f"    phase_delta  = "
            f"{summary['phase_difference']:.10f}"
        )

        print()
        print("CSV output")
        print(f"    {solution_file}")
        print(f"    {summary_file}")
        print(f"    {parameters_file}")

        print()
        print("=" * 72)
        print("MODEL I completed successfully.")
        print("=" * 72)

        return 0

    except Exception as exc:

        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
