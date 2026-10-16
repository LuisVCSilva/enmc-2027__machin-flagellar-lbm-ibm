# Machin's Flagellar Wave Attenuation via LBM--IBM

Computational framework for studying finite-length flagellar wave attenuation using Machin's elastohydrodynamic model and a Lattice Boltzmann--Immersed Boundary (LBM--IBM) formulation.

## Overview

This repository contains the computational codes developed for the study

> **Machin's Flagellar Wave Attenuation Revisited via a Lattice Boltzmann--Immersed Boundary Method**

The computational framework is organized into two main models.

### Model I -- Machin's finite-length problem

The first model solves the classical finite-length Machin boundary-value problem

\[
F^{(4)} + iF = 0,
\]

with the appropriate proximal and distal boundary conditions.

The numerical solution provides the spatially varying amplitude and phase of the flagellar deformation and serves as the reference kinematics for the subsequent fluid simulation.

### Model II -- Prescribed-kinematics LBM--IBM

The second model couples the Machin deformation to a two-dimensional D2Q9 Lattice Boltzmann solver with an Immersed Boundary Method.

The flagellar deformation is prescribed according to the Model I solution. The fluid problem therefore represents a hydrodynamic benchmark with known structural kinematics rather than a fully coupled fluid--structure interaction model.

The simulation provides:

- fluid density;
- velocity field;
- immersed-boundary force;
- vorticity;
- flagellar displacement;
- flagellar velocity;
- temporal diagnostics;
- raw simulation data.
