# Navier-Stokes Equations

A 2D finite-difference solver for the incompressible Navier-Stokes equations and Rayleigh-Bénard convection, implemented on a Cartesian grid using Chorin's projection method and operator-split upwind advection.

---

## Overview

The project is structured around three simulation modules, a shared function library, a test suite, and a single entry point (`setup.py`):

| File | Purpose |
|---|---|
| `NavierStokesFunctions.py` | Shared grid, operators, BCs, and Chorin projection steps |
| `Introduction.py` | 3.2 — 2D linear advection with variable field (upwind scheme) |
| `IncompressibleFlow.py` | 3.3 — Incompressible Navier-Stokes (β = 0) |
| `RayleighBenardConvection.py` | 3.4 — Coupled Boussinesq system with buoyancy |
| `tests.py` | Unit tests for all three simulation modules |
| `setup.py` | Entry point — configure and run everything from here |

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create the output directory

All plots and animations are saved to `./imgs/`. Create it before running:

```bash
mkdir -p imgs
```

### 3. Run the solver from the root directory

```bash
python setup.py
```

This will:
1. Build the grid and initialise the velocity field
2. Run all unit tests
3. Execute the Introduction (3.2), Incompressible Flow (3.3), and Rayleigh-Bénard (3.4) simulations in sequence
4. Save plots and a GIF animation to `./imgs/`

---

## Configuration

All user-facing parameters are set at the top of `setup.py`:

```python
IC_MODE     = "mode1"   # initial condition mode (see below)
IC_CUSTOM_V = None      # callable for custom v-field
IC_CUSTOM_W = None      # callable for custom w-field
```

### Initial Condition Modes

| `IC_MODE` | v-field | w-field |
|---|---|---|
| `"mode1"` | `sin(ky·y) cos(kz·z)` | `-(ky/kz) cos(ky·y) sin(kz·z)` |
| `"mode2"` | `sin(2ky·y) cos(kz·z)` | `-(2ky/kz) cos(2ky·y) sin(kz·z)` |
| `"mode3"` | `sin(ky·y) cos(2kz·z)` | `-(ky/2kz) cos(ky·y) sin(2kz·z)` |
| `"double"` | mode1 + mode2 superposition | mode1 + mode2 superposition |
| `"custom"` | user-supplied callable | user-supplied callable |

For `"custom"` mode, provide callables with signature `f(Y, Z, ky, kz) -> np.ndarray`:

```python
IC_MODE     = "custom"
IC_CUSTOM_V = lambda Y, Z, ky, kz: np.sin(3*ky*Y) * np.cos(kz*Z)
IC_CUSTOM_W = lambda Y, Z, ky, kz: -(3*ky/kz) * np.cos(3*ky*Y) * np.sin(kz*Z)
```

### Physical and Grid Parameters

Core parameters are set in `NavierStokesFunctions.py`:

```python
Ny, Nz = 50, 50    # grid resolution
Ly, Lz = 1.0, 1.0  # domain size
nu     = 0.01       # kinematic viscosity
rho    = 1.0        # fluid density
nt     = 100        # time steps for the incompressible flow simulation
```

The thermal expansion coefficient for Rayleigh-Bénard is set in `RayleighBenardConvection.py`:

```python
beta = 50.0
```

---

## Domain and Boundary Conditions

The domain is `[0, Ly) × [0, Lz]` (y periodic, z wall-bounded).

- **y-direction**: periodic
- **z-direction**: no-penetration (`w = 0`) and free-slip (`∂v/∂z = 0`) at both walls
- **Temperature (RBC)**: Dirichlet — `T = 1` at the hot bottom wall (`z = 0`), `T = 0` at the cold top wall (`z = Lz`)
- **Pressure**: homogeneous Neumann (`∂p/∂n = 0`) at both walls; zero-mean gauge

Time steps are chosen adaptively to satisfy both the advection CFL condition and the diffusion stability bound.

---

## Numerical Methods

**3.2 — Linear Advection**
Operator-split first-order upwind scheme. Analytical solution available for all built-in IC modes, enabling direct error comparison and convergence-order verification.

**3.3 — Incompressible Flow**
Chorin's projection method:
1. *Predictor* — explicit Euler step for advection (split upwind) + diffusion (explicit Laplacian), ignoring pressure.
2. *Poisson solve* — spectral direct solver using FFT (periodic y) and DCT-I (Neumann z) to enforce `∇·u = 0`.
3. *Projection* — velocity correction `u ← u* − (dt/ρ) ∇p`.

**3.4 — Rayleigh-Bénard Convection**
Boussinesq extension: the scalar temperature equation `∂_t T + u·∇T = ΔT` is coupled to the momentum equation via the buoyancy term `T β e_z`. Temperature uses upwind advection and an explicit Laplacian; the momentum predictor adds the buoyancy increment relative to the conductive background `T_c = 1 − z`.

---

## Unit Tests

Tests run automatically at the start of `setup.py`. They can also be run independently:

```python
import tests
tests.intro_tests()
tests.incomp_flow_tests()
tests.rayleigh_benard_tests()
```

The test suite covers:

**Introduction (upwind scheme)**
- Zero-velocity identity
- Constant-field invariance
- CFL stability boundary (stable vs. unstable step)
- First-order convergence under grid refinement

**Incompressible Flow**
- IC divergence is O(h²)
- Laplacian consistency on a manufactured solution
- Poisson solver accuracy (error < 1 × 10⁻³)
- Projection reduces divergence by ≥ 95%

**Rayleigh-Bénard**
- Temperature BCs enforced by `apply_temperature_bc`
- `Δ(1 − z) = 0` in the interior (conductive profile)
- Zero temperature fluctuation produces no buoyancy force
- One coupled RB step preserves all BCs and produces finite values

---

## Output Files

All output is written to `./imgs/`:

| File | Content |
|---|---|
| `upwind_<mode>.png` | Numerical vs. analytical velocity fields (3.2) |
| `incompressible_flow__<mode>.png` | Speed, pressure, and divergence fields (3.3) |
| `rb_temp_compare_<mode>.png` | Initial vs. final temperature (3.4) |
| `rb_state_<mode>.png` | Final RBC temperature + velocity quiver |
| `rb_diagnostics_<mode>.png` | Full T, fluctuation θ, vertical velocity w, streamlines |
| `rb_animation_<mode>.gif` | Animated evolution of the four RBC panels |