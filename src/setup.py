import NavierStokesFunctions as nsf
import tests as tests
import Introduction as intro
import IncompressibleFlow as incompFlow
import RayleighBenardConvection as rb

import numpy as np

def main():
    '''
    Entry point: set IC_MODE here to choose the initial velocity field.

    Available modes
    ---------------
    "mode1"  — v =  sin(ky y) cos(kz z),   w = -(ky/kz) cos(ky y) sin(kz z)
    "mode2"  — v =  sin(2ky y) cos(kz z),  w = -(2ky/kz) cos(2ky y) sin(kz z)
    "mode3"  — v =  sin(ky y) cos(2kz z),  w = -(ky/2kz) cos(ky y) sin(2kz z)
    "double" — superposition of mode1 + mode2
    "custom" — supply your own functions via IC_CUSTOM_V / IC_CUSTOM_W below.

    Custom mode example
    -------------------
    IC_MODE = "custom"
    IC_CUSTOM_V = lambda Y, Z, ky, kz: np.sin(3*ky*Y) * np.cos(kz*Z)
    IC_CUSTOM_W = lambda Y, Z, ky, kz: -(3*ky/kz) * np.cos(3*ky*Y) * np.sin(kz*Z)
    '''

    # ── USER PARAMETER ─────────────────────────────────────────────────────────
    IC_MODE     = "mode1"   # <-- change to "mode2", "mode3", "double", or "custom"
    IC_CUSTOM_V = None      # set to a callable when IC_MODE = "custom"
    IC_CUSTOM_W = None      # set to a callable when IC_MODE = "custom"
    # ───────────────────────────────────────────────────────────────────────────

    print("=" * 55)
    print("NAVIER-STOKES!!!")
    print("=" * 55)
    print()

    y, z, Y, Z = nsf.create_grid()
    v, w, p, label = nsf.initial_conditions(Y, Z, IC_MODE, IC_CUSTOM_V, IC_CUSTOM_W)

    print(f"IC mode             : {IC_MODE}")
    print(f"velocity            : {label}")
    print(f"Grid                : dy={nsf.dy:.5f}  dz={nsf.dz:.5f}")
    print(f"Initial Conditions  : max|∇·u| = {np.max(np.abs(nsf.divergence(v, w))):.3e}\n\n")

    print("=" * 55)
    print("Unit Tests!!!")
    print("=" * 55)
    print()
    
    tests.intro_tests(IC_MODE, IC_CUSTOM_V, IC_CUSTOM_W)
    tests.incomp_flow_tests(IC_MODE, IC_CUSTOM_V, IC_CUSTOM_W, custom_components=[(3, 1)])
    tests.rayleigh_benard_tests(IC_MODE, IC_CUSTOM_V, IC_CUSTOM_W)

    intro.main(v, w, Y, Z, IC_MODE, IC_CUSTOM_V, IC_CUSTOM_W)
    incompFlow.main(v, w, p, y, z, IC_MODE, IC_CUSTOM_V, IC_CUSTOM_W)
    rb.main(v, w, p, y, z, Y, Z, IC_MODE, IC_CUSTOM_V, IC_CUSTOM_W)

if __name__ == "__main__":
    main()