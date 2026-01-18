'''
3.2 Introduction.
Comp_Sci_Navier_Stokes_Equations.src.Introduction

We aim to solve the quite involved equations by splitting the terms into sub problems of easier type. These are then
approximated by finite-difference methods on a cartesian grid.
'''

'''
Explain the phrase 'operator splitting'. Why / how does it work?
Refer to Ceren's math notes -- do we need to do anything specifically in the code?
'''

'''
Implement the upwind finite difference method for the linear advection equation with variable advection field in two 
spatial dimensions using operator splitting for each of the spatial derivatives. 
(u & a below are vectors)
(δ_t * u) + a * ∇u = 0
u(0, y, z) = u(y, z)
As boundary conditions consider the ones given in the previous section. 
Check which restrictions occur due to the explicit treatment of this term. 
'''
def upward_finite_difference(a):
    return