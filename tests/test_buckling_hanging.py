"""
Iterative hanging beam analysis using LTB eigenvalue solver
Stratford & Burgoyne (1999) "Lateral stability of long precast concrete beams" 

"""

from pybeamnlfea.model.frame import Frame
from pybeamnlfea.model.material import LinearElastic
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 
from pybeamnlfea.model.boundary import BoundaryCondition 

# Author: James Whiteley (github.com/jamesalexwhiteley) 

# When beam rolls by θ, cable height h provides a
# restoring moment of (wL/2) * h per support (small deflection assumption). 
# This is modelled by a rotational spring on the theta_x DOF. 
# Spring stiffness depends on w => iterate.

# Beam geometry
n = 20 
L = 40.0  # m

# SY-6 section properties
d = 2.0               # m
b = 0.75              # m 
A = 0.709             # m2 
Iy = 0.2837           # m4 
Iz = 0.014            # m4 
J = 0.0221            # m4 
Iw = 0.00075     
yb = 0.855            # m height of centroid above soffit 
h = 1.6               # m load height 

# Concrete properties 
E = 34e9    # 34 GPa 
G = 14.8e9  # 14.8 GPa 
w = 16.73e3  # kN/m
rho = (16.73 * 1e3 / 9.81) / A  # kg/m3   # rho = (w/g) / A; w = 16.73 kN/m

# Stratford Eq. 34 initial estimate (vertical cables, a=0)
wcr_estimate = 120 * E * Iz * h / L**4

# Iterative solution
wcr_prev = wcr_estimate
tolerance = 0.01  # 1% convergence
max_iter = 20

for iteration in range(max_iter):
    
    # Rotational spring stiffness at each support
    # Total restoring moment = wcr * L * h (for whole beam)
    # Split between 2 supports
    k_theta = wcr_prev * L * h / 2.0
    
    # Build fresh model each iteration
    beam = Frame() 
    beam.add_nodes([[i * L / n, 0, 0] for i in range(n + 1)]) 
    beam.add_material("concrete", LinearElastic(rho=rho, E=E, G=G))
    beam.add_section("SY6", Section(A=A, Iy=Iy, Iz=Iz, J=J, Iw=Iw, y0=(d - yb), z0=0))
    beam.add_elements([[i, i + 1] for i in range(n)], "concrete", "SY6", element_class=ThinWalledBeamElement) 
    
    # Supports: fix translations, free all rotations including twist
    beam.add_boundary_condition(0, [0, 0, 0, 0, 1, 1, 0], BoundaryCondition)
    beam.add_boundary_condition(n, [0, 0, 0, 0, 1, 1, 0], BoundaryCondition)
    
    # Rotational spring on twist DOF (dof_index=3 for θx) at each support
    # This represents the cable-height restoring effect
    beam.add_elastic_boundary_condition(0, dof_index=3, stiffness=k_theta)
    beam.add_elastic_boundary_condition(n, dof_index=3, stiffness=k_theta)
    
    # Self-weight at centroid 
    for e in range(n):
        beam.add_uniform_load(element_id=e, forces=[0, 0, -1], load_height=0)
    
    # Solve eigenvalue
    eigenvalues, eigenvectors = beam.solve_eigen(num_modes=1) 
    wcr_new = eigenvalues[0] * 1.0  # unit load × eigenvalue
    
    change = abs(wcr_new - wcr_prev) / wcr_prev * 100
    
    print(f"  iter {iteration+1:2d}: k_θ = {k_theta:.0f} Nm/rad | "
          f"wcr = {wcr_new/1e3:.2f} kN/m | change = {change:.3f}%")
    
    if change < tolerance:
        print(f"\n  Converged after {iteration+1} iterations")
        break
    
    wcr_prev = wcr_new

# Compare with analytical
wcr_stratford = 120 * E * Iz * h / L**4
print(f"\n  wcr (FEA, converged) = {wcr_new/1e3:.2f} kN/m")
print(f"  wcr (Stratford Eq.34) = {wcr_stratford/1e3:.2f} kN/m")
print(f"  Difference = {(wcr_new - wcr_stratford)/wcr_stratford * 100:.1f}%")
print(f"  (Expect FEA < analytic by 10-15% due to finite torsional stiffness)")