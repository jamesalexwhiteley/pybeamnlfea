from pybeamnlfea.model.frame import Frame
from pybeamnlfea.model.material import LinearElastic
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import NodalLoad 
import numpy as np 

# Author: James Whiteley (github.com/jamesalexwhiteley) 

# ======== Torsional displacement ======== #
# Column 
n = 10 
L = 3  # short column 
beam = Frame() 
beam.add_nodes([[i*L/n, 0, 0] for i in range(n+1)])

# Material: steel
E = 210e9   # N/m2
G = 80e9    # N/m2 
rho = 7850  # kg/m3

# UB127x76x13 section 
A = 1650e-6    # m2
Iy = 0.746e-6  # m4
Iz = 0.147e-6  # m4
J = 0.0285e-6  # m4
Iw = 0.002e-12 # m6

beam.add_material("steel", LinearElastic(rho=rho, E=E, G=G))
beam.add_section("UB127x76x13", Section(A=A, Iy=Iy, Iz=Iz, J=J, Iw=Iw, y0=0, z0=0))
beam.add_elements([[i, i+1] for i in range(n)], "steel", "UB127x76x13", element_class=ThinWalledBeamElement)

beam.add_boundary_condition(0, [0, 0, 0, 0, 0, 0, 0], BoundaryCondition) # Fixed 
beam.add_boundary_condition(n, [1, 1, 1, 1, 1, 1, 1], BoundaryCondition) # Free 

# Apply torque
T = 1
beam.add_nodal_load(n, [0, 0, 0, T, 0, 0, 0], NodalLoad)

# # Solve model 
# results = beam.solve() 
# beam.show_deformed_shape(scale=5e2, show_undeformed=True, show_local_axes=False)

# torsional_disp = results.get_nodal_displacements(node_id=n, dof_ind=3)
# analytic_torsional_disp = T * L / (G * J)
# error = abs(torsional_disp - analytic_torsional_disp) / analytic_torsional_disp * 100
# print(f"error in torsional displacement: {error:.2f}%")

# ======== Axial-torsional buckling ======== #
# Cruciform column 
b = 0.076 # flange width (m)
t = 0.007 # flange thickness (m)

# Section properties 
A = 4 * b * t   
Iy = (2 * b * t**3) / 12 + (2 * t * b**3) / 12  
Iz = Iy 
J = (4 * b * t**3) / 3  
Iw = 0 

# Create beam model 
beam = Frame() 
beam.add_nodes([[i*L/n, 0, 0] for i in range(n+1)])

beam.add_material("steel", LinearElastic(rho=rho, E=E, G=G))
beam.add_section("cruciform", Section(A=A, Iy=Iy, Iz=Iz, J=J, Iw=Iw, y0=0, z0=0))
beam.add_elements([[i, i+1] for i in range(n)], "steel", "cruciform", element_class=ThinWalledBeamElement)

# Add boundary conditions
beam.add_boundary_condition(0, [0, 0, 0, 0, 0, 0, 0], BoundaryCondition) # fixed 
[beam.add_boundary_condition(i+1, [1, 0, 0, 1, 0, 0, 1], BoundaryCondition) for i in range(n)]  # lateral modes constrained 

# Apply end load
beam.add_nodal_load(n, [-1, 0, 0, 0, 0, 0, 0], NodalLoad)

eigenvalues, eigenvectors = beam.solve_eigen(num_modes=1)  # single torsional mode 
for n in range(len(eigenvalues)):
    load_analytic = G * J * A / (Iy + Iz)
    error = (np.abs(eigenvalues[n] - load_analytic)) / load_analytic * 100
    print(f"mode {n+1}: PcrT analytic = {load_analytic:.4e} | PcrT fea {eigenvalues[n]:.4e} | error = {error:.2f} %") 
    beam.show_mode_shape(eigenvectors[n], scale=10) 