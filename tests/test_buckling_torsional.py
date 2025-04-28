from pybeamnlfea.model.frame import Frame
from pybeamnlfea.model.material import LinearElastic
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import NodalLoad 
import numpy as np

# Author: James Whiteley (github.com/jamesalexwhiteley) 

# Column 
n = 3 # num elements 
L = 3.0
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

beam.add_boundary_condition(0, [0, 0, 0, 0, 0, 0, 0], BoundaryCondition)  # Fixed
beam.add_boundary_condition(n, [0, 0, 1, 1, 1, 0, 0], BoundaryCondition)  # Free to rotate about x

# Apply torque
T = 1e0
V = 1e1
beam.add_nodal_load(n, [0, 0, -V, 0, 0, 0, 0], NodalLoad) # NOTE is global z updwards or downwards? Seems nodal load is different to uniform/gravity loads? 

results = beam.solve() 
# print(results.get_nodal_forces(0))
beam.show(scale=1e3) # NOTE presumably updated local system should "follow" deflected shape? Think only torsional displacement [of axes] is implemeted?

# eigenvalues, eigenvectors = beam.solve_eigen(num_modes=3)
# for i in range(len(eigenvalues)):
#     beam.show_mode_shape(eigenvectors[i], scale=10)

# # Try different formula interpretations
# T_cr1 = (np.pi/(2*L)) * np.sqrt(G*J * E*Iw)  # Standard formula
# T_cr2 = (np.pi**2/(4*L**2)) * E*Iw + G*J  # Alternative formula

# print(f"Mode 1: Critical torque = {eigenvalues[0]:.4e} N·m")
# print(f"Standard formula: {T_cr1:.4e} N·m, Ratio: {eigenvalues[0]/T_cr1:.4f}")
# print(f"Alternative formula: {T_cr2:.4e} N·m, Ratio: {eigenvalues[0]/T_cr2:.4f}")