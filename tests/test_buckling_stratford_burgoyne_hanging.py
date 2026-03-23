"""
Reproduce result from Stratford & Burgoyne (1999)
"Lateral stability of long precast concrete beams" 

"""

from pybeamnlfea.model.frame import Frame
from pybeamnlfea.model.material import LinearElastic
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import UniformLoad, NodalLoad
import numpy as np

# Author: James Whiteley (github.com/jamesalexwhiteley) 

# Create a beam structure 
n = 20 
L = 10 # m 
beam = Frame() 
beam.add_nodes([[i*L/n, 0, 0] for i in range(n+1)]) 

# Concrete properties (from SY-6 column)
E = 34e9    # 34 GPa 
G = 14.8e9  # 14.8 GPa 
# rho = 2400  # kg/m3  

# SY-6 40m section properties
d = 2.0               # m
A = 0.709             # m2 
Iy = 0.2837           # m4 
Iz = 0.014            # m4 
J = 0.0221            # m4 
Iw = 0.0      
yb = 0.709            # m height of centroid above soffit 
h = 1.6               # m load height 
rho = (16.73 * 1e3 / 9.81) / A  # kg/m3   # rho = (w/g) / A; w = 16.73 kN/m

beam.add_material("steel", LinearElastic(rho=rho, E=E, G=G))
beam.add_section("rectangular", Section(A=A, Iy=Iy, Iz=Iz, J=J, Iw=Iw, y0=(d - yb), z0=0))

# Add element 
beam.add_elements([[i, i+1] for i in range(n)], "steel", "rectangular", element_class=ThinWalledBeamElement) 

# Add boundary conditions; Global (ux, uy, uz, θx, θy, θz, φ); 0=fixed, 1=free; (simply supported)
beam.add_boundary_condition(0, [0, 0, 0, 0, 1, 1, 1], BoundaryCondition)
beam.add_boundary_condition(n, [0, 0, 0, 0, 1, 1, 1], BoundaryCondition)

# beam.add_elastic_boundary_condition(0, dof_index=3, stiffness=1e12) 

# Add loads (self weight)
# beam.add_gravity_load([0, 0, -1])
[beam.add_uniform_load(element_id=e, forces=[0, 0, -1], load_height=h) for e in range(n)] 

# # Linear solver 
# results = beam.solve() 
# beam.show_deformed_shape(scale=1e-3, cross_section_scale=5) 

# Linear eigenvalue analysis 
eigenvalues, eigenvectors = beam.solve_eigen(num_modes=1) 
for n in range(len(eigenvalues)):
    
    # wcr_stratford = 28.5  # Stratford and Burgoyne coefficient
    wcr_stratford = 120  # Stratford and Burgoyne coefficient
    
    # eigenvalues[n] is a load factor; critical load = factor × applied load
    # wcr_actual = eigenvalues[n] * (rho * A * 9.81)  # self-weight
    wcr_actual = eigenvalues[n] * 1
    # wcr_fea = wcr_actual / (np.sqrt(G * J * E * Iz) / L**3)
    wcr_fea = wcr_actual / ((E * Iz * h) / L**4)
    
    error = (np.abs(wcr_fea - wcr_stratford)) / wcr_stratford * 100
    print(f"mode {n+1}: wcr analytic = {wcr_stratford:.4e} | wcr fea {wcr_fea:.4e} | error = {error:.2f} %")
    beam.show_mode_shape(eigenvectors[n], scale=5, cross_section_scale=5)