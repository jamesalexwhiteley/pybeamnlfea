import numpy as np
from pybeamnlfea.model.frame import Frame
from pybeamnlfea.model.material import LinearElastic
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import NodalLoad 

# Author: James Whiteley (github.com/jamesalexwhiteley)

# Create a beam structure 
beam = Frame() 

n = 20
radius = 1 
theta_max = np.pi 
nodes = []
for i in range(n + 1):
    theta = (i / n) * theta_max  
    x = radius * np.cos(theta)  
    y = radius * np.sin(theta)  
    nodes.append([x, y, 0])  
beam.add_nodes(nodes)

# Add material and section (UB127x76x13) 
beam.add_material("material", LinearElastic(rho=1, E=1, G=1)) 
beam.add_section("section", Section(A=1, Iy=1, Iz=1, J=1, Iw=1, y0=0, z0=0)) 

# Add element 
beam.add_elements([[i, i+1] for i in range(n)], "material", "section", element_class=ThinWalledBeamElement) 

# Add boundary conditions and loads
beam.add_boundary_condition(0, [0, 0, 0, 0, 0, 0, 1], BoundaryCondition) 
beam.add_boundary_condition(n, [0, 0, 0, 0, 0, 0, 1], BoundaryCondition) 
beam.add_nodal_load(n/2, [0, 0, -1, 0, 0, 0, 0], NodalLoad) 

# Support torsion  
torsion_analytic = 1 * (1 / 2 - 1 / np.pi)  # test_beam.pdf 

# Solve the model
results = beam.solve() 
torsion_fea = results.get_nodal_forces(0)[4]
print(f"torsion analytic = {torsion_analytic:.4e} Nm | torsion fea = {torsion_fea:.4e} Nm | error = {(np.abs(torsion_fea - torsion_analytic)) / torsion_analytic * 100:.4f} %") 
beam.show_deformed_shape() 

