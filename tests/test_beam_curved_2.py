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

nelems = 10
radius = 1 
theta_max = np.pi 
nodes = []
for i in range(nelems + 1):
    theta = (i / nelems) * theta_max  
    x = radius * np.cos(theta)  
    y = radius * np.sin(theta)  
    nodes.append([x, y, 0])  
beam.add_nodes(nodes)

# Add material and section (UB127x76x13) 
beam.add_material("material", LinearElastic(rho=1, E=1, G=1)) 
beam.add_section("section", Section(A=1, Iy=1, Iz=1, J=1, Iw=1, y0=0, z0=0)) 

# Add element 
beam.add_elements([[i, i+1] for i in range(nelems)], "material", "section", element_class=ThinWalledBeamElement) 

# Add boundary conditions and loads
beam.add_boundary_condition(0, [0, 0, 0, 0, 0, 0, 1], BoundaryCondition) 
beam.add_boundary_condition(nelems, [0, 0, 0, 0, 0, 0, 1], BoundaryCondition) 

# Add gravity load 
beam.add_gravity_load([0, 0, 0.1])

# Solve the model
results = beam.solve() 
beam.show() 

# Linear buckling analysis 
beam.solve_eigen(num_modes=10)
beam.show_mode_shapes(scale=1.0)

