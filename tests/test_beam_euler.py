from pybeamnlfea.model.frame import Frame
from pybeamnlfea.model.material import LinearElastic
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import NodalLoad 

# Author: James Whiteley (github.com/jamesalexwhiteley) 

# Create a beam structure 
n = 10
beam = Frame() 
beam.add_nodes([[i/n, 0, 0] for i in range(n+1)])

# Add material and section (UB127x76x13) 
beam.add_material("material", LinearElastic(E=1, G=1)) 
beam.add_section("section", Section(A=1, Iy=1, Iz=1e4, J=1, Iw=1, y0=0, z0=0)) 

# Add element 
beam.add_elements([[i, i+1] for i in range(n)], "material", "section", element_class=ThinWalledBeamElement) 

# Add boundary conditions and loads
beam.add_boundary_condition(0, [0, 0, 0, 0, 1, 0, 0], BoundaryCondition) 
beam.add_boundary_condition(n, [1, 0, 0, 0, 1, 0, 0], BoundaryCondition) 
beam.add_nodal_load(n, [-1, 0, 0, 0, 0, 0, 0], NodalLoad) 

# Linear buckling analysis 
beam.eigen_solve()
beam.show_mode_shapes()