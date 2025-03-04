from pybeamnlfea.model.frame import Frame
from pybeamnlfea.model.material import LinearElastic
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import NodalLoad 

# Author: James Whiteley (github.com/jamesalexwhiteley)

# Create a beam structure
beam = Frame()

# Add nodes
n0, n1 = 0, 1 # tags 
beam.add_node(0, 0, 0, n0)
beam.add_node(1, 0, 0, n1)

# Add material and section (UB127x76x13)
beam.add_material("material", LinearElastic(210e9, 0.3))
beam.add_section("section", Section(A=16.5e-4, Ix=74.6e-6, Iy=14.7e-4, J=2.85e-8, Iw=2e-9))

# Add element 
beam.add_element([n0, n1], "material", "section", element_class=ThinWalledBeamElement)

print(beam.nodes)
print(beam.elements)

# Add boundary conditions and loads
beam.add_boundary_condition(0, [0, 0, 0, 1, 1, 1, 1], BoundaryCondition) 
beam.add_nodal_load(1, [0, -10000, 0, 0, 0, 0, 0], NodalLoad) # vertical load at node 1 

# # Solve the model
# results = beam.solve()

# # Get results
# disp_node2 = beam.get_displacement_at_node(2)
# print(f"Displacement at node 2: {disp_node2}")