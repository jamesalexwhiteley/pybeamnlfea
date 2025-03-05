from pybeamnlfea.model.frame import Frame
from pybeamnlfea.model.material import LinearElastic
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import NodalLoad 
from pybeamnlfea.solver.assembly import Assembler 
from pybeamnlfea.solver.linear import LinearSolver 

# Author: James Whiteley (github.com/jamesalexwhiteley)

# Create a beam structure
beam = Frame()

# Add nodes
nelems = 10
length = 1 # m 
beam.add_nodes([[(length / nelems) * i, 0, 0] for i in range(nelems + 1)])

# Add material and section 
beam.add_material("material", LinearElastic(1, 1))
beam.add_section("section", Section(A=1, Ix=1, Iy=1, J=1, Iw=1, x0=0, y0=0))

# Add element 
beam.add_elements([[i, i+1] for i in range(nelems)], "material", "section", element_class=ThinWalledBeamElement)

# Add boundary conditions and loads
beam.add_boundary_condition(0, [0, 0, 0, 0, 0, 0, 0], BoundaryCondition) 
beam.add_nodal_load(nelems, [0, -1, 0, 0, 0, 0, 0], NodalLoad) # vertical load at node 1                                     

assembler = Assembler(beam)
K = assembler.assemble_stiffness_matrix()
F = assembler.assemble_force_vector()
displacements = LinearSolver().solve(assembler)

# Print displacements for node 2
for (n, dof), displacement in displacements.items():
    if n == 2:
        if dof == 1:
            dof_name = beam.nodes[0].dof_names[dof]
            unit = "mm" if dof <= 2 else "rad" if dof <= 5 else "rad/mm"
            multiplier = 1e3 if dof <= 2 else 1.0 if dof <= 5 else 1e3
            
            print(f"Node 2, {dof_name}: {displacement * multiplier:.6f} {unit}")

# # Solve the model
# results = beam.solve()

# # Get results
# disp_node2 = beam.get_displacement_at_node(2)
# print(f"Displacement at node 2: {disp_node2}")

# (UB127x76x13)
# beam.add_material("material", LinearElastic(210e9, 0.3))
# beam.add_section("section", Section(A=16.5e-4, Ix=74.6e-6, Iy=14.7e-4, J=2.85e-8, Iw=2e-9, x0=0, y0=0))