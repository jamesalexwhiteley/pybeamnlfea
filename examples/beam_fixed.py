from pybeamnlfea.model.frame import Frame
from pybeamnlfea.model.material import LinearElastic
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import NodalLoad 
from pybeamnlfea.postprocess.visualise import Visualiser 
from pybeamnlfea.solver.assembly import Assembler 
from pybeamnlfea.solver.linear import LinearSolver 
from pybeamnlfea.postprocess.results import Results 

# Author: James Whiteley (github.com/jamesalexwhiteley)

# Create a beam structure
beam = Frame()

# Add nodes 
nelems = 10
length = 1 # m 
# beam.add_nodes([[0, 0, (length / nelems) * i] for i in range(nelems + 1)])
# beam.add_nodes([[0, (length / nelems) * i, 0] for i in range(nelems + 1)])
beam.add_nodes([[(length / nelems) * i, 0, 0] for i in range(nelems + 1)]) 

# Add material and section (UB127x76x13) 
beam.add_material("material", LinearElastic(1, 1)) 
beam.add_section("section", Section(A=1, Ix=1, Iy=1, J=1, Iw=1, x0=0, y0=0)) 

# Add element 
beam.add_elements([[i, i+1] for i in range(nelems)], "material", "section", element_class=ThinWalledBeamElement) 

# Add boundary conditions and loads
beam.add_boundary_condition(0, [0, 0, 0, 0, 0, 0, 0], BoundaryCondition) 
beam.add_nodal_load(nelems, [0, 0, 0, 1, 0, 0, 0], NodalLoad) # load at end node # NOTE spurious wiggles? 
# beam.add_nodal_load(nelems, [1, 1, 1, 1, 1, 1, 1], NodalLoad) # load at end node

# # Solve the model 
# results = beam.solve() 

assembler = Assembler(beam)
solver = LinearSolver(solver_type='direct')
node_disp = solver.solve(assembler)
results = Results(assembler, node_disp)

# print(results._calculate_local_displacements())
visualiser = Visualiser(results)
visualiser.plot_deformed_shape()

# # Plot the model TODO 
# results = beam.plot() 
