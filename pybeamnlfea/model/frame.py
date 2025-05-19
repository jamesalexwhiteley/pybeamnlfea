import numpy as np
from typing import List, Dict, Tuple, Optional, Union

from pybeamnlfea.model.node import Node 
from pybeamnlfea.model.material import Material 
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import Element 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import Load, NodalLoad, UniformLoad

from pybeamnlfea.solver.assembly import Assembler 
from pybeamnlfea.solver.linear import LinearSolver 
from pybeamnlfea.solver.eigen import EigenSolver 
from pybeamnlfea.postprocess.results import Results
from pybeamnlfea.postprocess.visualiser import Visualiser

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Frame:
    """
    Structural frame FEA model. 
    """
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.elements: Dict[int, Element] = {}
        self.materials: Dict[str, Material] = {}
        self.sections: Dict[str, Section] = {}
        self.boundary_conditions: Dict[int, BoundaryCondition] = {}
        self.loads: Dict[int, Load] = {} 
        self.self_weight: float = 0 
        self.results = None
        
    def add_node(self, x: float, y: float, z: float, node_id: int = None) -> Node:
        """Add a node to the frame."""
        if node_id is None:
            # Find the next available ID
            node_id = 0 if not self.nodes else max(self.nodes.keys()) + 1
        elif node_id in self.nodes:
            raise ValueError(f"Node ID {node_id} already exists")
        
        # Create and store the node
        node = Node(node_id, np.array([x, y, z]))
        self.nodes[node_id] = node
        
        return node
    
    def add_nodes(self, coords: list[list[float]]) -> None:
        """Add nodes to the frame."""
        for coord in coords: 
            self.add_node(coord[0], coord[1], coord[2])
    
    def add_material(self, name: str, material: Material) -> None:
        """Add a material to the frame."""
        if name in self.materials:
            raise ValueError(f"Material '{name}' already exists")
        
        self.materials[name] = material
        
    def add_section(self, name: str, section: Section) -> None:
        """Add a cross-section to the frame."""
        if name in self.sections:
            raise ValueError(f"Section '{name}' already exists")
        self.sections[name] = section

    def reset_sections(self) -> None: 
        """
        Reseat all sections. 
        """
        self.sections = {} 
            
    def add_element(self, node_ids: List[int], material_name: str, section_name: str, element_class: Element = Element, element_id: int=None) -> Element:
        """Add an element to the frame, connecting specified nodes with given properties."""
        # Find the nodes by ID
        element_nodes = []
        for node_id in node_ids:
            if node_id not in self.nodes:
                raise ValueError(f"Node {node_id} not found in the frame")
            element_nodes.append(self.nodes[node_id])
        
        # Get the material and section
        if material_name not in self.materials:
            raise KeyError(f"Material '{material_name}' not found")
        material = self.materials[material_name]
        
        if section_name not in self.sections:
            raise KeyError(f"Section '{section_name}' not found")
        section = self.sections[section_name]
        
        # Assign an ID to the element
        if element_id is None:
            # Find the next available ID
            element_id = 0 if not self.elements else max(self.elements.keys()) + 1
        elif element_id in self.elements:
            raise ValueError(f"Element ID {element_id} already exists")
        
        element = element_class(element_id, element_nodes, material, section)
        
        # Add the element to the dictionary
        self.elements[element_id] = element
        
        return element
    
    def reset_elements(self) -> None: 
        """
        Reseat all elements. 
        """
        self.elements = {} 
    
    def add_elements(self, node_ids_list: List[list[int]], material_name: str, section_name: str, element_class=Element, element_id=None) -> None:
        """Add elements to the frame."""
        for node_ids in node_ids_list: 
            self.add_element(node_ids, material_name, section_name, element_class, element_id)
        
    def add_boundary_condition(self, node_id: int, constraints: List[bool], boundary_class) -> None:
        """Add a boundary condition to a node IN GLOBAL COORDINATES."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found in the frame")
        
        self.boundary_conditions[node_id] = boundary_class(node_id, constraints)

    def add_elastic_boundary_condition(self, node_id: int, dof_index: int, stiffness: float, 
                                    prescribed_displacement: float = 0.0) -> None:
        """
        Add an elastic boundary condition to a node.
        
        Args:
            node_id: Node ID
            dof_index: Index of DOF (0=ux, 1=uy, 2=uz, 3=θx, 4=θy, 5=θz, 6=φ)
            stiffness: Stiffness value of the elastic support
            prescribed_displacement: Optional prescribed displacement/rotation
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found in the frame")
        
        # Create a new boundary condition if none exists
        if node_id not in self.boundary_conditions:
            # Initialize with all DOFs free
            self.boundary_conditions[node_id] = BoundaryCondition(node_id, [True, True, True, True, True, True, True])
        
        # Add the elastic support to the existing boundary condition
        self.boundary_conditions[node_id].add_elastic_support(dof_index, stiffness, prescribed_displacement)

    def reset_boundary_conditions(self) -> None: 
        """
        Reseat all boundary conditions. 
        """
        self.boundary_conditions = {} 
        
    def add_nodal_load(self, node_id: int, forces: List[float], load_class=NodalLoad) -> None:
        """Add a load to a node IN THE GLOBAL COORDINATE SYSTEM."""
        if node_id not in self.nodes:   
            raise ValueError(f"Node {node_id} not found in the frame")
        
        self.loads[node_id] = load_class(node_id, forces)

    def add_uniform_load(self, element_id: int, forces: List[float], load_class=UniformLoad) -> None:
        """
        Add a uniform load to an element IN ITS LOCAL COORDINATE SYSTEM.
        
        Args:
            element_id: Element to apply the load to
            force_vector: [wx, wy, wz] intensities in local coordinates (force per unit length)
                - wx: axial direction 
                - wy, wz: transverse directions 
        """
        if element_id not in self.elements:
            raise ValueError(f"Element {element_id} not found in the frame")
        
        # Store the uniform load object
        if not hasattr(self, 'uniform_loads'):
            self.uniform_loads = {}
        self.uniform_loads[element_id] = load_class(element_id, forces)
        
        # Get element information
        element = self.elements[element_id]
        node_i_id, node_j_id = element.nodes[0].id, element.nodes[1].id
        length = element.L
        
        # Get local components
        wx, wy, wz = forces
        
        # Calculate nodal forces and moments in local coordinates
        local_force_i = np.zeros(7)  # [Fx, Fy, Fz, Mx, My, Mz, Bx]
        local_force_j = np.zeros(7)
        
        # Axial load (wx) - only creates forces, no moments
        local_force_i[0] = wx * length / 2
        local_force_j[0] = wx * length / 2
        
        # Transverse loads - create both forces and moments
        # wy load (local y-direction)
        local_force_i[1] =  wy * length / 2 
        local_force_j[1] =  wy * length / 2 
        local_force_i[5] = -wy * length**2 / 12   # Mz at node i 
        local_force_j[5] =  wy * length**2 / 12   # Mz at node j 
        
        # wz load (local z-direction)
        local_force_i[2] =  wz * length / 2
        local_force_j[2] =  wz * length / 2
        local_force_i[4] = -wz * length**2 / 12   # My at node i
        local_force_j[4] =  wz * length**2 / 12   # My at node j
        
        # Transform local forces to global coordinates
        T = element.compute_transformation_matrix() 
        T_i = T[:7, :7] 
        T_j = T[7:, 7:] 
        
        # Transform local forces -> global coordinates
        global_force_i = T_i @ local_force_i
        global_force_j = T_j @ local_force_j

        # Add or update existing nodal loads in global coordinates
        if node_i_id in self.loads:
            self.loads[node_i_id].force_vector += global_force_i
        else:
            self.loads[node_i_id] = NodalLoad(node_i_id, global_force_i)
        
        if node_j_id in self.loads:
            self.loads[node_j_id].force_vector += global_force_j
        else:
            self.loads[node_j_id] = NodalLoad(node_j_id, global_force_j)

    def get_self_weight(self) -> float: 
        """Return self weight of frame object."""
        if self.self_weight == 0: 
            for element_id in self.elements:
            
                # Weight of element -> gravity load 
                element = self.elements[element_id]
                vol = element.section.A * element.L 
                weight = 9.81 * vol * element.material.density 
                self.self_weight += weight
            
        return self.self_weight

    def add_gravity_load(self, scale: List[float]=[0, 0, 1]) -> None:
        """
        Add a uniform load to an element in its local coordinate system equivalent to self weight under gravity.
        
        Args:
            element_id: Element to apply the load to
            scale: [wx, wy, wz] scale in local coordinates (force per unit length)

        """
        self.self_weight = 0 
        for element_id in self.elements:
            
            # Weight of element -> gravity load 
            element = self.elements[element_id]
            vol = element.section.A * element.L 
            weight = 9.81 * vol * element.material.density 
            self.self_weight += weight
        
            self.add_uniform_load(element_id, np.array(scale) * (weight / element.L), UniformLoad) 

    def reset_loads(self) -> None: 
        """
        Reseat all loads. 
        """
        self.loads = {} 

    def update_state(self, global_displacements):
        """
        Update the deformed state of the entire frame.
        
        Args:
            global_displacements: Dictionary mapping node IDs to displacements
        """
        if isinstance(global_displacements, dict):
            nodal_displacements = global_displacements
        else:
            raise ValueError("global_displacements must be dictionary of form (node_id, dof): value")
        
        # Update each element's state
        for elem_id, element in self.elements.items():
            # Get element nodal displacements 
            node1_id, node2_id = element.nodes[0].id, element.nodes[1].id
            disp1 = [nodal_displacements.get((node1_id, i)) for i in range(7)] # [ux,uy,uz,θx,θy,θz,θx'] 
            disp2 = [nodal_displacements.get((node2_id, i)) for i in range(7)]
            
            # Extract translational displacements
            trans_disp1 = disp1[:3] # [ux,uy,uz]
            trans_disp2 = disp2[:3]
            
            # Update element's state (position and local axes)
            element.update_state([trans_disp1, trans_disp2]) # NOTE only translation updated 

    def reset_state(self):
        """Reset the state of all elements to their initial configuration."""
        for elem_id, element in self.elements.items():
            element.reset_state()
            
        # Clear any stored results
        self.results = None

    def solve(self, solver_type: str='direct') -> Results:
        """Solve the frame model with a LinearSolver and return results."""
        # Assemble 
        assembler = Assembler(self)
        
        # Run solver
        solver = LinearSolver(solver_type=solver_type)
        nodal_displacements, nodal_forces = solver.solve(assembler)
        results = Results(assembler, nodal_displacements, nodal_forces)
        
        # Store results 
        self.results = results 
        self.update_state(nodal_displacements)  
        return results 
    
    def solve_eigen(self, num_modes: int=5) -> Results:
        """Solve the frame model with a linear EigenSolver and return results."""

        self.assembler = Assembler(self)
        solver = EigenSolver(num_modes=num_modes)
        self.critical_loads, self.buckling_modes = solver.solve_eigen(self.assembler)
    
        return self.critical_loads, self.buckling_modes 
    
    def show_deformed_shape(self, scale: float=1.0, show_undeformed: bool=True, show_local_axes: bool=True, 
             show_cross_section: bool=True, cross_section_scale: float=1.0) -> None:
        """Plot the current step of the frame analysis."""

        if self.results is None:
            visualiser = Visualiser(self) 
            visualiser.plot_undeformed_model(
                show_local_axes=show_local_axes
            )
            visualiser.show()
        
        else: 
            visualiser = Visualiser(self, self.results) 
            visualiser.plot_deformed_shape(
                scale=scale, 
                show_undeformed=show_undeformed,
                show_local_axes=show_local_axes, 
                show_cross_section=show_cross_section,
                cross_section_scale=cross_section_scale
            )
            visualiser.show()

    def show_force_field(self, force_type: str='Fx', scale: float=1.0, npoints: int=10) -> None: 
        """Plot the current step of the frame analysis."""

        if self.results is None:
            print("Error: self.results is None")
        
        else: 
            visualiser = Visualiser(self, self.results)
            visualiser.plot_undeformed_model(show_local_axes=True)
            visualiser.plot_force_field(force_type=force_type, scale=scale, npoints=npoints, line_width=3, show_values=True, value_frequency=5)
            visualiser.show()

    def show_mode_shape(self, mode, scale: float=1.0, show_undeformed: bool=True, show_local_axes: bool=False, 
                        show_cross_section: bool=True, cross_section_scale: float=1.0) -> None: 
        """Plot the deformed mode shape.""" 

        results = Results(self.assembler, mode) 
        self.update_state(results.global_displacements) 

        visualiser = Visualiser(self, results)
        visualiser.plot_deformed_shape(
            scale=scale, 
            show_undeformed=show_undeformed,
            show_local_axes=show_local_axes,
            show_cross_section=show_cross_section,
            cross_section_scale=cross_section_scale
        )

        visualiser.show()
        self.reset_state()
        
    def show_mode_shapes(self, scale: float=1.0, show_undeformed: bool=True) -> None:
        """Plot the deformed mode shapes."""

        if self.buckling_modes is None:
            print("Eigen has not been solved yet. Solving with default settings...")
            self.solve_eigen()
        
        for i, (mode, load_factor) in enumerate(zip(self.buckling_modes, self.critical_loads)):
            print(f"Mode {i+1}: Critical load factor = {load_factor}")
            self.show_mode_shape(mode, scale=scale, show_undeformed=show_undeformed)