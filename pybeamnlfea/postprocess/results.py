import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Results:
    def __init__(self, assembler, global_displacements, global_forces=None):
        """
        Results class to store and process solution results.

        """
        self.assembler = assembler
        self.frame = assembler.frame
        self.global_displacements = global_displacements
        self.global_forces = global_forces
    
    def extract_local_dofs(self, element, R):
        """
        Extract the 14 DOFs in local coordinates for an element
        [[ux, uy, uz, θx, θy, θz, φ]_1, [ux, uy, uz, θx, θy, θz, φ]_2] 
        """
        # Get nodes
        start_node, end_node = element.nodes
        n1_id, n2_id = start_node.id, end_node.id
        
        # Collect global DOFs for both nodes
        dof1_global = np.zeros(7)
        dof2_global = np.zeros(7)
        
        # Fill in available displacements from dictionary
        for dof_ind in range(7):  # Assuming 7 DOFs per node
            dof1_global[dof_ind] = self.global_displacements.get((n1_id, dof_ind), 0.0)
            dof2_global[dof_ind] = self.global_displacements.get((n2_id, dof_ind), 0.0)
        
        # Split into translations, rotations, and warping
        t1g, r1g, w1g = dof1_global[:3], dof1_global[3:6], dof1_global[6]
        t2g, r2g, w2g = dof2_global[:3], dof2_global[3:6], dof2_global[6]

        # Transform translations and rotations to local coordinates
        t1l = R.T @ t1g
        t2l = R.T @ t2g
        r1l = R.T @ r1g
        r2l = R.T @ r2g

        return np.hstack([t1l, r1l, w1g, t2l, r2l, w2g])
    
    def shape_thin_walled_beam(self, xi, L, dof_loc):
        """
        Shape functions for thin-walled beam.
        """
        # Extract local DOFs
        (u1, v1, w1, rx1, ry1, rz1, phi1,
         u2, v2, w2, rx2, ry2, rz2, phi2) = dof_loc

        # Linear interpolation for axial 
        Nx1 = 1 - xi
        Nx2 = xi
        u_xl = Nx1*u1 + Nx2*u2

        # Hermite cubic interpolation for bending
        H1 = 1 - 3*xi**2 + 2*xi**3
        H2 = 3*xi**2 - 2*xi**3
        H3 = L*(xi - 2*xi**2 + xi**3)
        H4 = L*(-xi**2 + xi**3)

        # Bending about z_l (v displacement)
        v_yl = H1*v1 + H2*v2 + H3*rz1 + H4*rz2

        # Bending about y_l (w displacement)
        w_zl = H1*w1 + H2*w2 - H3*ry1 - H4*ry2  # Negative signs for right-hand rule

        # Linear interpolation for torsion rotation
        rx_xl = Nx1*rx1 + Nx2*rx2

        # Linear interpolation for warping
        phi_xl = Nx1*phi1 + Nx2*phi2

        return (u_xl, v_yl, w_zl, rx_xl, phi_xl)
    
    def get_nodal_displacements(self, node_id, dof_ind=None, coordinate_system='global'):
        """
        Get displacement for a specific node.
        """
        if coordinate_system == 'global':
            if dof_ind is not None:
                return self.global_displacements.get((node_id, dof_ind), 0.0)
            else:
                # Return all DOFs for this node
                node_disps = {}
                for k, v in self.global_displacements.items():
                    if k[0] == node_id:
                        node_disps[k[1]] = v
                return node_disps
        else:
            raise ValueError("Local coordinate system not implemented for nodal displacements")
    
    def get_nodal_forces(self, node_id, dof_ind=None, coordinate_system='global'):
        """
        Get forces for a specific node.
        """
        if coordinate_system == 'global':
            if dof_ind is not None:
                return self.global_forces.get((node_id, dof_ind), 0.0)
            else:
                # Return all DOFs for this node
                node_forces = {}
                for k, v in self.global_forces.items():
                    if k[0] == node_id:
                        node_forces[k[1]] = v
                return node_forces
        else:
            raise ValueError("Local coordinate system not implemented for nodal forces")

    # def compute_element_internal_forces(self, element_id, xi): # NOTE (WIP)
    #     """
    #     Calculate interpolated internal forces at a point along the element.
        
    #     Args:
    #         element_id: Element identifier
    #         xi: Normalized position (0 to 1) along the element  
            
    #     Returns:
    #         dict: Internal forces at specified position
    #     """
    #     element = self.assembler.frame.elements[element_id]
    #     R = element.initial_state['R']
    #     L = element.initial_state['L']
    #     E = element.material.E
    #     G = element.material.G
        
    #     # Cross-section properties
    #     A = element.section.A
    #     Iy = element.section.Iy
    #     Iz = element.section.Iz
    #     J = element.section.J
    #     Iw = element.section.Iw
        
    #     # Get local DOFs
    #     local_dofs = self.extract_local_dofs(element, R)
        
    #     # Extract local DOFs
    #     (u1, v1, w1, rx1, ry1, rz1, phi1,
    #     u2, v2, w2, rx2, ry2, rz2, phi2) = local_dofs
        
    #     # Shape function derivatives for internal forces
        
    #     # 1. Axial Force (P)
    #     # Derivative of axial shape functions (constant)
    #     dNx1_dx = -1/L
    #     dNx2_dx = 1/L
        
    #     # Axial force P = EA * du/dx
    #     P = E * A * (dNx1_dx * u1 + dNx2_dx * u2)
        
    #     # 2. Torsional Moment (Mx)
    #     # Derivative of torsional shape functions (constant)
    #     dNrx1_dx = -1/L 
    #     dNrx2_dx = 1/L
        
    #     # Torsional moment Mx = GJ * dθx/dx
    #     Mx = G * J * (dNrx1_dx * rx1 + dNrx2_dx * rx2)
        
    #     # 3. Calculate second derivatives of bending shape functions
    #     # For Hermite cubic shape functions
        
    #     # d²H1/dx²
    #     d2H1_dx2 = -6/(L**2) * (1 - xi) + 12/(L**2) * xi
    #     # d²H2/dx²
    #     d2H2_dx2 = 6/(L**2) * (1 - xi) - 12/(L**2) * xi
    #     # d²H3/dx²
    #     d2H3_dx2 = -4/L * (1 - xi) + 6/L * xi
    #     # d²H4/dx²
    #     d2H4_dx2 = -2/L * (1 - xi) + 6/L * xi
        
    #     # 4. Bending Moments - using proper second derivatives
    #     # Curvatures (second derivatives of displacement)
    #     d2w_dx2 = d2H1_dx2 * w1 + d2H2_dx2 * w2 - d2H3_dx2 * ry1 - d2H4_dx2 * ry2
    #     d2v_dx2 = d2H1_dx2 * v1 + d2H2_dx2 * v2 + d2H3_dx2 * rz1 + d2H4_dx2 * rz2
        
    #     # Bending moments (M = EI * curvature)
    #     My = E * Iy * d2w_dx2
    #     Mz = -E * Iz * d2v_dx2
        
    #     # 5. Shear Forces (V = dM/dx)
    #     # Using curvature derivatives or directly from shape functions
    #     Vy = -E * Iz * d2v_dx2
    #     Vz = E * Iy * d2w_dx2
        
    #     # 6. Bimoment (Bw)
    #     # Derivative of warping shape functions
    #     dNphi1_dx = -1/L
    #     dNphi2_dx = 1/L
        
    #     # Bimoment Bw = -EIw * dφ/dx
    #     Bw = -E * Iw * (dNphi1_dx * phi1 + dNphi2_dx * phi2)
        
    #     return {
    #         'axial': P,
    #         'torsion': Mx,
    #         'shear_y': Vy,
    #         'shear_z': Vz,
    #         'moment_y': My,
    #         'moment_z': Mz,
    #         'bimoment': Bw,
    #         'position': xi
    #     }

    def compute_element_internal_forces(self, element_id, xi):
        """
        Calculate interpolated internal forces at a point along the element
        using simple linear interpolation from nodal forces.
        
        Args:
            element_id: Element identifier
            xi: Normalized position (0 to 1) along the element  
            
        Returns:
            dict: Internal forces at specified position
        """
        element = self.assembler.frame.elements[element_id]
        R = element.initial_state['R']
        
        # Get local DOFs
        local_dofs = self.extract_local_dofs(element, R)
        
        # Use the element's method to compute end forces
        k_elastic = element.compute_elastic_stiffness_matrix()
        internal_forces_vector = k_elastic @ local_dofs
        
        # Extract nodal forces (end 1)
        Fx1 = internal_forces_vector[0]
        Fy1 = internal_forces_vector[1]
        Fz1 = internal_forces_vector[2]
        Mx1 = internal_forces_vector[3]
        My1 = internal_forces_vector[4]
        Mz1 = internal_forces_vector[5]
        Bm1 = internal_forces_vector[6]
        
        # Extract nodal forces (end 2)
        Fx2 = -internal_forces_vector[7]  # Negative for equilibrium
        Fy2 = -internal_forces_vector[8]
        Fz2 = -internal_forces_vector[9]
        Mx2 = internal_forces_vector[10]
        My2 = internal_forces_vector[11]
        Mz2 = internal_forces_vector[12]
        Bm2 = internal_forces_vector[13]
        
        # Simple linear interpolation between end forces
        Fx = (1 - xi) * Fx1 + xi * Fx2
        Fy = (1 - xi) * Fy1 + xi * Fy2
        Fz = (1 - xi) * Fz1 + xi * Fz2
        Mx = (1 - xi) * Mx1 + xi * Mx2
        My = (1 - xi) * My1 + xi * My2
        Mz = (1 - xi) * Mz1 + xi * Mz2
        Bm = (1 - xi) * Bm1 + xi * Bm2
        
        return {
            'axial': Fx,
            'shear_y': Fy,
            'shear_z': Fz,
            'torsion': Mx,
            'moment_y': My,
            'moment_z': Mz,
            'bimoment': Bm,
            'position': xi
        }

    def get_element_forces(self, element_id, n_points=5, force_type='all', summary_type='max'):
        """
        Get summary of internal forces for a specific element.
        
        Args:
            element_id: Element identifier
            n_points: Number of points to sample along the element 
            force_type: Type of force to analyze ('all', 'axial', 'shear_y', 'shear_z', 
                    'torsion', 'moment_y', 'moment_z', 'bimoment')
            summary_type: Type of summary ('max', 'min', 'avg', 'range', 'all')
            
        Returns:
            dict: Summary of internal forces with requested statistics
        """
        # Validate inputs
        valid_force_types = ['all', 'axial', 'shear_y', 'shear_z', 'torsion', 
                            'moment_y', 'moment_z', 'bimoment']
        valid_summary_types = ['max', 'min', 'avg', 'range', 'all']
        
        if force_type not in valid_force_types:
            raise ValueError(f"Invalid force_type. Must be one of {valid_force_types}")
        
        if summary_type not in valid_summary_types:
            raise ValueError(f"Invalid summary_type. Must be one of {valid_summary_types}")
        
        # Generate sample points
        xi_points = np.linspace(0, 1, n_points)
        
        # Collect forces at each point
        forces_at_points = []
        for xi in xi_points:
            point_forces = self.compute_element_internal_forces(element_id, xi)
            forces_at_points.append(point_forces)
        
        # Initialize results dictionary
        result = {
            'element_id': element_id,
            'n_points': n_points,
            'xi_points': xi_points.tolist()
        }
        
        # Handle force types
        if force_type == 'all':
            force_types_to_process = valid_force_types[1:]  # All except 'all'
        else:
            force_types_to_process = [force_type]
        
        # Process each force type
        for force in force_types_to_process:
            # Extract values for this force type
            force_values = [point_forces[force] for point_forces in forces_at_points]
            
            # Calculate statistics
            force_max = max(force_values)
            force_min = min(force_values)
            force_avg = sum(force_values) / len(force_values)
            force_range = force_max - force_min
            
            # Find position of extrema
            max_index = force_values.index(force_max)
            min_index = force_values.index(force_min)
            max_position = xi_points[max_index]
            min_position = xi_points[min_index]
            
            # Store results based on summary type
            if summary_type == 'max' or summary_type == 'all':
                result[f"{force}_max"] = force_max
                result[f"{force}_max_position"] = float(max_position)
            
            if summary_type == 'min' or summary_type == 'all':
                result[f"{force}_min"] = force_min
                result[f"{force}_min_position"] = float(min_position)
            
            if summary_type == 'avg' or summary_type == 'all':
                result[f"{force}_avg"] = force_avg
            
            if summary_type == 'range' or summary_type == 'all':
                result[f"{force}_range"] = force_range
        
        # Add all sampled points if requested
        if summary_type == 'all':
            result['sampled_points'] = forces_at_points
        
        return result
        