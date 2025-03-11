# import numpy as np

# # Author: James Whiteley (github.com/jamesalexwhiteley)

# class Results:
#     def __init__(self, assembler, global_displacements, element_forces=None):
#         """
#         Results class to store and process solution results.
        
#         Args:
#             assembler: The assembler used to build the system
#             global_displacements: Dictionary mapping (node_id, dof_idx) to displacement values in global coords
#             element_forces: Dictionary of element forces and local displacements
#         """
#         self.assembler = assembler
#         self.frame = assembler.frame
#         self.global_displacements = global_displacements
#         self.element_forces = element_forces
        
#         # Calculate local displacements for each element
#         self.local_element_displacements = self._calculate_local_displacements()
    
#     def _calculate_local_displacements(self):
#         """
#         Transform global displacements to local coordinate system for each element.
#         """
#         local_displacements = {}
        
#         for element_id, element in self.frame.elements.items():
#             # Get transformation matrix for this element
#             T = element.compute_transformation_matrix()
            
#             # Get global displacements for all DOFs of this element
#             global_disp_vector = []
#             for node in element.nodes:
#                 for i in range(node.ndof):
#                     disp = self.global_displacements.get((node.id, i), 0.0)
#                     global_disp_vector.append(disp)
                    
#             global_disp_vector = np.array(global_disp_vector)
            
#             # Transform to local coordinates
#             # Note: T transforms local->global, so T.transpose() transforms global->local
#             local_disp_vector = T.transpose() @ global_disp_vector
            
#             # Store by element ID
#             local_displacements[element_id] = local_disp_vector
            
#         return local_displacements
    
#     def get_nodal_displacement(self, node_id, dof_idx=None, coordinate_system='global'):
#         """
#         Get displacement for a specific node.
#         """
#         if coordinate_system == 'global':
#             if dof_idx is not None:
#                 return self.global_displacements.get((node_id, dof_idx), 0.0)
#             else:
#                 # Return all DOFs for this node
#                 node_disps = {}
#                 for k, v in self.global_displacements.items():
#                     if k[0] == node_id:
#                         node_disps[k[1]] = v
#                 return node_disps
#         else:
#             raise ValueError("For local displacements, use get_element_local_displacement method")
    
#     def get_element_local_displacement(self, element_id):
#         """
#         Get displacements for an element in its local coordinate system.
#         """
#         return self.local_element_displacements.get(element_id, None)
    
#     def get_element_forces(self, element_id):
#         """
#         Get forces for a specific element (already in local coordinates).
#         """
#         return self.element_forces.get(element_id, None)
    
#     def calculate_element_deflection(self, element_id, num_points=100):
#         """
#         Calculate deflected shape of an element in its local coordinate system
#         """
#         element = self.frame.elements[element_id]
#         L = element.length
#         local_disps = self.local_element_displacements[element_id]
        
#         # Create points along element length
#         c = np.linspace(0, L, num_points)
#         deflection = np.zeros((num_points, 6))  # For 6 deflection components (ux, uy, uz, θx, θy, θz)
        
#         if len(element.nodes) == 2 and len(local_disps) == 14:
#             # First node 
#             ux1, uy1, uz1 = local_disps[0], local_disps[1], local_disps[2]  # Translations 
#             theta_x1, theta_y1, theta_z1, phi_x1 = local_disps[3], local_disps[4], local_disps[5], local_disps[6]  # Rotations
            
#             # Second node 
#             ux2, uy2, uz2 = local_disps[7], local_disps[8], local_disps[9]  # Translations 
#             theta_x2, theta_y2, theta_z2, phi_x2 = local_disps[10], local_disps[11], local_disps[12], local_disps[13]  # Rotations
            
#             for i, x in enumerate(c):
#                 # Calculate all 14 shape functions at point x
#                 N = [
#                     (L - x)/L,                          # N1
#                     (L**3 - 3*L*x**2 + 2*x**3)/L**3,    # N2
#                     (L**3 - 3*L*x**2 + 2*x**3)/L**3,    # N3
#                     (L**3 - 3*L*x**2 + 2*x**3)/L**3,    # N4
#                     -x + 2*x**2/L - x**3/L**2,          # N5
#                     x - 2*x**2/L + x**3/L**2,           # N6
#                     x - 2*x**2/L + x**3/L**2,           # N7
#                     x/L,                                # N8
#                     x**2*(3*L - 2*x)/L**3,              # N9
#                     x**2*(3*L - 2*x)/L**3,              # N10
#                     x**2*(3*L - 2*x)/L**3,              # N11
#                     x**2*(L - x)/L**2,                  # N12
#                     x**2*(-L + x)/L**2,                 # N13
#                     x**2*(-L + x)/L**2                  # N14
#                 ]

#                 uy_prime1, uy_prime2 = theta_z1, theta_z2
#                 uz_prime1, uz_prime2 = theta_y1, theta_y2
#                 # theta1, theta2, theta_prime1, theta_prime2 = theta_x1, theta_x2, phi_x1, phi_x2
                
#                 # Axial deflection (ux)
#                 deflection[i, 0] = N[0]*ux1 + N[7]*ux2

#                 # Transverse deflection in y-direction (uȳ)
#                 deflection[i, 1] = N[1]*uy1 + N[5]*uy_prime1 + N[8]*uy2 + N[12]*uy_prime2

#                 # Transverse deflection in z-direction (uz̄)
#                 deflection[i, 2] = N[2]*uz1 + N[4]*uz_prime1 + N[9]*uz2 + N[11]*uz_prime2
        
#         return c, deflection

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Results:
    def __init__(self, assembler, global_displacements, element_forces=None):
        """
        Results class to store and process solution results.
        
        Args:
            assembler: The assembler used to build the system
            global_displacements: Dictionary mapping (node_id, dof_idx) to displacement values in global coords
            element_forces: Dictionary of element forces and local displacements
        """
        self.assembler = assembler
        self.frame = assembler.frame
        self.global_displacements = global_displacements
        self.element_forces = element_forces
        
        # Convert dictionary to array format for compatibility with the plotting code
        self.a = self._dict_to_array()
        self.ndof = 7  # Number of DOFs per node (considering warping)
        self.nodes = {node.id: node.coords for node in self.frame.nodes.values()}
        self.elems = [[element.nodes[0].id, element.nodes[1].id] for element in self.frame.elements.values()]
        self.nelems = len(self.elems)
        
    def _dict_to_array(self):
        """Convert global displacement dictionary to array format."""
        # Find the maximum node ID and DOF index
        max_node_id = max([k[0] for k in self.global_displacements.keys()]) if self.global_displacements else 0
        max_dof_idx = max([k[1] for k in self.global_displacements.keys()]) if self.global_displacements else 0
        ndof = max_dof_idx + 1
        
        # Create array with zeros
        a = np.zeros((max_node_id + 1) * ndof)
        
        # Fill in the displacements
        for (node_id, dof_idx), value in self.global_displacements.items():
            a[node_id * ndof + dof_idx] = value
            
        return a
    
    def element_rotation_matrix(self, n1, n2):
        """
        Compute the rotation matrix for element from node n1 to n2.
        Similar to compute_transformation_matrix in the original code.
        """
        x1, y1, z1 = self.nodes[n1]
        x2, y2, z2 = self.nodes[n2]
        
        # Element direction vector (local x-axis)
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        L = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        if L < 1e-10:
            raise ValueError(f"Element {n1}-{n2} has zero length")
        
        ex = np.array([dx, dy, dz]) / L
        
        # Define local y-axis (perpendicular to x-axis)
        # If element is not vertical, use global Z for reference
        if abs(ex[2]) < 0.99:
            ey_ref = np.cross(ex, np.array([0, 0, 1]))
        else:
            # If nearly vertical, use global Y for reference
            ey_ref = np.cross(ex, np.array([0, 1, 0]))
            
        ey = ey_ref / np.linalg.norm(ey_ref)
        
        # Local z-axis completes right-handed system
        ez = np.cross(ex, ey)
        
        # Rotation matrix (each row is a local basis vector)
        R = np.vstack([ex, ey, ez])
        
        return R
    
    def extract_local_dofs(self, n1, n2, R):
        """
        For element from n1->n2, extract the 14 DOFs in local coordinates
        [u1, v1, w1, rx1, ry1, rz1, φ1, u2, v2, w2, rx2, ry2, rz2, φ2]
        """
        # Get global DOFs for both nodes
        dof1 = self.a[self.ndof*n1 : self.ndof*n1 + 7]
        dof2 = self.a[self.ndof*n2 : self.ndof*n2 + 7]

        # Split into translations, rotations, and warping
        t1g, r1g, w1g = dof1[:3], dof1[3:6], dof1[6]
        t2g, r2g, w2g = dof2[:3], dof2[3:6], dof2[6]

        # Transform translations and rotations to local coordinates
        t1l = R.T @ t1g
        t2l = R.T @ t2g
        r1l = R.T @ r1g
        r2l = R.T @ r2g

        # Warping DOFs stay the same (scalar values)
        return np.hstack([t1l, r1l, w1g, t2l, r2l, w2g])
    
    def shape_thin_walled_beam(self, xi, L, dof_loc):
        """
        Shape functions for thin-walled beam with warping.

        Args:
            xi: Normalized position along beam (0 to 1)
            L: Element length
            dof_loc: Local DOFs [u1, v1, w1, rx1, ry1, rz1, φ1, u2, v2, w2, rx2, ry2, rz2, φ2]
        
        Returns:
            (u_xl, v_yl, w_zl, rx_xl, phi): Interpolated displacements, rotations, and warping
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
        w_zl = H1*w1 + H2*w2 - H3*ry1 - H4*ry2  # Note negative signs for right-hand rule

        # Linear interpolation for torsion rotation
        rx_xl = Nx1*rx1 + Nx2*rx2

        # Linear interpolation for warping
        phi_xl = Nx1*phi1 + Nx2*phi2

        return (u_xl, v_yl, w_zl, rx_xl, phi_xl)
    
    def plot_deformed_shape(self, scale=1.0, npoints=20, figsize=(10, 8), show_warping=True, show_undeformed=True):
        """
        Plot the deformed shape of the structure
        
        Args:
            scale: Scale factor for the deformation
            npoints: Number of points to use for interpolation along each element
            figsize: Figure size
            show_warping: Whether to visualize warping effects with color
            show_undeformed: Whether to show the undeformed shape
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Track max warping for color scaling
        max_warping = 0
        
        # Process each element
        for e in range(self.nelems):
            n1, n2 = self.elems[e]
            x1, y1, z1 = self.nodes[n1]
            x2, y2, z2 = self.nodes[n2]
            dx, dy, dz = (x2 - x1), (y2 - y1), (z2 - z1)
            L = np.sqrt(dx*dx + dy*dy + dz*dz)
            if L < 1e-14:
                continue

            # Plot undeformed configuration
            if show_undeformed:
                ax.plot([x1, x2], [y1, y2], [z1, z2], 'k--', lw=0.5)

            # Get the rotation matrix
            R = self.element_rotation_matrix(n1, n2)
            
            # Extract local DOFs
            dof_loc = self.extract_local_dofs(n1, n2, R)

            # Build array of points along the element
            xyz_def = np.zeros((npoints+1, 3))
            if show_warping:
                warping_values = np.zeros(npoints+1)
                
            for i in range(npoints+1):
                xi = i / npoints
                if show_warping:
                    (u_xl, v_yl, w_zl, rx_xl, phi_xl) = self.shape_thin_walled_beam(xi, L, dof_loc)
                    warping_values[i] = abs(phi_xl)
                    max_warping = max(max_warping, abs(phi_xl))
                else:
                    (u_xl, v_yl, w_zl, rx_xl, _) = self.shape_thin_walled_beam(xi, L, dof_loc)
                    
                # Local displacement vector
                disp_loc = np.array([u_xl, v_yl, w_zl])
                disp_g = R @ disp_loc
                # Base point
                base = np.array([x1, y1, z1]) + xi*np.array([dx, dy, dz])
                xyz_def[i] = base + scale*disp_g

            # Plot the deformed shape
            if show_warping:
                # Use a colormap to visualize warping
                points = np.array([xyz_def[:-1, 0], xyz_def[:-1, 1], xyz_def[:-1, 2]]).T
                segments = np.array([xyz_def[:-1], xyz_def[1:]]).transpose((1, 0, 2))
                for i in range(npoints):
                    ax.plot([segments[i, 0, 0], segments[i, 1, 0]], 
                            [segments[i, 0, 1], segments[i, 1, 1]], 
                            [segments[i, 0, 2], segments[i, 1, 2]],
                            color=plt.cm.coolwarm(warping_values[i]/max(max_warping, 1e-10)), 
                            lw=1.5)
            else:
                ax.plot(xyz_def[:, 0], xyz_def[:, 1], xyz_def[:, 2], 'b-', lw=1.5)
                
            # Plot the deformed endpoints
            ax.scatter(xyz_def[0, 0], xyz_def[0, 1], xyz_def[0, 2], color='b', s=25)
            ax.scatter(xyz_def[-1, 0], xyz_def[-1, 1], xyz_def[-1, 2], color='b', s=25)

        # Configure the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Structural Deformation (scale factor: {scale})')
        
        # Set equal aspect ratio for all axes
        ax.set_box_aspect([1, 1, 1])
        
        # Remove background
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Make panes transparent
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        
        # Add a colorbar if showing warping
        if show_warping and max_warping > 1e-10:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(0, max_warping))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.1)
            cbar.set_label('Warping magnitude')
            
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', linestyle='--', lw=0.5, alpha=0.5, label='Undeformed'),
            Line2D([0], [0], color='blue', lw=1.5, label='Deformed')
        ]
        if show_warping:
            legend_elements.append(
                Line2D([0], [0], color='red', lw=1.5, label='High Warping')
            )
        ax.legend(handles=legend_elements, loc='upper right')
            
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    def get_nodal_displacement(self, node_id, dof_idx=None, coordinate_system='global'):
        """
        Get displacement for a specific node.
        
        Args:
            node_id: ID of the node
            dof_idx: Index of the degree of freedom (0=ux, 1=uy, 2=uz, 3=rx, 4=ry, 5=rz, 6=φ)
            coordinate_system: 'global' or 'local'
            
        Returns:
            Displacement value or dictionary of all DOFs for the node
        """
        if coordinate_system == 'global':
            if dof_idx is not None:
                return self.global_displacements.get((node_id, dof_idx), 0.0)
            else:
                # Return all DOFs for this node
                node_disps = {}
                for k, v in self.global_displacements.items():
                    if k[0] == node_id:
                        node_disps[k[1]] = v
                return node_disps
        else:
            raise ValueError("Local coordinate system not implemented for nodal displacements")
    
    def get_element_forces(self, element_id):
        """
        Get forces for a specific element (already in local coordinates).
        """
        return self.element_forces.get(element_id, None)