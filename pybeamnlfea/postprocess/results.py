import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Results:
    def __init__(self, assembler, global_displacements, global_forces):
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
        for dof_idx in range(7):  # Assuming 7 DOFs per node
            dof1_global[dof_idx] = self.global_displacements.get((n1_id, dof_idx), 0.0)
            dof2_global[dof_idx] = self.global_displacements.get((n2_id, dof_idx), 0.0)
        
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
        Shape functions for thin-walled beam with warping.

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
    
    def plot_deformed_shape(self, scale=1.0, npoints=20, figsize=(10, 8), show_undeformed=True, show_node_id=True):
        """
        Plot the deformed shape of the structure
        
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        plotted_nodes = set()
        
        for _, element in self.frame.elements.items():

            # Get node coordinates
            start_node, end_node = element.nodes
            start_pos = start_node.coords
            end_pos = end_node.coords
            
            # Element direction and length
            element_dir = end_pos - start_pos
            L = np.linalg.norm(element_dir)
            
            if L < 1e-14:
                continue

            # Plot undeformed configuration
            if show_undeformed:
                ax.plot([start_pos[0], end_pos[0]], 
                        [start_pos[1], end_pos[1]], 
                        [start_pos[2], end_pos[2]], 'k--', lw=0.5)
            
            # Extract local DOFs
            dof_loc = self.extract_local_dofs(element, element.R)

            # Interpolate along the element
            xyz_def = np.zeros((npoints+1, 3))
                
            for i in range(npoints+1):
                xi = i / npoints
                (u_xl, v_yl, w_zl, rx_xl, _) = self.shape_thin_walled_beam(xi, L, dof_loc)
                    
                # Local displacement vector
                disp_loc = np.array([u_xl, v_yl, w_zl])
                disp_g = element.R @ disp_loc
                
                # Base point along undeformed element
                base = start_pos + xi * element_dir
                
                # Apply displacement
                xyz_def[i] = base + scale*disp_g

            ax.plot(xyz_def[:, 0], xyz_def[:, 1], xyz_def[:, 2], 'b-', lw=1.5)
                
            # Plot the deformed endpoints
            ax.scatter(xyz_def[0, 0], xyz_def[0, 1], xyz_def[0, 2], color='b', s=25)
            ax.scatter(xyz_def[-1, 0], xyz_def[-1, 1], xyz_def[-1, 2], color='b', s=25)

            if show_node_id:
                # Start node
                if start_node.id not in plotted_nodes:
                    ax.text(xyz_def[0, 0], xyz_def[0, 1], xyz_def[0, 2], 
                            f' {start_node.id}', fontsize=8, ha='left', va='bottom')
                    plotted_nodes.add(start_node.id)
                
                # End node
                if end_node.id not in plotted_nodes:
                    ax.text(xyz_def[-1, 0], xyz_def[-1, 1], xyz_def[-1, 2], 
                            f' {end_node.id}', fontsize=8, ha='left', va='bottom')
                    plotted_nodes.add(end_node.id)

        # Configure the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Remove background
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Make transparent
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        # plt.axis('off') 
        plt.gca().set_aspect('equal', adjustable='box')    
        # ax.set_box_aspect([1, 1, 1])        
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    def get_nodal_displacement(self, node_id, dof_idx=None, coordinate_system='global'):
        """
        Get displacement for a specific node.
        
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
    
    def get_nodal_forces(self, node_id, dof_idx=None, coordinate_system='global'):
        """
        Get forces for a specific node.
        
        """
        if coordinate_system == 'global':
            if dof_idx is not None:
                return self.global_forces.get((node_id, dof_idx), 0.0)
            else:
                # Return all DOFs for this node
                node_forces = {}
                for k, v in self.global_forces.items():
                    if k[0] == node_id:
                        node_forces[k[1]] = v
                return node_forces
        else:
            raise ValueError("Local coordinate system not implemented for nodal forces")
    