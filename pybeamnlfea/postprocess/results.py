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
        w_zl = H1*w1 + H2*w2 - H3*ry1 - H4*ry2  # Negative signs for right-hand rule

        # Linear interpolation for torsion rotation
        rx_xl = Nx1*rx1 + Nx2*rx2

        # Linear interpolation for warping
        phi_xl = Nx1*phi1 + Nx2*phi2

        return (u_xl, v_yl, w_zl, rx_xl, phi_xl)
    
    def get_nodal_displacements(self, node_id, dof_idx=None, coordinate_system='global'):
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
    