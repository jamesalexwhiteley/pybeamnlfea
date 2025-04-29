import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Author: James Whiteley (github.com/jamesalexwhiteley)
# Modified to include thin-walled beam effects with warping

# ======================================================================= # 
# 3D Thin-Walled Frame model
# ======================================================================= # 
class ThinWalledFrame3D:
    def __init__(self, nodes, elems, E, G, A, Iy, Iz, J, Iw, x0=None, y0=None, r=None):
        """
        3D FEM model of a thin-walled frame with warping effects.

        Args:
            nodes : np.array(
                        [[x1, y1, z1], 
                        [x2, y2, z2], 
                        ... 
            elements : np.array( 
                        [[0, 1], 
                        [1, 2], 
                        ...
            E : float or array
                Young's modulus
            G : float or array
                Shear modulus
            A : float or array
                Cross-sectional area
            Iy : float or array
                Second moment of area about y axis
            Iz : float or array
                Second moment of area about z axis
            J : float or array
                Torsion constant
            Iw : float or array
                Warping constant
            x0, y0 : float or array, optional
                Coordinates of shear center relative to centroid
            r : float or array, optional
                Polar radius of gyration
        """
        self.nodes = nodes
        self.elems = elems
        self.nnodes = nodes.shape[0]
        self.nelems = elems.shape[0]
        self.ndof = 7  # 7 dofs per node (including warping)

        # Section properties
        self.E  = E  if hasattr(E,  '__len__') else np.full(self.nelems, E)
        self.G  = G  if hasattr(G,  '__len__') else np.full(self.nelems, G)
        self.A  = A  if hasattr(A,  '__len__') else np.full(self.nelems, A)
        self.Iy = Iy if hasattr(Iy, '__len__') else np.full(self.nelems, Iy)
        self.Iz = Iz if hasattr(Iz, '__len__') else np.full(self.nelems, Iz)
        self.J  = J  if hasattr(J,  '__len__') else np.full(self.nelems, J)
        self.Iw = Iw if hasattr(Iw, '__len__') else np.full(self.nelems, Iw)
        
        # Shear center coordinates (default to centroid)
        self.x0 = x0 if hasattr(x0, '__len__') else np.zeros(self.nelems) if x0 is None else np.full(self.nelems, x0)
        self.y0 = y0 if hasattr(y0, '__len__') else np.zeros(self.nelems) if y0 is None else np.full(self.nelems, y0)
        
        # Polar radius of gyration (default to sqrt(Iy+Iz)/A)
        if r is None:
            self.r = np.sqrt((self.Iy + self.Iz) / self.A)
        else:
            self.r = r if hasattr(r, '__len__') else np.full(self.nelems, r)

        # Initialise system matrices and vectors
        self.K = np.zeros((self.nnodes * self.ndof, self.nnodes * self.ndof))
        self.f = np.zeros(self.nnodes * self.ndof)
        self.a = np.zeros(self.nnodes * self.ndof)

        # DOFs by default are all free
        self.dof_free = np.arange(self.nnodes * self.ndof, dtype=int)
        self.dof_con = np.array([], dtype=int)
        self.dof_stiff = []  # finite stiffness supports

    def k_thin_walled_beam(self, E, G, A, Iy, Iz, Iw, J, L, x0=0, y0=0, r=0, 
                          P0=0, Mx0=0, My0=0, B0_bar=0, W_bar=0, beta_x=0, beta_y=0):
        """
        Local stiffness matrix for a 3D thin-walled beam element with warping.
        
        Returns a 14x14 stiffness matrix (7 DOFs per node).
        DOF order: [u1, v1, w1, θx1, θy1, θz1, φ1, u2, v2, w2, θx2, θy2, θz2, φ2]
        """
        # Initialise the stiffness matrix
        k = np.zeros((14, 14))
        
        # Helper function to set symmetric entries
        def set_symmetric(i, j, value):
            k[i, j] = value
            if i != j:
                k[j, i] = value
        
        # Axial terms
        set_symmetric(0, 0, A*E/L)
        set_symmetric(0, 7, -A*E/L)
        set_symmetric(7, 7, A*E/L)
        
        # Y-direction bending terms (using Iz for y-direction bending)
        set_symmetric(1, 1, 12*E*Iz/(L**3) + 6*P0/(5*L))
        set_symmetric(1, 8, -12*E*Iz/(L**3) - 6*P0/(5*L))
        set_symmetric(8, 8, 12*E*Iz/(L**3) + 6*P0/(5*L))
        
        set_symmetric(1, 3, -3*Mx0/(5*L) + 3*P0*y0/(5*L))
        set_symmetric(1, 10, 3*Mx0/(5*L) - 3*P0*y0/(5*L))
        set_symmetric(8, 10, -3*Mx0/(5*L) + 3*P0*y0/(5*L))
        set_symmetric(3, 8, -3*Mx0/(5*L) + 3*P0*y0/(5*L))
        
        set_symmetric(1, 5, 6*E*Iz/(L**2) + P0/10)
        set_symmetric(1, 12, 6*E*Iz/(L**2) + P0/10)
        set_symmetric(5, 8, -6*E*Iz/(L**2) - P0/10)
        set_symmetric(8, 12, -6*E*Iz/(L**2) - P0/10)
        
        cross_term_val = -Mx0/20 + P0*y0/20
        set_symmetric(1, 6, cross_term_val)
        set_symmetric(1, 13, cross_term_val)
        set_symmetric(8, 13, -cross_term_val)
        set_symmetric(5, 10, -cross_term_val)
        set_symmetric(3, 5, cross_term_val)
        set_symmetric(3, 12, cross_term_val)
        set_symmetric(6, 8, -cross_term_val)
        set_symmetric(10, 12, -cross_term_val)
        
        # X-direction bending terms (using Iy for x-direction bending)
        set_symmetric(2, 2, 12*E*Iy/(L**3) + 6*P0/(5*L))
        set_symmetric(2, 9, -12*E*Iy/(L**3) - 6*P0/(5*L))
        set_symmetric(9, 9, 12*E*Iy/(L**3) + 6*P0/(5*L))
        
        set_symmetric(2, 3, -3*My0/(5*L) - 3*P0*x0/(5*L))
        set_symmetric(2, 10, 3*My0/(5*L) + 3*P0*x0/(5*L))
        set_symmetric(3, 9, -3*My0/(5*L) - 3*P0*x0/(5*L))
        set_symmetric(9, 10, 3*My0/(5*L) + 3*P0*x0/(5*L))
        
        set_symmetric(2, 4, -6*E*Iy/(L**2) - P0/10)
        set_symmetric(2, 11, -6*E*Iy/(L**2) - P0/10)
        set_symmetric(4, 9, 6*E*Iy/(L**2) + P0/10)
        set_symmetric(9, 11, 6*E*Iy/(L**2) + P0/10)
        
        cross_term_val2 = -My0/20 - P0*x0/20
        set_symmetric(2, 6, cross_term_val2)
        set_symmetric(2, 13, cross_term_val2)
        set_symmetric(9, 13, -cross_term_val2)
        set_symmetric(3, 4, -cross_term_val2)
        set_symmetric(3, 11, -cross_term_val2)
        set_symmetric(10, 11, cross_term_val2)
        set_symmetric(4, 10, cross_term_val2)
        set_symmetric(6, 9, -cross_term_val2)
        
        # Warping and torsion terms
        warping_term = 12*E*Iw/(L**3) + 6*G*J/(5*L) + 6*P0*r**2/(5*L) + 3*Mx0*beta_y/(5*L) - 3*My0*beta_x/(5*L) - 6*B0_bar*W_bar/(5*L)
        set_symmetric(3, 3, warping_term)
        set_symmetric(3, 10, -warping_term)
        set_symmetric(10, 10, warping_term)
        
        warping_term2 = 6*E*Iw/(L**2) + G*J/10 + P0*r**2/10 + Mx0*beta_y/20 - My0*beta_x/20 - B0_bar*W_bar/10
        set_symmetric(3, 6, warping_term2)
        set_symmetric(3, 13, warping_term2)
        set_symmetric(6, 10, -warping_term2)
        set_symmetric(10, 13, -warping_term2)
        
        warping_term3 = 4*E*Iw/L + 2*G*J*L/15 + 2*L*P0*r**2/15 + L*Mx0*beta_y/15 - L*My0*beta_x/15 - 2*B0_bar*L*W_bar/15
        set_symmetric(6, 6, warping_term3)
        set_symmetric(13, 13, warping_term3)
        
        warping_term4 = 2*E*Iw/L - G*J*L/30 - L*P0*r**2/30 - L*Mx0*beta_y/60 + L*My0*beta_x/60 + B0_bar*L*W_bar/30
        set_symmetric(6, 13, warping_term4)

        # Additional terms
        set_symmetric(4, 4, 4*E*Iy/L + 2*L*P0/15)
        set_symmetric(11, 11, 4*E*Iy/L + 2*L*P0/15)
        
        set_symmetric(4, 6, L*My0/15 + L*P0*x0/15)
        set_symmetric(11, 13, L*My0/15 + L*P0*x0/15)
        
        set_symmetric(4, 13, -L*My0/60 - L*P0*x0/60)
        set_symmetric(6, 11, -L*My0/60 - L*P0*x0/60)
        
        set_symmetric(4, 11, 2*E*Iy/L - L*P0/30)
        
        set_symmetric(5, 5, 4*E*Iz/L + 2*L*P0/15)
        set_symmetric(12, 12, 4*E*Iz/L + 2*L*P0/15)
        
        set_symmetric(5, 6, -L*Mx0/15 + L*P0*y0/15)
        set_symmetric(12, 13, -L*Mx0/15 + L*P0*y0/15)
        
        set_symmetric(5, 12, 2*E*Iz/L - L*P0/30)
        
        set_symmetric(5, 13, L*Mx0/60 - L*P0*y0/60)
        set_symmetric(6, 12, L*Mx0/60 - L*P0*y0/60)

        return k

    def element_rotation_matrix(self, n1, n2):
        """
        Compute a (3x3) rotation matrix R for the element local axes {x_l, y_l, z_l}:
        - x_l along the element
        - y_l, z_l chosen to form a right-handed system 
        """
        x1, y1, z1 = self.nodes[n1]
        x2, y2, z2 = self.nodes[n2]
        dx, dy, dz = (x2 - x1), (y2 - y1), (z2 - z1)

        L = np.sqrt(dx*dx + dy*dy + dz*dz)
        if L < 1e-14:
            return np.eye(3)

        # local x-axis
        x_local = np.array([dx, dy, dz]) / L

        # "reference up" vector
        ref_up = np.array([0, 0, 1], dtype=float)

        # try cross(x_local, ref_up)
        y_l_temp = np.cross(x_local, ref_up)
        nrm = np.linalg.norm(y_l_temp)
        if nrm < 1e-12:
            # fallback if nearly vertical
            ref_up = np.array([0, 1, 0], dtype=float)
            y_l_temp = np.cross(x_local, ref_up)
            nrm = np.linalg.norm(y_l_temp)
            if nrm < 1e-12:
                return np.eye(3)

        y_l_temp /= nrm
        # now z_l = x_l x y_l
        z_local = np.cross(x_local, y_l_temp)
        z_local /= np.linalg.norm(z_local)
        # refine y_l to ensure orthogonality
        y_local = np.cross(z_local, x_local)
        y_local /= np.linalg.norm(y_local)

        R = np.vstack([x_local, y_local, z_local])
        return R

    def transform_to_global(self, k_local, R):
        """
        Transform a 14x14 local element stiffness to global:
        K_global = T^T @ k_local @ T
        
        For the thin-walled beam, the transformation includes:
        - 3x3 rotation matrices for translations and rotations
        - Identity (1) for warping DOFs
        """
        T = np.zeros((14, 14))
        # Standard displacements and rotations
        T[0:3, 0:3] = R    # Node 1 translations
        T[3:6, 3:6] = R    # Node 1 rotations
        T[7:10, 7:10] = R  # Node 2 translations
        T[10:13, 10:13] = R  # Node 2 rotations
        
        # Warping DOFs (no transformation)
        T[6, 6] = 1.0      # Node 1 warping
        T[13, 13] = 1.0    # Node 2 warping
        
        return T.T @ k_local @ T

    def assemble(self):
        """
        Build/assemble the global stiffness matrix K
        """
        self.K[:] = 0.0  # reset

        for e in range(self.nelems):
            n1, n2 = self.elems[e]
            x1, y1, z1 = self.nodes[n1]
            x2, y2, z2 = self.nodes[n2]
            dx, dy, dz = (x2 - x1), (y2 - y1), (z2 - z1)
            L = np.sqrt(dx*dx + dy*dy + dz*dz)
            if L < 1e-14:
                continue

            # Get element properties
            E = self.E[e]
            G = self.G[e]
            A = self.A[e]
            Iy = self.Iy[e]
            Iz = self.Iz[e]
            J = self.J[e]
            Iw = self.Iw[e]
            x0 = self.x0[e]
            y0 = self.y0[e]
            r = self.r[e]

            # local stiffness matrix (thin-walled beam)
            k_loc = self.k_thin_walled_beam(
                E=E, G=G, A=A, Iy=Iy, Iz=Iz, Iw=Iw, J=J, L=L,
                x0=x0, y0=y0, r=r
            )

            # rotation matrix
            R = self.element_rotation_matrix(n1, n2)

            # transform to global coordinates
            k_g = self.transform_to_global(k_loc, R)

            # global DOFs (7 per node)
            dof = np.zeros(14, dtype=int)
            dof[0:7] = self.ndof*n1 + np.arange(7)
            dof[7:14] = self.ndof*n2 + np.arange(7)

            # for i in range(k_g.shape[0]):
            #     for j in range(k_g.shape[1]):
            #         if abs(k_g[i, j]) > 1e-10: 
            #             print(f"{i, j}  {k_g[i, j]}")

            # add to global stiffness matrix
            for i in range(14):
                for j in range(14):
                    self.K[dof[i], dof[j]] += k_g[i, j]

        # # add any boundary springs
        # for sup in self.dof_stiff:
        #     dof_id = sup['dof']
        #     k_spring = sup['stiffness']
        #     self.K[dof_id, dof_id] += k_spring

    def solve(self):
        """
        Partition the global system, solve for the free DOFs, and compute reactions.
        """
        self.assemble()

        # Extract the free DOF submatrix and force vector
        Kf = self.K[np.ix_(self.dof_free, self.dof_free)]
        ff = self.f[self.dof_free]

        # solve
        af = np.linalg.solve(Kf, ff)
        self.a[self.dof_free] = af
        print(af)

        # for i in range(Kf.shape[0]):
        #     for j in range(Kf.shape[1]):
        #         if abs(Kf[i, j]) > 1e-10: 
        #             print(f"{i, j}  {Kf[i, j]}")

        # compute reactions
        if len(self.dof_con) > 0:
            Kc = self.K[np.ix_(self.dof_con, self.dof_free)]
            self.f[self.dof_con] = Kc @ af

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

    def plot_deformed_shape(self, scale=1.0, npoints=20, figsize=(10, 8), show_warping=True):
        """
        Plot the deformed shape of the structure
        
        Args:
        scale: Scale factor for the deformation
        npoints: Number of points to use for interpolation along each element
        figsize: Figure size
        show_warping: Whether to visualise warping effects with color
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

            # Plot show_undeformed configuration
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
                # Use a colormap to visualise warping
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
        plt.gca().set_aspect('equal', adjustable='box')
        
        # Add a colorbar if showing warping
        if show_warping and max_warping > 1e-10:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(0, max_warping))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.1)
            cbar.set_label('Warping magnitude')
            
        plt.tight_layout()
        plt.show()

    # def add_point_load(self, node_id, direction, magnitude):
    #     """
    #     Add a point load at a specific node.
        
    #     Args:
    #         node_id: Node index
    #         direction: Direction index (0-6 for u, v, w, θx, θy, θz, φ)
    #         magnitude: Load magnitude
    #     """
    #     dof_id = self.ndof * node_id + direction
    #     self.f[dof_id] += magnitude

    # def add_point_displacement(self, node_id, direction, value):
    #     """
    #     Add a prescribed displacement boundary condition.
        
    #     Args:
    #         node_id: Node index
    #         direction: Direction index (0-6 for u, v, w, θx, θy, θz, φ)
    #         value: Prescribed displacement value
    #     """
    #     dof_id = self.ndof * node_id + direction
        
    #     # Add to constrained DOFs if not already there
    #     if dof_id not in self.dof_con:
    #         self.dof_con = np.append(self.dof_con, dof_id)
    #         # Update free DOFs
    #         self.dof_free = np.setdiff1d(np.arange(self.nnodes*self.ndof), self.dof_con)
        
    #     # Set the prescribed value
    #     self.a[dof_id] = value

    # def add_elastic_support(self, node_id, direction, stiffness):
    #     """
    #     Add an elastic support (spring) at a node.
        
    #     Args:
    #     node_id: Node index
    #     direction: Direction index (0-6 for u, v, w, θx, θy, θz, φ)
    #     stiffness: Spring stiffness
    #     """
    #     dof_id = self.ndof * node_id + direction
    #     self.dof_stiff.append({
    #         'dof': dof_id,
    #         'stiffness': stiffness
    #     })

class Frame3D(ThinWalledFrame3D):   
    def __init__(self, nodes, elems, E=200e9, A=0.1):
        """
        Frame object. Input node xyz locations and element connectivity array. 

        Args: 
            nodes : np.array(
                        [[x1, y1, z1], 
                        [x2, y2, z2], 
                        ... 
            elements : np.array( 
                        [[0, 1], 
                        [1, 2], 
                        ...

        """
        nelems = elems.shape[0]

        E = np.full(nelems, 1)  
        G = np.full(nelems, 1)   
        A = np.full(nelems, 1)   
        Iy = np.full(nelems, 1)  
        Iz = np.full(nelems, 1)  
        J = np.full(nelems, 1)   
        Iw = np.full(nelems, 1) 

        frame = ThinWalledFrame3D(nodes, elems, E, G, A, Iy, Iz, J, Iw)

        dof_constrained = np.array([0,1,2,3,4,5], dtype=int) 

        # dof_constrained = np.array([0,1,2,3,4,5,
        #                             7,8,9,10,11,12,
        #                             14,15,16,17,18,19,
        #                             21,22,23,24,25,26], dtype=int) 

        # dof_constrained = np.array([0,1,2,
        #                             6,7,8,
        #                             12,13,14,
        #                             18,19,20], dtype=int) 

        frame.dof_con = dof_constrained
        frame.dof_free = np.setdiff1d(np.arange(frame.nnodes*frame.ndof), frame.dof_con) 
        frame.f[7*1 + 1] = -1 
        # frame.f[7*19 + 2] = -1

        frame.solve()
        frame.plot_deformed_shape(scale=1.0, npoints=30)


if __name__ == "__main__":

    nodes = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    elems = np.array([
        [0, 1],
    ])

    # # Create nodes 
    # num_nodes = 20
    # theta = np.linspace(0, np.pi/2, num_nodes)
    # radius = 1.0
    # nodes = np.zeros((num_nodes, 3))
    # nodes[:, 0] = radius * np.cos(theta) 
    # nodes[:, 1] = radius * np.sin(theta)  

    # # Create elements 
    # elems = np.zeros((num_nodes-1, 2), dtype=int)
    # for i in range(num_nodes-1):
    #     elems[i] = [i, i+1]

    Frame3D(nodes, elems)