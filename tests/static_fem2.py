import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg

# Author: James Whiteley (github.com/jamesalexwhiteley)
# Modified to include thin-walled beam effects with warping
# Further modified to include Euler buckling analysis

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
        self.Kg = np.zeros((self.nnodes * self.ndof, self.nnodes * self.ndof))  # Geometric stiffness matrix
        self.f = np.zeros(self.nnodes * self.ndof)
        self.a = np.zeros(self.nnodes * self.ndof)

        # DOFs by default are all free
        self.dof_free = np.arange(self.nnodes * self.ndof, dtype=int)
        self.dof_con = np.array([], dtype=int)
        self.dof_stiff = []  # finite stiffness supports
        
        # For buckling analysis
        self.buckling_modes = None
        self.buckling_loads = None

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

    def kg_thin_walled_beam(self, P, L):
        """
        Geometric stiffness matrix for a 3D thin-walled beam element.
        This matrix represents the effect of axial load on bending stiffness.
        
        Args:
            P: Axial force (positive for tension, negative for compression)
            L: Element length
            
        Returns:
            kg: 14x14 geometric stiffness matrix
        """
        kg = np.zeros((14, 14))
        
        # Helper function to set symmetric entries
        def set_symmetric(i, j, value):
            kg[i, j] = value
            if i != j:
                kg[j, i] = value
        
        # Terms for lateral displacement v (y-direction)
        set_symmetric(1, 1, 6/5)
        set_symmetric(1, 5, L/10)
        set_symmetric(1, 8, -6/5)
        set_symmetric(1, 12, L/10)
        
        set_symmetric(5, 5, 2*L*L/15)
        set_symmetric(5, 8, -L/10)
        set_symmetric(5, 12, -L*L/30)
        
        set_symmetric(8, 8, 6/5)
        set_symmetric(8, 12, -L/10)
        
        set_symmetric(12, 12, 2*L*L/15)
        
        # Terms for lateral displacement w (z-direction)
        set_symmetric(2, 2, 6/5)
        set_symmetric(2, 4, -L/10)
        set_symmetric(2, 9, -6/5)
        set_symmetric(2, 11, -L/10)
        
        set_symmetric(4, 4, 2*L*L/15)
        set_symmetric(4, 9, L/10)
        set_symmetric(4, 11, -L*L/30)
        
        set_symmetric(9, 9, 6/5)
        set_symmetric(9, 11, L/10)
        
        set_symmetric(11, 11, 2*L*L/15)
        
        # Terms for torsional rotation and warping
        # Note: these are simplified; more complex models may include more terms
        set_symmetric(3, 3, 6/5)
        set_symmetric(3, 6, L/10)
        set_symmetric(3, 10, -6/5)
        set_symmetric(3, 13, L/10)
        
        set_symmetric(6, 6, 2*L*L/15)
        set_symmetric(6, 10, -L/10)
        set_symmetric(6, 13, -L*L/30)
        
        set_symmetric(10, 10, 6/5)
        set_symmetric(10, 13, -L/10)
        
        set_symmetric(13, 13, 2*L*L/15)
        
        # Scale by axial force P
        kg *= P/L
        
        return kg

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

    def assemble(self, buckling_load=1.0):
        """
        Build/assemble the global stiffness matrix K and geometric stiffness matrix Kg
        
        Args:
            buckling_load: Load factor for scaling the geometric stiffness matrix
        """
        self.K[:] = 0.0  # reset
        self.Kg[:] = 0.0  # reset

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

            # local geometric stiffness matrix (for buckling)
            kg_loc = self.kg_thin_walled_beam(P=-buckling_load, L=L)  # Negative sign for compression

            # rotation matrix
            R = self.element_rotation_matrix(n1, n2)

            # transform to global coordinates
            k_g = self.transform_to_global(k_loc, R)
            kg_g = self.transform_to_global(kg_loc, R)

            # global DOFs (7 per node)
            dof = np.zeros(14, dtype=int)
            dof[0:7] = self.ndof*n1 + np.arange(7)
            dof[7:14] = self.ndof*n2 + np.arange(7)

            # add to global stiffness matrix
            for i in range(14):
                for j in range(14):
                    self.K[dof[i], dof[j]] += k_g[i, j]
                    self.Kg[dof[i], dof[j]] += kg_g[i, j]

        # add any boundary springs
        for sup in self.dof_stiff:
            dof_id = sup['dof']
            k_spring = sup['stiffness']
            self.K[dof_id, dof_id] += k_spring

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

        # compute reactions
        if len(self.dof_con) > 0:
            Kc = self.K[np.ix_(self.dof_con, self.dof_free)]
            self.f[self.dof_con] = Kc @ af
            
    def solve_buckling(self, num_modes=3):
        """
        Solve the buckling eigenvalue problem to find critical loads and mode shapes.
        
        Args:
            num_modes: Number of buckling modes to calculate
            
        Returns:
            eigenvalues: Critical buckling loads
            eigenvectors: Corresponding buckling mode shapes
        """
        # Assemble with a unit reference load
        self.assemble(buckling_load=1.0)
        
        # Extract the free DOF submatrices
        Kf = self.K[np.ix_(self.dof_free, self.dof_free)]
        Kgf = self.Kg[np.ix_(self.dof_free, self.dof_free)]
        
        # Solve generalized eigenvalue problem: Kf·φ = λ·Kgf·φ
        try:
            # Use scipy.linalg.eigh for symmetric matrices
            eigenvalues, eigenvectors = linalg.eigh(Kf, Kgf)
            
            # Sort eigenvalues and eigenvectors
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Filter out negative or very small eigenvalues (if any)
            pos_idx = np.where(eigenvalues > 1e-10)[0]
            if len(pos_idx) == 0:
                print("Warning: No positive eigenvalues found.")
                return None, None
                
            eigenvalues = eigenvalues[pos_idx]
            eigenvectors = eigenvectors[:, pos_idx]
            
            # Take only requested number of modes
            num_available = min(num_modes, len(eigenvalues))
            eigenvalues = eigenvalues[:num_available]
            eigenvectors = eigenvectors[:, :num_available]
            
            # For buckling problem, the eigenvalue λ is P_cr/P_ref
            # Since P_ref = 1.0, eigenvalues directly represent critical loads
            critical_loads = eigenvalues
            
            # Expand eigenvectors to full DOF space
            full_modes = np.zeros((self.nnodes * self.ndof, num_available))
            for i in range(num_available):
                full_modes[self.dof_free, i] = eigenvectors[:, i]
            
            # Store results
            self.buckling_loads = critical_loads
            self.buckling_modes = full_modes
            
            return critical_loads, full_modes
            
        except np.linalg.LinAlgError:
            print("Error in eigenvalue calculation. Check model stability.")
            return None, None

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

    def plot_deformed_shape(self, scale=1.0, npoints=20, figsize=(10, 8), show_warping=True, mode_index=None):
        """
        Plot the deformed shape of the structure or a buckling mode
        
        Args:
            scale: Scale factor for the deformation
            npoints: Number of points to use for interpolation along each element
            figsize: Figure size
            show_warping: Whether to visualize warping effects with color
            mode_index: Index of buckling mode to plot (None for standard deformation)