"""
Standalone LTB Eigenvalue Analysis Test

This script validates the LTB geometric stiffness formulation against
the analytical solution for a simply-supported beam under uniform moment.

Analytical solution (Timoshenko, no warping):
    M_cr = (π/L) * sqrt(E*Iz * G*J)
    
For higher modes:
    M_cr,n = (n*π/L) * sqrt(E*Iz * G*J)
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, bmat
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh


def thin_wall_stiffness_matrix_ltb(E, G, A, Iy, Iz, Iw, J, L, 
                                    My=0, Mz=0,
                                    include_elastic=True, include_geometric=True):
    """
    Simplified stiffness matrix for LTB eigenvalue analysis.
    
    Uses moment MAGNITUDE for coupling (not gradient).
    """
    K = lil_matrix((14, 14))
    
    def add_sym(i, j, value):
        K[i-1, j-1] += value
        if i != j:
            K[j-1, i-1] += value
    
    if include_elastic:
        # Axial
        add_sym(1, 1, A*E/L)
        add_sym(1, 8, -A*E/L)
        add_sym(8, 8, A*E/L)
        
        # Bending v (lateral, DOFs 2,6,9,13)
        add_sym(2, 2, 12*E*Iz/L**3)
        add_sym(2, 6, 6*E*Iz/L**2)
        add_sym(2, 9, -12*E*Iz/L**3)
        add_sym(2, 13, 6*E*Iz/L**2)
        add_sym(6, 6, 4*E*Iz/L)
        add_sym(6, 9, -6*E*Iz/L**2)
        add_sym(6, 13, 2*E*Iz/L)
        add_sym(9, 9, 12*E*Iz/L**3)
        add_sym(9, 13, -6*E*Iz/L**2)
        add_sym(13, 13, 4*E*Iz/L)
        
        # Bending w (vertical, DOFs 3,5,10,12)
        add_sym(3, 3, 12*E*Iy/L**3)
        add_sym(3, 5, -6*E*Iy/L**2)
        add_sym(3, 10, -12*E*Iy/L**3)
        add_sym(3, 12, -6*E*Iy/L**2)
        add_sym(5, 5, 4*E*Iy/L)
        add_sym(5, 10, 6*E*Iy/L**2)
        add_sym(5, 12, 2*E*Iy/L)
        add_sym(10, 10, 12*E*Iy/L**3)
        add_sym(10, 12, 6*E*Iy/L**2)
        add_sym(12, 12, 4*E*Iy/L)
        
        # Torsion (DOFs 4,7,11,14)
        # Using exact St. Venant torsion (not the 6/5 approximation)
        add_sym(4, 4, 12*E*Iw/L**3 + G*J/L)
        add_sym(4, 7, 6*E*Iw/L**2)
        add_sym(4, 11, -12*E*Iw/L**3 - G*J/L)
        add_sym(4, 14, 6*E*Iw/L**2)
        add_sym(7, 7, 4*E*Iw/L + G*J*L/3)
        add_sym(7, 11, -6*E*Iw/L**2)
        add_sym(7, 14, 2*E*Iw/L - G*J*L/6)
        add_sym(11, 11, 12*E*Iw/L**3 + G*J/L)
        add_sym(11, 14, -6*E*Iw/L**2)
        add_sym(14, 14, 4*E*Iw/L + G*J*L/3)
    
    if include_geometric:
        # LTB coupling from My (bending about y, strong axis)
        # Couples lateral v with torsion θx
        # DOFs: v1=2, θz1=6, v2=9, θz2=13 | θx1=4, φ1=7, θx2=11, φ2=14
        
        # Using consistent shape function integration:
        # ∫ N_v'' * N_θx dx gives the coupling coefficients
        
        c1 = 6*My/(5*L)    # v-θx coupling
        c2 = My/10         # v-φ and θz-θx coupling  
        c3 = 2*My*L/15     # θz-φ diagonal coupling
        c4 = -My*L/30      # θz-φ off-diagonal
        
        # v - θx coupling
        add_sym(2, 4, c1)
        add_sym(2, 11, -c1)
        add_sym(9, 4, -c1)
        add_sym(9, 11, c1)
        
        # v - φ coupling
        add_sym(2, 7, c2)
        add_sym(2, 14, c2)
        add_sym(9, 7, -c2)
        add_sym(9, 14, -c2)
        
        # θz - θx coupling
        add_sym(6, 4, c2)
        add_sym(6, 11, -c2)
        add_sym(13, 4, c2)
        add_sym(13, 11, -c2)
        
        # θz - φ coupling
        add_sym(6, 7, c3)
        add_sym(6, 14, c4)
        add_sym(13, 7, c4)
        add_sym(13, 14, c3)
        
        # Similar for Mz (couples w with θx) - sign change
        c1z = 6*Mz/(5*L)
        c2z = Mz/10
        c3z = 2*Mz*L/15
        c4z = -Mz*L/30
        
        # w - θx coupling (note signs)
        add_sym(3, 4, -c1z)
        add_sym(3, 11, c1z)
        add_sym(10, 4, c1z)
        add_sym(10, 11, -c1z)
        
        # w - φ coupling
        add_sym(3, 7, -c2z)
        add_sym(3, 14, -c2z)
        add_sym(10, 7, c2z)
        add_sym(10, 14, c2z)
        
        # θy - θx coupling
        add_sym(5, 4, -c2z)
        add_sym(5, 11, c2z)
        add_sym(12, 4, -c2z)
        add_sym(12, 11, c2z)
        
        # θy - φ coupling
        add_sym(5, 7, -c3z)
        add_sym(5, 14, -c4z)
        add_sym(12, 7, -c4z)
        add_sym(12, 14, -c3z)
    
    return K.tocsr()


def assemble_global_matrices(n_elements, L_total, E, G, A, Iy, Iz, Iw, J,
                             My_distribution='constant', My_max=1.0):
    """
    Assemble global stiffness matrices for a beam with n elements.
    
    Args:
        n_elements: number of elements
        L_total: total beam length
        My_distribution: 'constant' or 'linear'
        My_max: maximum moment value
    """
    n_nodes = n_elements + 1
    n_dof = 7 * n_nodes
    L_elem = L_total / n_elements
    
    Ke_global = lil_matrix((n_dof, n_dof))
    Kg_global = lil_matrix((n_dof, n_dof))
    
    for elem in range(n_elements):
        # Moment at this element (for constant moment, same everywhere)
        if My_distribution == 'constant':
            My = My_max
        else:
            # Linear: varies from 0 to My_max
            x_mid = (elem + 0.5) * L_elem
            My = My_max * x_mid / L_total
        
        # Element stiffness matrices
        Ke_elem = thin_wall_stiffness_matrix_ltb(E, G, A, Iy, Iz, Iw, J, L_elem,
                                                  My=0, include_elastic=True, include_geometric=False)
        Kg_elem = thin_wall_stiffness_matrix_ltb(E, G, A, Iy, Iz, Iw, J, L_elem,
                                                  My=My, include_elastic=False, include_geometric=True)
        
        # DOF mapping
        dof_start = 7 * elem
        local_dofs = list(range(14))
        global_dofs = [dof_start + i for i in range(7)] + [dof_start + 7 + i for i in range(7)]
        
        # Assemble
        for i, gi in enumerate(global_dofs):
            for j, gj in enumerate(global_dofs):
                Ke_global[gi, gj] += Ke_elem[i, j]
                Kg_global[gi, gj] += Kg_elem[i, j]
    
    return Ke_global.tocsr(), Kg_global.tocsr()


def apply_simply_supported_bc(Ke, Kg, n_nodes):
    """
    Apply simply supported boundary conditions for LTB.
    
    Supports at both ends:
    - v = 0 (no lateral displacement)
    - w = 0 (no vertical displacement) 
    - θx = 0 (no twist)
    - u = 0 at one end (axial restraint)
    
    Free:
    - θy, θz (rotations about y and z)
    - φ (warping - unless warping is restrained)
    """
    n_dof = 7 * n_nodes
    
    # DOF indices for each node: [u, v, w, θx, θy, θz, φ]
    # Indices:                    0  1  2   3   4   5  6
    
    # Restrained DOFs
    restrained = []
    
    # Node 0 (start)
    restrained.extend([0, 1, 2, 3])  # u, v, w, θx
    
    # Node n-1 (end)  
    end_node = n_nodes - 1
    restrained.extend([7*end_node + 1, 7*end_node + 2, 7*end_node + 3])  # v, w, θx
    
    # Remove restrained DOFs
    all_dofs = list(range(n_dof))
    free_dofs = [d for d in all_dofs if d not in restrained]
    
    # Extract submatrices
    Ke_red = Ke[np.ix_(free_dofs, free_dofs)]
    Kg_red = Kg[np.ix_(free_dofs, free_dofs)]
    
    return Ke_red, Kg_red, free_dofs


def solve_ltb_eigenvalue(Ke, Kg, n_modes=5):
    """
    Solve the LTB eigenvalue problem: [Ke + λ*Kg] * φ = 0
    
    This is a generalized eigenvalue problem where we want to find λ such that
    det(Ke + λ*Kg) = 0
    
    Rearranging: Ke * φ = -λ * Kg * φ
    
    We solve this using the standard form: A * x = λ * x
    by computing Ke^{-1} * Kg * φ = -1/λ * φ
    """
    # Convert to dense for small problems
    Ke_dense = Ke.toarray() if hasattr(Ke, 'toarray') else Ke
    Kg_dense = Kg.toarray() if hasattr(Kg, 'toarray') else Kg
    
    # Method: Convert to standard eigenvalue problem
    # (Ke + λ*Kg) * φ = 0
    # Ke * φ = -λ * Kg * φ
    # Ke^{-1} * Kg * φ = -1/λ * φ
    # Let μ = -1/λ, then Ke^{-1} * Kg * φ = μ * φ
    # So λ = -1/μ
    
    # Compute Ke^{-1} * Kg
    Ke_inv = np.linalg.inv(Ke_dense)
    A = Ke_inv @ Kg_dense
    
    # Solve standard eigenvalue problem
    eigenvalues_mu, eigenvectors = np.linalg.eig(A)
    
    # Convert back: λ = -1/μ
    # We want positive λ (buckling loads), so we need negative μ
    eigenvalues_mu = np.real(eigenvalues_mu)
    eigenvectors = np.real(eigenvectors)
    
    # Filter for negative μ (which gives positive λ)
    negative_mask = eigenvalues_mu < -1e-12
    neg_mu = eigenvalues_mu[negative_mask]
    neg_eigvecs = eigenvectors[:, negative_mask]
    
    # Convert to λ = -1/μ
    lambdas = -1.0 / neg_mu
    
    # Sort by smallest positive λ (critical load)
    sort_idx = np.argsort(lambdas)
    
    n_found = min(n_modes, len(lambdas))
    if n_found < n_modes:
        print(f"  Warning: Only found {n_found} positive eigenvalues")
    
    return lambdas[sort_idx[:n_found]], neg_eigvecs[:, sort_idx[:n_found]]


def main():
    print("=" * 70)
    print("LTB Eigenvalue Analysis - Validation Test")
    print("=" * 70)
    
    # Beam properties
    L = 20.0  # m
    n_elements = 10
    
    # Material
    E = 210e9   # Pa
    G = 80e9    # Pa
    
    # Section (narrow rectangle - prone to LTB)
    b = 0.05    # m (width - weak direction)
    d = 0.5     # m (depth - strong direction)
    
    A = b * d
    Iy = b * d**3 / 12  # Strong axis
    Iz = b**3 * d / 12  # Weak axis
    J = b**3 * d / 3    # Torsion constant (approximate for thin rectangle)
    Iw = 0              # No warping for rectangle
    
    print(f"\nBeam: L = {L} m, {n_elements} elements")
    print(f"Section: {b*1000:.0f}mm x {d*1000:.0f}mm")
    print(f"  A  = {A:.6e} m²")
    print(f"  Iy = {Iy:.6e} m⁴ (strong)")
    print(f"  Iz = {Iz:.6e} m⁴ (weak)")
    print(f"  J  = {J:.6e} m⁴")
    
    # Analytical solution for simply-supported beam under uniform moment
    # M_cr = (π/L) * sqrt(E*Iz * G*J)
    M_cr_analytical = (np.pi / L) * np.sqrt(E * Iz * G * J)
    
    print(f"\nAnalytical LTB critical moment:")
    print(f"  M_cr = (π/L) * √(E·Iz·G·J)")
    print(f"  M_cr = {M_cr_analytical:.4e} Nm")
    print(f"       = {M_cr_analytical/1e6:.2f} MNm")
    
    # Assemble with unit moment (λ will be the critical moment)
    My_unit = 1.0
    Ke, Kg = assemble_global_matrices(n_elements, L, E, G, A, Iy, Iz, Iw, J,
                                       My_distribution='constant', My_max=My_unit)
    
    print(f"\nGlobal matrices assembled:")
    print(f"  Size: {Ke.shape}")
    
    # Apply boundary conditions
    n_nodes = n_elements + 1
    Ke_red, Kg_red, free_dofs = apply_simply_supported_bc(Ke, Kg, n_nodes)
    
    print(f"  After BC: {Ke_red.shape}")
    print(f"  Free DOFs: {len(free_dofs)}")
    
    # Check matrices
    print(f"\nMatrix diagnostics:")
    print(f"  Ke symmetric: {np.allclose(Ke_red.toarray(), Ke_red.toarray().T)}")
    print(f"  Kg symmetric: {np.allclose(Kg_red.toarray(), Kg_red.toarray().T)}")
    print(f"  Ke positive definite: {np.all(np.linalg.eigvalsh(Ke_red.toarray()) > 0)}")
    
    # Check for LTB coupling in Kg
    Kg_dense = Kg_red.toarray()
    print(f"  Kg has non-zero off-diagonal: {np.any(np.abs(Kg_dense - np.diag(np.diag(Kg_dense))) > 1e-12)}")
    
    # Solve eigenvalue problem
    print("\nSolving eigenvalue problem...")
    eigenvalues, eigenvectors = solve_ltb_eigenvalue(Ke_red, Kg_red, n_modes=5)
    
    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    
    for i, lam in enumerate(eigenvalues):
        M_cr_fem = lam * My_unit
        error = abs(M_cr_fem - M_cr_analytical) / M_cr_analytical * 100
        
        # Higher modes
        n = i + 1
        M_cr_analytical_n = n * (np.pi / L) * np.sqrt(E * Iz * G * J)
        error_n = abs(M_cr_fem - M_cr_analytical_n) / M_cr_analytical_n * 100
        
        print(f"\nMode {n}:")
        print(f"  FEM M_cr     = {M_cr_fem:.4e} Nm ({M_cr_fem/1e6:.2f} MNm)")
        print(f"  Analytical   = {M_cr_analytical_n:.4e} Nm ({M_cr_analytical_n/1e6:.2f} MNm)")
        print(f"  Error        = {error_n:.2f}%")
    
    return eigenvalues, M_cr_analytical


if __name__ == "__main__":
    eigenvalues, M_cr_analytical = main()