import numpy as np 
from scipy.sparse.linalg import spsolve, eigsh, eigs
from scipy.linalg import inv
from scipy.sparse import eye, csc_matrix
from pybeamnlfea.solver.linear import LinearSolver 

# Author: James Whiteley (github.com/jamesalexwhiteley)

class EigenSolver(LinearSolver):
    def __init__(self, num_modes=5, solver_type='direct'):
        super().__init__(solver_type=solver_type)
        self.num_modes = num_modes

    def solve_eigen(self, assembler):
        """Solve linear eigenvalue problem."""
        # Linear analysis -> internal member forces 
        nodal_displacements, _ = self.solve(assembler)
        element_internal_forces = self._calculate_element_internal_forces(assembler, nodal_displacements) 
                
        # Classical approach (generalised eigenvalue problem |Km − PKg| = 0)
        try: 
            Ktot = assembler.assemble_stiffness_matrix(geometric_stiffness=True, element_internal_forces=element_internal_forces) 
            Km = assembler.assemble_stiffness_matrix(geometric_stiffness=False)
            Kg = Ktot - Km

            # print("=" * 70)
            # print("Test local stiffness matrix")
            # print("=" * 70)

            # np.set_printoptions(
            #     linewidth=200,   # increase line width
            #     precision=3,     # decimals
            #     suppress=True    # scientific notation
            # )
            # print(np.asarray(Ktot.todense()))

            # |Km − PKg| = 0 can be rearranged to |A − (1/λ)I| = 0 where A = Km^-1 @ Kg 
            A = spsolve(Km.tocsc(), (Kg + 1e-10 * eye(Kg.shape[0])).tocsc())
            eigenvalues, eigenvectors = eigs(csc_matrix(A), k=self.num_modes, which='LR')
            eigenvalues, eigenvectors = np.real(eigenvalues), np.real(eigenvectors)

            # Sort results 
            ind = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[ind]
            eigenvectors = eigenvectors[:, ind]
            critical_loads = 1 / eigenvalues

        except Exception as e: 
            print(f"EigenSolver failed with error message: {e}")
            return [], [] 
        
        # Eigenvectors -> nodal displacements (modes)
        buckling_modes = []
        for i in range(self.num_modes):
            mode_displacements = self._get_nodal_displacements(assembler, eigenvectors[:, i])
            buckling_modes.append(mode_displacements)
            
        return critical_loads, buckling_modes

# class EigenSolver(LinearSolver):
#     def __init__(self, num_modes=5, solver_type='direct'):
#         super().__init__(solver_type=solver_type)
#         self.num_modes = num_modes

#     def solve_eigen(self, assembler):
#         """
#         Solve (Ke + λ Kg) φ = 0
#         consistent with solve_ltb_eigenvalue
#         """

#         # 1. Linear analysis → internal forces
#         nodal_displacements, _ = self.solve(assembler)
#         element_internal_forces = self._calculate_element_internal_forces(
#             assembler, nodal_displacements
#         )

#         try:
#             # 2. Assemble matrices
#             Ke = assembler.assemble_stiffness_matrix(
#                 geometric_stiffness=False
#             ).tocsc()

#             Ktot = assembler.assemble_stiffness_matrix(
#                 geometric_stiffness=True,
#                 element_internal_forces=element_internal_forces
#             ).tocsc()

#             Kg = Ke - Ktot

#             # 3. Form A = Ke^{-1} Kg
#             # (small diagonal shift for numerical stability)
#             eps = 1e-10
#             A = spsolve(Ke + eps * eye(Ke.shape[0]), Kg)

#             # 4. Eigenvalue solve
#             eigenvalues_mu, eigenvectors = eigs(
#                 csc_matrix(A),
#                 k=self.num_modes * 2,     # oversample
#                 which='LM'
#             )

#             eigenvalues_mu = np.real(eigenvalues_mu)
#             eigenvectors = np.real(eigenvectors)

#             # 5. Filter negative μ → positive λ
#             neg_mask = eigenvalues_mu < -1e-12
#             mu_neg = eigenvalues_mu[neg_mask]
#             vecs_neg = eigenvectors[:, neg_mask]

#             if len(mu_neg) == 0:
#                 raise RuntimeError("No compressive buckling modes found")

#             lambdas = -1.0 / mu_neg

#             # 6. Sort by smallest positive λ (critical load)
#             idx = np.argsort(lambdas)
#             lambdas = lambdas[idx][:self.num_modes]
#             vecs = vecs_neg[:, idx][:, :self.num_modes]

#         except Exception as e:
#             print(f"EigenSolver failed with error message: {e}")
#             return [], []

#         # 7. Eigenvectors → nodal displacement fields
#         buckling_modes = []
#         for i in range(len(lambdas)):
#             mode_disp = self._get_nodal_displacements(
#                 assembler, vecs[:, i]
#             )
#             buckling_modes.append(mode_disp)

#         return lambdas, buckling_modes
    
    def _calculate_element_internal_forces(self, assembler, nodal_displacements):                
        """Calculate internal forces for each element based on global displacements."""
        element_forces = {}
        
        for element_id, element in assembler.frame.elements.items():
            # Element nodes 
            node_i_id = element.nodes[0].id
            node_j_id = element.nodes[1].id
            
            # Extract nodal displacements
            ndof_i = assembler.frame.nodes[node_i_id].ndof
            ndof_j = assembler.frame.nodes[node_j_id].ndof
            
            u_i = np.array([nodal_displacements.get((node_i_id, dof), 0.0) for dof in range(ndof_i)])
            u_j = np.array([nodal_displacements.get((node_j_id, dof), 0.0) for dof in range(ndof_j)])
            
            # Combine into element displacement vector (in global coordinates)
            u_elem_global = np.concatenate([u_i, u_j])
            
            # Transform to local coordinates
            T = element.compute_transformation_matrix()                                                    
            u_elem_local = T @ u_elem_global
            
            # Calculate internal forces
            internal_forces = element.compute_internal_forces(u_elem_local)

            element_forces[element_id] = internal_forces
            
        return element_forces