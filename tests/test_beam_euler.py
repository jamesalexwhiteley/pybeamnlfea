# from pybeamnlfea.model.frame import Frame
# from pybeamnlfea.model.material import LinearElastic
# from pybeamnlfea.model.section import Section 
# from pybeamnlfea.model.element import ThinWalledBeamElement 
# from pybeamnlfea.model.boundary import BoundaryCondition 
# from pybeamnlfea.model.load import NodalLoad 

# from pybeamnlfea.solver.assembly import Assembler 
# from pybeamnlfea.solver.eigen import EigenSolver 
# from pybeamnlfea.postprocess.results import Results

# # Author: James Whiteley (github.com/jamesalexwhiteley) 

# # Create a beam structure 
# n = 1
# beam = Frame() 
# beam.add_nodes([[i/n, 0, 0] for i in range(n+1)])

# # Add material and section (UB127x76x13) 
# beam.add_material("material", LinearElastic(E=1, G=1)) 
# beam.add_section("section", Section(A=1, Iy=1, Iz=0.1, J=1e4, Iw=1e4, y0=0, z0=0)) 

# # Add element 
# beam.add_elements([[i, i+1] for i in range(n)], "material", "section", element_class=ThinWalledBeamElement) 

# # Add boundary conditions and loads
# beam.add_boundary_condition(0, [0, 0, 0, 0, 0, 0, 0], BoundaryCondition) 
# beam.add_boundary_condition(n, [1, 1, 1, 0, 1, 1, 0], BoundaryCondition) 
# beam.add_nodal_load(n, [-0.1, 0, -1, 0, 0, 0, 0], NodalLoad) 

# beam.solve()
# beam.show()

# assembler = Assembler(beam)
# solver = EigenSolver(num_modes=1)
# critical_factors, buckling_modes = solver.eigen_solve(assembler)

# for i, (mode, load_factor) in enumerate(zip(buckling_modes, critical_factors)):
#     print(f"Mode {i+1}: Critical load factor = {load_factor}")
#     results = Results(assembler, mode, None)
#     # print(results.get_nodal_displacements(n/2))
#     results.plot_deformed_shape(scale=1, show_undeformed=True)

# # # Solve the model TODO 
# # critical_loads, buckling_modes = beam.eigen_solve() 
# # beam.mode_shapes(buckling_modes) 

from pybeamnlfea.model.frame import Frame
from pybeamnlfea.model.material import LinearElastic
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import NodalLoad 

from pybeamnlfea.solver.assembly import Assembler 
from pybeamnlfea.solver.eigen import EigenSolver 
from pybeamnlfea.postprocess.results import Results

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.sparse import eye, csr_matrix, coo_matrix
from scipy.linalg import eigh

# Function to visualize a matrix with color map
def plot_matrix(matrix, title="Matrix Visualization"):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.show()

# Function to analyze matrix properties
def analyze_matrix(matrix, name="Matrix"):
    """Analyze properties of a matrix to help with debugging."""
    # Convert to numpy array if sparse
    if hasattr(matrix, 'toarray'):
        matrix_array = matrix.toarray()
    else:
        matrix_array = np.array(matrix)
    
    # Basic properties
    print(f"\n{name} Analysis:")
    print(f"Shape: {matrix_array.shape}")
    print(f"Rank: {np.linalg.matrix_rank(matrix_array)}")
    print(f"Norm: {np.linalg.norm(matrix_array)}")
    print(f"Min value: {np.min(matrix_array)}")
    print(f"Max value: {np.max(matrix_array)}")
    
    # Check for symmetry
    is_symmetric = np.allclose(matrix_array, matrix_array.T)
    print(f"Symmetric: {is_symmetric}")
    
    # Check positive definiteness (if symmetric)
    if is_symmetric:
        try:
            if np.linalg.matrix_rank(matrix_array) == matrix_array.shape[0]:
                eigenvalues = np.linalg.eigvals(matrix_array)
                is_positive_definite = np.all(eigenvalues > 0)
                is_positive_semi = np.all(eigenvalues >= -1e-10)
                print(f"Positive definite: {is_positive_definite}")
                print(f"Positive semi-definite: {is_positive_semi}")
                print(f"Smallest eigenvalue: {np.min(eigenvalues)}")
                print(f"Largest eigenvalue: {np.max(eigenvalues)}")
                print(f"Condition number: {np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))}")
            else:
                print("Matrix is singular (rank deficient)")
        except np.linalg.LinAlgError:
            print("Could not compute eigenvalues - may be ill-conditioned")
    
    # Check for singularity
    det = np.linalg.det(matrix_array)
    print(f"Determinant: {det}")
    
    # Check sparsity
    non_zeros = np.count_nonzero(matrix_array)
    total_elements = matrix_array.size
    sparsity = 100 * (1 - non_zeros / total_elements)
    print(f"Sparsity: {sparsity:.2f}% ({non_zeros} non-zeros out of {total_elements})")
    
    # Return for further analysis
    return matrix_array

# Create a beam structure 
n = 1  # Start with a single element for debugging
beam = Frame() 
beam.add_nodes([[i/n, 0, 0] for i in range(n+1)])

# Add material and section
beam.add_material("material", LinearElastic(E=1, G=1)) 
beam.add_section("section", Section(A=1, Iy=1, Iz=0.1, J=1e4, Iw=1e4, y0=0, z0=0)) 

# Add element 
beam.add_elements([[i, i+1] for i in range(n)], "material", "section", element_class=ThinWalledBeamElement) 

# Try different boundary conditions (start with cantilever which is simple)
beam.add_boundary_condition(0, [0, 0, 0, 0, 0, 0, 0], BoundaryCondition)  # Fixed
beam.add_boundary_condition(n, [1, 1, 1, 1, 1, 1, 1], BoundaryCondition)  # Free

# Add a combination of axial and lateral load
beam.add_nodal_load(n, [-1, 0, 0, 0, 0, 0, 0], NodalLoad)

# Solve linear problem to get displacements
print("\nSolving linear problem...")
beam.solve()

# Get assembler
assembler = Assembler(beam)

# Debug DOF mapping
print("\nDOF Mapping:")
for (node_id, dof_idx), global_dof in assembler.dof_map.items():
    if global_dof >= 0:  # Only show unconstrained DOFs
        dof_name = ["ux", "uy", "uz", "θx", "θy", "θz", "warping"][dof_idx]
        print(f"Node {node_id}, DOF {dof_idx} ({dof_name}) -> Global DOF {global_dof}")

# Get stiffness matrices
print("\nAssembling matrices...")
K = assembler.assemble_stiffness_matrix()
print(f"K matrix size: {K.shape}")

# Modify EigenSolver to print internal forces
class DebuggingEigenSolver(EigenSolver):
    def eigen_solve(self, assembler):
        """Override to add debugging information."""
        # Linear analysis -> internal member forces 
        nodal_displacements, _ = self.solve(assembler)
        element_internal_forces = self._calculate_element_internal_forces(assembler, nodal_displacements)
        
        # Print internal forces for debugging
        print("\nInternal forces for each element:")
        for elem_id, forces in element_internal_forces.items():
            print(f"Element {elem_id}: {forces}")
        
        # Assemble elastic stiffness matrix
        K = assembler.assemble_stiffness_matrix()
        K_array = analyze_matrix(K, "Elastic Stiffness Matrix (K)")
        
        # Assemble geometric stiffness matrix 
        K_full = assembler.assemble_stiffness_matrix(geometric_stiffness=True, element_internal_forces=element_internal_forces)
        Kg = K_full - K  
        Kg_array = analyze_matrix(Kg, "Geometric Stiffness Matrix (Kg)")
        
        # Visualize matrices
        plot_matrix(K_array, "Elastic Stiffness Matrix (K)")
        plot_matrix(Kg_array, "Geometric Stiffness Matrix (Kg)")
        
        # Check matrix conditioning
        print("\nChecking generalized eigenvalue problem conditioning...")
        try:
            # Try calculating a few eigenvalues directly using scipy.linalg.eigh
            # for a small subset of the matrix
            max_size = 10
            size = min(max_size, K_array.shape[0])
            print(f"Testing eigenvalue calculation with {size}x{size} submatrix...")
            K_sub = K_array[:size, :size]
            Kg_sub = Kg_array[:size, :size]
            
            # Regularize Kg slightly
            Kg_reg_sub = Kg_sub - 1e-10 * np.eye(size)
            
            # Solve generalized eigenvalue problem
            eigenvalues_test, _ = eigh(K_sub, Kg_reg_sub)
            print(f"Test eigenvalues: {eigenvalues_test}")
        except Exception as e:
            print(f"Test eigenvalue calculation failed: {e}")
        
        # Buckling analysis (K-λKg)Φ=0
        print("\nSolving eigenvalue problem...")
        try:
            # Regularize Kg
            Kg_reg = Kg - 1e-10 * eye(Kg.shape[0]) 
            eigenvalues, eigenvectors = eigsh(K, k=self.num_modes, M=-Kg_reg, which='LM')
            print(f"Eigenvalues: {eigenvalues}")
            critical_factors = eigenvalues
            
            # Eigenvectors -> nodal displacements (modes)
            buckling_modes = []
            for i in range(self.num_modes):
                mode_displacements = self._get_nodal_displacements(assembler, eigenvectors[:, i])
                buckling_modes.append(mode_displacements)
                
            return critical_factors, buckling_modes
            
        except Exception as e:
            print(f"Eigenvalue solution failed: {e}")
            print("Trying alternative approach...")
            
            # Try with different regularization
            try:
                Kg_reg = Kg - 1e-8 * eye(Kg.shape[0])
                eigenvalues, eigenvectors = eigsh(K, k=self.num_modes, M=-Kg_reg, which='LM')
                print(f"Alternative eigenvalues: {eigenvalues}")
                critical_factors = eigenvalues
                
                # Eigenvectors -> nodal displacements (modes)
                buckling_modes = []
                for i in range(self.num_modes):
                    mode_displacements = self._get_nodal_displacements(assembler, eigenvectors[:, i])
                    buckling_modes.append(mode_displacements)
                    
                return critical_factors, buckling_modes
            except Exception as e:
                print(f"Alternative approach also failed: {e}")
                
                # Last resort: try standard eigenvalue problem
                try:
                    print("Trying standard eigenvalue problem...")
                    from scipy.sparse.linalg import spsolve
                    K_inv = spsolve(K.tocsc(), eye(Kg.shape[0]))
                    KG_scaled = K_inv @ Kg
                    eigenvalues, eigenvectors = eigsh(KG_scaled, k=self.num_modes, which='LM')
                    print(f"Standard eigenvalues: {eigenvalues}")
                    critical_factors = 1.0 / eigenvalues
                    
                    # Eigenvectors -> nodal displacements (modes)
                    buckling_modes = []
                    for i in range(self.num_modes):
                        mode_displacements = self._get_nodal_displacements(assembler, eigenvectors[:, i])
                        buckling_modes.append(mode_displacements)
                        
                    return critical_factors, buckling_modes
                except Exception as e:
                    print(f"All eigenvalue approaches failed: {e}")
                    raise RuntimeError("Could not solve eigenvalue problem")

# Use the debugging solver
solver = DebuggingEigenSolver(num_modes=5)

try:
    critical_factors, buckling_modes = solver.eigen_solve(assembler)
    
    # Display results
    for i, (mode, load_factor) in enumerate(zip(buckling_modes, critical_factors)):
        print(f"\nMode {i+1}: Critical load factor = {load_factor}")
        results = Results(assembler, mode, None)
        results.plot_deformed_shape(scale=1, show_undeformed=True)
except Exception as e:
    print(f"Eigenvalue analysis failed: {e}")
    print("\nTroubleshooting recommendations:")
    print("1. Try with a single element first (n=1)")
    print("2. Try with simple cantilever boundary conditions")
    print("3. Apply both axial and lateral loads")
    print("4. Check that internal forces are non-zero")
    print("5. Inspect Kg matrix - it should have non-zero values related to the DOFs")
    print("6. Manually debug the EigenSolver.eigen_solve() method")