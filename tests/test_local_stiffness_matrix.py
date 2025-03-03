from pybeamnlfea.utils.stiffness_matrix import thin_wall_stiffness_matrix

# Author: James Whiteley (github.com/jamesalexwhiteley)

if __name__ == "__main__":
    # Example parameters
    E = 210e9      # Young's modulus (Pa)
    G = 80e9       # Shear modulus (Pa)
    A = 0.01       # Cross-sectional area (m²)
    Ix = 1e-4      # Second moment of area about x-axis (m⁴)
    Iy = 2e-4      # Second moment of area about y-axis (m⁴)
    Iw = 1e-6      # Warping constant (m⁶)
    J = 5e-5       # Torsion constant (m⁴)
    L = 2.0        # Element length (m)
    
    # Create stiffness matrix
    K = thin_wall_stiffness_matrix(E, G, A, Ix, Iy, Iw, J, L)

    print("Element stiffness matrix:")
    print(K.todense())
    
    # Save to file
    import os
    
    def dict_to_string(sparse_dict):
        """Convert sparse dictionary to formatted string"""
        lines = []
        for (i, j), value in sparse_dict.items():
            if abs(value) > 1e-10:  # Filter out near-zero values
                lines.append(f"K[{i}, {j}] = {value:.6e}")
        return "\n".join(lines)
    
    # Create output directory
    output_dir = "output_files"
    os.makedirs(output_dir, exist_ok=True)
    
    # Write to file
    file_path = os.path.join(output_dir, "stiffness_matrix.txt")
    with open(file_path, 'w') as f:
        f.write(dict_to_string(K.todok()))