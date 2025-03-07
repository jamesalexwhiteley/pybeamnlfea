from pybeamnlfea.utils.stiffness_matrix import thin_wall_stiffness_matrix

# Author: James Whiteley (github.com/jamesalexwhiteley)

if __name__ == "__main__":
    # Parameters
    E  = 1      # Young's modulus (N/m)
    G  = 1      # Shear modulus (N/m)
    A  = 1      # Cross-sectional area (m2)
    Ix = 1      # Second moment of area about x-axis (m4)
    Iy = 1      # Second moment of area about y-axis (m4)
    Iw = 1      # Warping constant (m6)
    J  = 1      # Torsion constant (m4)
    L  = 1      # Element length (m)

    # Stiffness matrix
    K = thin_wall_stiffness_matrix(E, G, A, Ix, Iy, Iw, J, L)
    print("Element stiffness matrix:")
    print(K.todense())
    
    # Save to file
    import os
    
    def dict_to_string(sparse_dict):
        """Convert sparse dictionary to formatted string"""
        lines = []
        for (i, j), value in sparse_dict.items():
            if abs(value) > 1e-10: # Filter out near-zero values
                lines.append(f"K[{i}, {j}] = {value:.6e}")
        return "\n".join(lines)
    
    # Create output directory
    output_dir = "output_files"
    os.makedirs(output_dir, exist_ok=True)
    
    # Write to file
    file_path = os.path.join(output_dir, "stiffness_matrix.txt")
    with open(file_path, 'w') as f:
        f.write(dict_to_string(K.todok()))