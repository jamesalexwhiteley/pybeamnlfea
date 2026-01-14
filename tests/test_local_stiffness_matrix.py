from pybeamnlfea.utils.stiffness_matrix import thin_wall_stiffness_matrix
import numpy as np

# Author: James Whiteley (github.com/jamesalexwhiteley)

if __name__ == "__main__":

    # Stiffness matrix
    Ke = thin_wall_stiffness_matrix(E=1, G=1, A=1, Iy=1, Iz=1, Iw=0, J=1, L=1, 
                                P=0, My1=0, My2=0, Mz1=0, Mz2=0,
                                Mw=0, y0=0, z0=0, beta_y=0, beta_z=0, beta_w=0, r1=0,
                                # Vy=0, Vz=0,
                                include_elastic=True, include_geometric=True)
    
    K  = thin_wall_stiffness_matrix(E=1, G=1, A=1, Iy=1, Iz=1, Iw=0, J=1, L=1,  
                                P=0, My1=1, My2=-1, Mz1=0, Mz2=0,
                                Mw=0, y0=0, z0=0, beta_y=0, beta_z=0, beta_w=0, r1=0,
                                # Vy=0, Vz=0,
                                include_elastic=True, include_geometric=True)
    Kg = K - Ke
    
    print("=" * 70)
    print("Test local stiffness matrix")
    print("=" * 70)    

    np.set_printoptions(
        linewidth=200,   # increase line width
        precision=3,     # decimals
        suppress=True    # scientific notation
    )
    print(np.asarray(K.todense()))
    
    # Save to file
    import os
    
    def dict_to_string(sparse_dict):
        """Convert sparse dictionary to formatted string"""
        lines = []
        for (i, j), value in sparse_dict.items():
            if abs(value) > 1e-10: # filter out near-zero values
                lines.append(f"K[{i+1}, {j+1}] = {value:.6e}")
        return "\n".join(lines)
    
    # Create output directory
    output_dir = "output_files"
    os.makedirs(output_dir, exist_ok=True)
    
    # Write to file
    file_path = os.path.join(output_dir, "stiffness_matrix.txt")
    with open(file_path, 'w') as f:
        f.write(dict_to_string(K.todok()))