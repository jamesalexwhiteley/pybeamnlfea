"""
Monosymmetric I-section properties and LTB buckling verification.
Based on Anderson & Trahair (1972) formulas for monosymmetric beams.

"""
import numpy as np

def monosymmetric_section_properties(B1, T1, B2, T2, D, t):
    """
    Compute section properties for a monosymmetric I-section.
    """
    
    # Height between flange centroids
    h = D - (T1 + T2) / 2
    
    # Areas
    A1 = B1 * T1            # Top flange
    A2 = B2 * T2            # Bottom flange
    Aw = (D - T1 - T2) * t  # Web
    A = A1 + A2 + Aw        # Total area
    
    # Centroid location (y_bar from top)
    y_bar = (B1*T1**2/2 + B2*T2*(D - T2/2) + (D - T1 - T2)*(D + T1 - T2)*t/2) / (B1*T1 + B2*T2 + (D - T1 - T2)*t)
    
    # Second moment of area about z-axis (strong axis, bending in x-y plane)
    # Iz = sum of (I_local + A*d^2) for each part
    # Top flange
    Iz1 = B1 * T1**3 / 12 + A1 * (y_bar - T1/2)**2
    # Bottom flange  
    Iz2 = B2 * T2**3 / 12 + A2 * (D - T2/2 - y_bar)**2
    # Web
    hw = D - T1 - T2
    Izw = t * hw**3 / 12 + Aw * (T1 + hw/2 - y_bar)**2
    Iz = Iz1 + Iz2 + Izw
    
    # Second moment of area about y-axis (weak axis, bending in x-z plane)
    Iy1 = T1 * B1**3 / 12
    Iy2 = T2 * B2**3 / 12
    Iyw = hw * t**3 / 12
    Iy = Iy1 + Iy2 + Iyw
    
    # St. Venant torsion constant
    J = (B1 * T1**3 + B2 * T2**3 + hw * t**3) / 3
    
    # Alpha parameter (for warping constant)
    alpha = 1 / (1 + (B1/B2)**3 * (T1/T2))
    
    # Shear center location (y0 below centroid, positive downward)
    # Shear center is at distance alpha*h from top flange centroid
    y_shear_center_from_top = T1/2 + alpha * h
    y0 = alpha * h - y_bar + T1/2  # y0 = distance from centroid to shear center (+ if SC below C)
    
    # Warping constant
    Iw = alpha * B1**3 * T1 * h**2 / 12
    
    # beta_z (monosymmetry parameter) 
    # This is Wagner's coefficient for monosymmetric sections
    # β_x = (1/Ix) * [∫y(x² + y²)dA] - 2*y0
    
    # Distance from centroid to top flange centroid
    d1 = y_bar - T1/2
    # Distance from centroid to bottom flange centroid  
    d2 = D - T2/2 - y_bar
    
    # beta_z simplified calculation 
    # β_x = (1/Ix) * [(h-ȳ) * (B2³T2/12 + B2T2(h-ȳ)² + (h-ȳ)³t/4)
    #                 - ȳ * (B1³T1/12 + B1T1ȳ² + ȳ³t/4)] - 2y0
    
    # Here ȳ is measured from the shear center, but let's use a cleaner approach
    # For a monosymmetric I-section:
    
    # Contribution from flanges (dominant terms)
    term1 = (h - y_bar + T1/2) * (B2**3 * T2 / 12 + B2 * T2 * (h - y_bar + T1/2)**2)
    term2 = (y_bar - T1/2) * (B1**3 * T1 / 12 + B1 * T1 * (y_bar - T1/2)**2)
    
    # Web contribution (approximate)
    term_web = 0  # web contribution neglected for thin webs
    
    beta_z = (1/Iy) * (term1 - term2 + term_web) - 2*y0
    
    y_bar_from_SC = y_bar - (T1/2 + alpha*h)  # ȳ in image notation (from shear center)
    
    # More accurate beta_z using Anderson & Trahair's approach
    # For monosymmetric I-beams, a good approximation is:
    # beta_z_approx = 0.9 * h * (2 * (Iy1/Iy) - 1) * (1 - (Iy/Iz)**2)
    
    # β_x = (1/Iy) * ∫∫ y(x² + y²) dA - 2*y0  (where y is from centroid)
    
    # For I-section with thin flanges:
    # Top flange contribution: -d1 * (B1³T1/12 + A1*d1²)  [negative because above centroid]
    # Bottom flange contribution: +d2 * (B2³T2/12 + A2*d2²)
    
    Iy_top = B1**3 * T1 / 12
    Iy_bot = B2**3 * T2 / 12
    
    integral_top = -d1 * (Iy_top + A1 * d1**2)
    integral_bot = d2 * (Iy_bot + A2 * d2**2)
    
    beta_z = (integral_top + integral_bot) / Iy - 2 * y0
    
    return {
        'A': A,
        'Iy': Iy,      # Weak axis (lateral)
        'Iz': Iz,      # Strong axis  
        'J': J,
        'Iw': Iw,
        'y_bar': y_bar,
        'y0': y0,      # Shear center offset from centroid
        'beta_z': beta_z,
        'alpha': alpha,
        'h': h
    }

if __name__ == "__main__":
    print("=" * 70)
    print("MONOSYMMETRIC SECTION PROPERTY CALCULATOR")
    print("=" * 70)

    def compute_delta(beta_z, L, E, Iy, G, J):
        """Compute the monosymmetry parameter delta."""
        return (beta_z / L) * np.sqrt(E * Iy / (G * J))


    def compute_K(L, E, Iw, G, J):
        """Compute the torsion parameter K."""
        return np.sqrt(np.pi**2 * E * Iw / (G * J * L**2))
    
    # Example 1: Symmetric I-section (delta = 0)
    print("\n--- Example 1: Symmetric I-section ---")
    B1 = 0.1    # 100mm top flange
    T1 = 0.01   # 10mm thick
    B2 = 0.1    # 100mm bottom flange (same as top)
    T2 = 0.01   # 10mm thick
    D = 0.2     # 200mm total depth
    t = 0.006   # 6mm web
    
    props = monosymmetric_section_properties(B1, T1, B2, T2, D, t)
    print(f"A = {props['A']*1e6:.2f} mm²")
    print(f"Iy = {props['Iy']*1e12:.2f} mm⁴ (weak axis)")
    print(f"Iz = {props['Iz']*1e12:.2f} mm⁴ (strong axis)")
    print(f"J = {props['J']*1e12:.4f} mm⁴")
    print(f"Iw = {props['Iw']*1e18:.2f} mm⁶")
    print(f"y0 = {props['y0']*1e3:.4f} mm (shear center offset)")
    print(f"beta_z = {props['beta_z']*1e3:.4f} mm")
    
    E = 210e9
    G = 80e9
    L = 3.0

    K = compute_K(L, E, props['Iw'], G, props['J'])
    delta = compute_delta(props['beta_z'], L, E, props['Iy'], G, props['J'])
    print(f"\nK = {K:.4f}")
    print(f"delta = {delta:.4f}")
    
    # Example 2: Monosymmetric I-section (delta ≠ 0)
    print("\n--- Example 2: Monosymmetric I-section (larger top flange) ---")
    B1 = 0.15   # 150mm top flange
    T1 = 0.012  # 12mm thick
    B2 = 0.08   # 80mm bottom flange
    T2 = 0.008  # 8mm thick
    D = 0.25    # 250mm total depth
    t = 0.006   # 6mm web
    
    props = monosymmetric_section_properties(B1, T1, B2, T2, D, t)
    print(f"A = {props['A']*1e6:.2f} mm²")
    print(f"Iy = {props['Iy']*1e12:.2f} mm⁴ (weak axis)")
    print(f"Iz = {props['Iz']*1e12:.2f} mm⁴ (strong axis)")
    print(f"J = {props['J']*1e12:.4f} mm⁴")
    print(f"Iw = {props['Iw']*1e18:.2f} mm⁶")
    print(f"y0 = {props['y0']*1e3:.4f} mm (shear center offset)")
    print(f"beta_z = {props['beta_z']*1e3:.4f} mm")
    print(f"alpha = {props['alpha']:.4f}")

    K = compute_K(L, E, props['Iw'], G, props['J'])
    delta = compute_delta(props['beta_z'], L, E, props['Iy'], G, props['J'])
    print(f"\nK = {K:.4f}")
    print(f"delta = {delta:.4f}")
    
    # Example 3: Monosymmetric I-section (larger bottom flange)
    print("\n--- Example 3: Monosymmetric I-section (larger bottom flange) ---")
    B1 = 0.08   # 80mm top flange
    T1 = 0.008  # 8mm thick
    B2 = 0.15   # 150mm bottom flange
    T2 = 0.012  # 12mm thick
    D = 0.25    # 250mm total depth
    t = 0.006   # 6mm web
    
    props = monosymmetric_section_properties(B1, T1, B2, T2, D, t)
    print(f"A = {props['A']*1e6:.2f} mm²")
    print(f"Iy = {props['Iy']*1e12:.2f} mm⁴ (weak axis)")
    print(f"Iz = {props['Iz']*1e12:.2f} mm⁴ (strong axis)")
    print(f"J = {props['J']*1e12:.4f} mm⁴")
    print(f"Iw = {props['Iw']*1e18:.2f} mm⁶")
    print(f"y0 = {props['y0']*1e3:.4f} mm (shear center offset)")
    print(f"beta_z = {props['beta_z']*1e3:.4f} mm")
    print(f"alpha = {props['alpha']:.4f}")

    K = compute_K(L, E, props['Iw'], G, props['J'])
    delta = compute_delta(props['beta_z'], L, E, props['Iy'], G, props['J'])
    print(f"\nK = {K:.4f}")
    print(f"delta = {delta:.4f}")
