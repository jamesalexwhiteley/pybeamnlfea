import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Visualiser:
    def __init__(self, model=None, results=None):
        """
        Initialise visualiser with model and optional results.

        """
        self.model = model
        self.results = results
        self.fig = None
        self.ax = None
        self.local_axes_scale = 0.3

    def initialize_plot(self, figsize=(10, 8), projection='3d'):
        """Create figure and axis if they don't exist."""
        if self.fig is None or self.ax is None:
            self._initialize_plot(figsize=figsize, projection=projection)
        
    def _initialize_plot(self, figsize=(10, 8), projection='3d'):
        """Create figure and axis if they don't exist."""
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection=projection)
        
        # Configure plot settings
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # Remove background 
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('w')
        self.ax.yaxis.pane.set_edgecolor('w')
        self.ax.zaxis.pane.set_edgecolor('w')
            
        return self.fig, self.ax
    
    def draw_local_axes(self, origin, R, scale=0.2, linewidth=2, labels=True):
        """Draw local coordinate axes at a given origin with rotation matrix R."""
        self.initialize_plot()
        
        # Extract basis vectors
        x_axis = R[0, :] # Local x-axis in global coordinates
        y_axis = R[1, :] # Local y-axis in global coordinates
        z_axis = R[2, :] # Local z-axis in global coordinates
        
        # Plot the three axes with different colors
        self.ax.quiver(origin[0], origin[1], origin[2], 
                scale*x_axis[0], scale*x_axis[1], scale*x_axis[2], 
                color='r', linewidth=linewidth, arrow_length_ratio=0.2)
        
        self.ax.quiver(origin[0], origin[1], origin[2], 
                scale*y_axis[0], scale*y_axis[1], scale*y_axis[2], 
                color='g', linewidth=linewidth, arrow_length_ratio=0.2)
        
        self.ax.quiver(origin[0], origin[1], origin[2], 
                scale*z_axis[0], scale*z_axis[1], scale*z_axis[2], 
                color='b', linewidth=linewidth, arrow_length_ratio=0.2)
        
        if labels:
            # Add labels at the end of each axis
            self.ax.text(origin[0] + scale*x_axis[0]*1.1, 
                    origin[1] + scale*x_axis[1]*1.1, 
                    origin[2] + scale*x_axis[2]*1.1, "x", color='r')
            self.ax.text(origin[0] + scale*y_axis[0]*1.1, 
                    origin[1] + scale*y_axis[1]*1.1, 
                    origin[2] + scale*y_axis[2]*1.1, "y", color='g')
            self.ax.text(origin[0] + scale*z_axis[0]*1.1, 
                    origin[1] + scale*z_axis[1]*1.1, 
                    origin[2] + scale*z_axis[2]*1.1, "z", color='b')
                
    def plot_undeformed_model(self, nodes=True, dashed=True, show_local_axes=True, node_labels=True):
        """Visualise the show_undeformed model with local axes for each element."""

        if self.model is None:
            raise ValueError("No model provided to visualiser")
            
        self.initialize_plot()
        
        # Plot elements
        for elem_id, element in self.model.elements.items():
            start_node, end_node = element.nodes
            
            # Plot element
            if dashed: 
                line_style = 'k--'
            else: 
                line_style = 'k-'
            self.ax.plot([start_node.coords[0], end_node.coords[0]],
                         [start_node.coords[1], end_node.coords[1]],
                         [start_node.coords[2], end_node.coords[2]], line_style, lw=1, alpha=0.7)
            
            # Plot local axes at midpoint
            if show_local_axes: 
                midpoint = (start_node.coords + end_node.coords) / 2
                self.draw_local_axes(midpoint, element.R, scale=self.local_axes_scale*element.L)
            
            # Plot nodes
            if nodes: 
                self.ax.scatter(start_node.coords[0], start_node.coords[1], start_node.coords[2], 
                            color='gray', s=20, alpha=0.7)
                self.ax.scatter(end_node.coords[0], end_node.coords[1], end_node.coords[2], 
                            color='gray', s=20, alpha=0.7)
            
            # Add node labels 
            if node_labels:
                self.ax.text(start_node.coords[0], start_node.coords[1], start_node.coords[2], 
                          f" {start_node.id}", fontsize=10)
                self.ax.text(end_node.coords[0], end_node.coords[1], end_node.coords[2], 
                          f" {end_node.id}", fontsize=10)

        return self.fig, self.ax

    def plot_deformed_shape(self, scale=1.0, show_undeformed=False, npoints=20, show_local_axes=False, 
                            show_cross_section=True, cross_section_scale=1.0, num_rectangles=5):
        """Visualize the model with continuous displacement coloring across elements."""

        if self.model is None or self.results is None:
            raise ValueError("Both model and results must be provided")
        
        self.initialize_plot()
        
        if show_undeformed:
            self.plot_undeformed_model(nodes=True, dashed=True, show_local_axes=False, node_labels=False)
        
        # First pass: Calculate all points and displacements across all elements
        all_deformed_points = []
        all_init_points = []
        all_torsion_angles = []
        all_element_info = []  # Store element-specific info for later use
        
        for elem_id, element in self.model.elements.items():
            # Get initial state
            init_coords = element.initial_state['coords']
            init_R = element.initial_state['R']
            L = element.initial_state['L']
            
            # Get current state for local axes visualization
            curr_coords = element.current_state['coords']
            curr_R = element.current_state['R']
            curr_L = element.current_state['L']
            
            # Apply scaling to current state coordinates
            curr_coords = [
                init_coords[i] + scale * (curr_coords[i] - init_coords[i])
                for i in range(2)
            ]
            
            # Extract local DOFs from global displacements
            local_dofs = self.results.extract_local_dofs(element, init_R)
            
            # Create points along element for shape function evaluation
            xi_values = np.linspace(0, 1, npoints)
            elem_deformed_points = []
            elem_init_points = []  # Store initial points for displacement calculation
            elem_torsion_angles = []
            
            for xi in xi_values:
                # Evaluate shape functions
                u_xl, v_yl, w_zl, rx_xl, phi_xl = self.results.shape_thin_walled_beam(xi, L, local_dofs)
                
                # Local displacement vector
                local_disp = np.array([u_xl, v_yl, w_zl])
                
                # Convert to global coordinates
                global_disp = init_R.T @ local_disp
                
                # Initial position along element (linear interpolation)
                x0 = init_coords[0] * (1-xi) + init_coords[1] * xi
                elem_init_points.append(x0)
                
                # Apply scaled displacement
                deformed_point = x0 + scale * global_disp
                elem_deformed_points.append(deformed_point)
                
                # Store torsional angle
                elem_torsion_angles.append(scale * rx_xl)
            
            # Convert to arrays
            elem_deformed_points = np.array(elem_deformed_points)
            elem_init_points = np.array(elem_init_points)
            
            # Store for global processing
            all_deformed_points.append(elem_deformed_points)
            all_init_points.append(elem_init_points)
            all_torsion_angles.append(elem_torsion_angles)
            all_element_info.append({
                'elem_id': elem_id,
                'init_R': init_R,
                'curr_coords': curr_coords,
                'curr_R': curr_R,
                'curr_L': curr_L
            })
        
        # Calculate global displacement norms for consistent colouring
        all_displacements = []
        for i in range(len(all_deformed_points)):
            displacements = np.linalg.norm(all_deformed_points[i] - all_init_points[i], axis=1)
            all_displacements.extend(displacements)
        
        # Get global min/max for normalisation
        max_displacement = np.max(all_displacements) if all_displacements else 1.0
        if max_displacement < 1e-10:
            max_displacement = 1.0  # Prevent division by zero
        
        # Second pass: Plot elements with global displacement normalisation
        for i, (elem_deformed_points, elem_init_points, elem_torsion_angles, elem_info) in enumerate(
                zip(all_deformed_points, all_init_points, all_torsion_angles, all_element_info)):
            
            # Extract element info
            elem_id = elem_info['elem_id']
            init_R = elem_info['init_R']
            curr_coords = elem_info['curr_coords']
            curr_R = elem_info['curr_R']
            curr_L = elem_info['curr_L']
            
            if not show_cross_section: 
                # Plot the deformed shape (beam centerline)
                self.ax.plot(elem_deformed_points[:, 0], elem_deformed_points[:, 1], elem_deformed_points[:, 2], 
                        'b-', lw=2)
                
                # Plot nodes
                self.ax.scatter(elem_deformed_points[0, 0], elem_deformed_points[0, 1], elem_deformed_points[0, 2], 
                            color='b', s=30)
                self.ax.scatter(elem_deformed_points[-1, 0], elem_deformed_points[-1, 1], elem_deformed_points[-1, 2], 
                            color='b', s=30)
            
            # Calculate displacements for this element
            elem_displacement_norms = np.linalg.norm(elem_deformed_points - elem_init_points, axis=1)
            
            # Show rectangular cross section with global displacement normalisation
            if show_cross_section:
                # Select points for cross-sections
                rect_indices = np.linspace(0, len(elem_deformed_points)-1, num_rectangles, dtype=int)
                rect_deformed_points = elem_deformed_points[rect_indices]
                rect_torsion_angles = np.array(elem_torsion_angles)[rect_indices]
                rect_displacements = elem_displacement_norms[rect_indices]
                
                # Normalize displacements based on global max
                normalized_displacements = rect_displacements / max_displacement
                
                # Create rectangles at selected points
                rect_corners_list = []
                
                for j, (position, angle, norm_disp) in enumerate(zip(rect_deformed_points, rect_torsion_angles, normalized_displacements)):
                    # Create rectangle vertices in the YZ plane (perpendicular to X axis)
                    half_width = 0.05 * cross_section_scale / 2
                    half_height = 0.1 * cross_section_scale / 2
                    
                    # Create rectangle corners in YZ plane (x=0 in local coordinates)
                    rect_corners_local = np.array([
                        [0, -half_width, -half_height],  # Bottom left
                        [0, half_width, -half_height],   # Bottom right
                        [0, half_width, half_height],    # Top right
                        [0, -half_width, half_height]    # Top left
                    ])
                    
                    # Apply torsional rotation (around local X-axis)
                    cos_angle = np.cos(angle)
                    sin_angle = np.sin(angle)
                    
                    # Rotate YZ coordinates
                    rect_corners_rotated = np.array([
                        [0, pt[1]*cos_angle - pt[2]*sin_angle, pt[1]*sin_angle + pt[2]*cos_angle] 
                        for pt in rect_corners_local
                    ])
                    
                    # Transform to global coordinates using the original rotation matrix
                    rect_corners_global = np.array([
                        position + init_R.T @ corner for corner in rect_corners_rotated
                    ])
                    
                    # Associate displacement value with the rectangle for coloring
                    rect_corners_list.append((rect_corners_global, norm_disp))
                
                # Create quad strips between adjacent rectangles
                for j in range(len(rect_corners_list) - 1):
                    rect1, disp1 = rect_corners_list[j]
                    rect2, disp2 = rect_corners_list[j+1]
                    
                    # Use average displacement for coloring this segment
                    avg_disp = (disp1 + disp2) / 2
                    
                    # Create four quad faces (one for each side of the prism)
                    for k in range(4):
                        k_next = (k + 1) % 4
                        quad = Poly3DCollection([[
                            rect1[k], 
                            rect1[k_next], 
                            rect2[k_next],
                            rect2[k]
                        ]], alpha=0.6)
                        
                        # Color based on displacement magnitude
                        # color = plt.cm.Blues(avg_disp)
                        color = plt.cm.RdPu(avg_disp)
                        
                        quad.set_facecolor(color)
                        quad.set_edgecolor('gray')
                        self.ax.add_collection3d(quad)
            
            # Show local axes if requested
            if show_local_axes:
                # For current state (if we scale displacements, we also need to scale local axes)
                temp_curr_R = curr_R.copy()
                if curr_L > 1e-10:
                    # Update x-axis (tangent) to match scaled direction
                    temp_curr_R[0] = (curr_coords[1] - curr_coords[0]) / curr_L
                    
                    # Ensure y and z axes remain orthogonal to new x-axis
                    y_temp = temp_curr_R[1]
                    y_local = y_temp - np.dot(y_temp, temp_curr_R[0]) * temp_curr_R[0]
                    y_local = y_local / np.linalg.norm(y_local)
                    
                    z_local = np.cross(temp_curr_R[0], y_local)
                    z_local = z_local / np.linalg.norm(z_local)
                    
                    temp_curr_R[1] = y_local
                    temp_curr_R[2] = z_local
                
                midpoint = (curr_coords[0] + curr_coords[1]) / 2
                self.draw_local_axes(midpoint, temp_curr_R, 
                                scale=self.local_axes_scale*curr_L,
                                linewidth=2)
        
        return self.fig, self.ax

    def show(self):
        """Show the current plot."""
        if self.fig is not None:
            plt.tight_layout()
            # plt.axis('off') 
            plt.axis('equal') 
            plt.show()
            self.fig, self.ax = None, None 
        else:
            print("No plot has been created yet.")
    
    def save(self, filename, dpi=300):
        """Save the current plot to a file."""
        if self.fig is not None:
            plt.tight_layout()
            plt.savefig(filename, dpi=dpi)
            print(f"Plot saved to {filename}")
        else:
            print("No plot has been created yet.")