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
    

    def plot_rectanglular_cross_sections(self, deformed_points, init_R, num_rectangles, torsion_angles, rect_width, rect_height):
        """Create rectangles at selected points. """

        rect_indices = np.linspace(0, len(deformed_points)-1, num_rectangles, dtype=int)
        rect_corners_list = []  # Store all rectangle corners for connecting later
        
        # Create rectangles at selected points
        for idx in rect_indices:
            position = deformed_points[idx]
            angle = torsion_angles[idx]
            
            # Create rectangle vertices in the YZ plane (perpendicular to X axis)
            half_width = rect_width / 2
            half_height = rect_height / 2
            
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
            
            # Store corners for later connecting
            rect_corners_list.append(rect_corners_global)
        
        # Create quad strips between adjacent rectangles
        for i in range(len(rect_corners_list) - 1):
            rect1 = rect_corners_list[i]
            rect2 = rect_corners_list[i+1]
            
            # Create four quad faces (one for each side of the prism)
            for j in range(4):
                j_next = (j + 1) % 4
                quad = Poly3DCollection([[
                    rect1[j], 
                    rect1[j_next], 
                    rect2[j_next],
                    rect2[j]
                ]], alpha=0.6)
                
                # Color based on position along beam
                color_val = i / (len(rect_corners_list) - 2)
                color = plt.cm.coolwarm(color_val)
                
                quad.set_facecolor(color)
                quad.set_edgecolor('gray')
                self.ax.add_collection3d(quad)

    def plot_deformed_shape(self, scale=1.0, show_undeformed=False, npoints=20, show_local_axes=False, 
                            show_cross_section=True, rect_width=0.05, rect_height=0.1, num_rectangles=5):
        """Visualize the model."""

        if self.model is None or self.results is None:
            raise ValueError("Both model and results must be provided")
        
        self.initialize_plot()
        
        if show_undeformed:
            self.plot_undeformed_model(nodes=True, dashed=True, show_local_axes=False, node_labels=False)
        
        # Plot elements with shape function interpolation
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
            deformed_points = []
            torsion_angles = []
            
            for xi in xi_values:
                # Evaluate shape functions
                u_xl, v_yl, w_zl, rx_xl, phi_xl = self.results.shape_thin_walled_beam(xi, L, local_dofs)
                
                # Local displacement vector
                local_disp = np.array([u_xl, v_yl, w_zl])
                
                # Convert to global coordinates
                global_disp = init_R.T @ local_disp
                
                # Initial position along element (linear interpolation)
                x0 = init_coords[0] * (1-xi) + init_coords[1] * xi
                
                # Apply scaled displacement
                deformed_point = x0 + scale * global_disp
                deformed_points.append(deformed_point)
                
                # Store torsional angle
                torsion_angles.append(scale * rx_xl)
            
            # Convert to array for plotting
            deformed_points = np.array(deformed_points)
            
            # Plot the deformed shape (beam centerline)
            self.ax.plot(deformed_points[:, 0], deformed_points[:, 1], deformed_points[:, 2], 
                    'b-', lw=2)
            
            # Plot nodes
            self.ax.scatter(deformed_points[0, 0], deformed_points[0, 1], deformed_points[0, 2], 
                        color='b', s=30)
            self.ax.scatter(deformed_points[-1, 0], deformed_points[-1, 1], deformed_points[-1, 2], 
                        color='b', s=30)
            
            # Show rectangular cross section 
            if show_cross_section:
                self.plot_rectanglular_cross_sections(deformed_points=deformed_points, init_R=init_R, num_rectangles=num_rectangles, 
                                                    torsion_angles=torsion_angles, rect_width=rect_width, rect_height=rect_height)
            
            # For current state (if we scale displacements, we also need to scale local axes)
            curr_L = np.linalg.norm(curr_coords[1] - curr_coords[0])
            curr_R = curr_R.copy()
            if curr_L > 1e-10:
                # Update x-axis (tangent) to match scaled direction
                curr_R[0] = (curr_coords[1] - curr_coords[0]) / curr_L
                
                # Ensure y and z axes remain orthogonal to new x-axis
                y_temp = curr_R[1]
                y_local = y_temp - np.dot(y_temp, curr_R[0]) * curr_R[0]
                y_local = y_local / np.linalg.norm(y_local)
                
                z_local = np.cross(curr_R[0], y_local)
                z_local = z_local / np.linalg.norm(z_local)
                
                curr_R[1] = y_local
                curr_R[2] = z_local

            # Show local axes at midpoint of current state if requested
            if show_local_axes:
                midpoint = (curr_coords[0] + curr_coords[1]) / 2
                self.draw_local_axes(midpoint, curr_R, 
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