import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        self.local_axes_scale = 0.2
        
    def _initialize_plot(self, figsize=(10, 8), projection='3d'):
        """Create figure and axis if they don't exist."""
        if self.fig is None or self.ax is None:
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
        self._initialize_plot()
        
        # Extract basis vectors
        x_axis = R[0, :]  # Local x-axis in global coordinates
        y_axis = R[1, :]  # Local y-axis in global coordinates
        z_axis = R[2, :]  # Local z-axis in global coordinates
        
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
    
    def show_model_local_axes(self, scale_factor=1.0, node_labels=True):
        """Visualise the undeformed model with local axes for each element."""

        if self.model is None:
            raise ValueError("No model provided to visualiser")
            
        self._initialize_plot()
        
        # Plot elements
        for elem_id, element in self.model.elements.items():
            start_node, end_node = element.nodes
            
            # Plot element
            self.ax.plot([start_node.coords[0], end_node.coords[0]],
                      [start_node.coords[1], end_node.coords[1]],
                      [start_node.coords[2], end_node.coords[2]], 'k-', lw=1)
            
            # Plot local axes at midpoint
            midpoint = (start_node.coords + end_node.coords) / 2
            self.draw_local_axes(midpoint, element.R, scale=scale_factor*element.L)
            
            # Plot nodes
            self.ax.scatter(start_node.coords[0], start_node.coords[1], start_node.coords[2], 
                         color='k', s=30)
            self.ax.scatter(end_node.coords[0], end_node.coords[1], end_node.coords[2], 
                         color='k', s=30)
            
            # Add node labels if requested
            if node_labels:
                self.ax.text(start_node.coords[0], start_node.coords[1], start_node.coords[2], 
                          f" {start_node.id}", fontsize=10)
                self.ax.text(end_node.coords[0], end_node.coords[1], end_node.coords[2], 
                          f" {end_node.id}", fontsize=10)
        
        plt.title("Model with Local Coordinate Systems")
        plt.tight_layout()
        return self.fig, self.ax
    
    def plot_deformed_shape(self, scale=1.0, npoints=20, show_undeformed=True, 
                          show_node_id=True, show_local_axes=True, axes_points=3):
        """Plot the deformed shape of the structure using results data."""

        if self.results is None:
            raise ValueError("No results provided to visualiser")
            
        self._initialize_plot()
        
        # Call the plot_deformed_shape method from results 
        self.fig, self.ax = self.results.plot_deformed_shape(
            scale=scale, 
            npoints=npoints, 
            show_undeformed=show_undeformed,
            show_node_id=show_node_id,
            fig=self.fig,
            ax=self.ax
        )
        
        # Add local axes visualization if requested
        if show_local_axes and hasattr(self.model, 'elements'):
            self._add_deformed_local_axes(scale, npoints, axes_points)
        
        plt.tight_layout()
        return self.fig, self.ax
    
    def _add_deformed_local_axes(self, scale, npoints, axes_points):
        """Add local axes visualization to the deformed shape plot."""

        # Choose points evenly spaced along each beam
        indices = np.linspace(0, npoints, axes_points, dtype=int)
        
        for _, element in self.model.elements.items():
            # Get nodes
            start_node, end_node = element.nodes
            start_pos = start_node.coords
            end_pos = end_node.coords
            
            # Element direction and length
            element_dir = end_pos - start_pos
            L = np.linalg.norm(element_dir)
            
            # Extract local DOFs
            dof_loc = self.results.extract_local_dofs(element, element.R)
            
            for idx in indices:
                xi = idx / npoints # xi = 0.5?
                (u_xl, v_yl, w_zl, rx_xl, _) = self.results.shape_thin_walled_beam(xi, L, dof_loc)
                
                # Calculate position in deformed configuration
                base = start_pos + xi * element_dir
                disp_loc = np.array([u_xl, v_yl, w_zl])
                disp_g = element.R @ disp_loc
                deformed_pos = base + scale*disp_g
                
                # Create rotation matrix for torsion around local x-axis
                cos_t = np.cos(scale * rx_xl)
                sin_t = np.sin(scale * rx_xl)
                torsion_matrix = np.array([
                    [1, 0, 0],
                    [0, cos_t, -sin_t],
                    [0, sin_t, cos_t]
                ])
                
                # Update R: First apply torsion in local coords, then transform to global
                R_deformed = element.R @ torsion_matrix
                
                # Draw the updated local axes
                self.draw_local_axes(deformed_pos, R_deformed, scale=self.local_axes_scale)
    
    def show(self):
        """Show the current plot."""
        if self.fig is not None:
            plt.tight_layout()
            plt.axis('off') 
            plt.show()
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