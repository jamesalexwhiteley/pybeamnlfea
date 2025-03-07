import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Visualiser:
    def __init__(self, results):
        """
        Initialize the visualizer with analysis results.
        
        Args:
            results: Results object containing displacement data
        """
        self.results = results
        self.frame = results.frame
    
    def plot_deformed_shape(self, scale_factor=1.0, show_undeformed=True, num_points=20):
        """
        Plot the deformed and undeformed shape of the structure.
        
        Args:
            scale_factor: Factor to amplify displacements for better visualization
            show_undeformed: Whether to show the undeformed shape
            num_points: Number of points to use for curved elements
        """
        # Create a 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot undeformed structure
        if show_undeformed:
            self._plot_nodes_and_elements(ax, deformed=False, color='gray', alpha=0.3)
        
        # Plot deformed structure with nodes
        self._plot_nodes_and_elements(ax, deformed=True, scale_factor=scale_factor, color='blue')
        
        # Plot deformed structure with curved elements
        self._plot_curved_elements(ax, scale_factor=scale_factor, num_points=num_points, color='blue')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Structural Deformation (scale factor: {scale_factor})')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', lw=1.5, alpha=0.3, label='Undeformed'),
            Line2D([0], [0], color='blue', lw=1.5, label='Deformed')
        ]
        ax.legend(handles=legend_elements, loc='best')

        # Set equal aspect ratio for all axes
        ax.set_box_aspect([1, 1, 1]) 

        # Remove background
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Make panes transparent
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        
        plt.tight_layout()
        plt.show()
        return fig, ax
    
    def _plot_nodes_and_elements(self, ax, deformed=False, scale_factor=1.0, color='blue', alpha=1.0):
        """
        Helper method to plot the nodes and elements.
        """
        # Get node coordinates (original and deformed)
        node_coords = {}
        for node_id, node in self.frame.nodes.items():
            original_coords = node.coords
            
            if deformed:
                # Get displacements
                ux = self.results.get_nodal_displacement(node_id, 0) * scale_factor  # ux - local x
                uy_bar = self.results.get_nodal_displacement(node_id, 1) * scale_factor  # uȳ - local y
                uz_bar = self.results.get_nodal_displacement(node_id, 2) * scale_factor  # uz̄ - local z
    
                deformed_coords = original_coords + np.array([ux, uy_bar, uz_bar])
                
                node_coords[node_id] = deformed_coords
            else:
                node_coords[node_id] = original_coords
        
        # Plot nodes
        for node_id, coords in node_coords.items():
            ax.scatter(coords[0], coords[1], coords[2], s=50, color=color, alpha=alpha)
        
        # Plot elements as straight lines between nodes
        for element_id, element in self.frame.elements.items():
            # Get coordinates of element nodes
            element_nodes = [node.id for node in element.nodes]
            node_points = [node_coords[node_id] for node_id in element_nodes]
            
            # Extract x, y, z coordinates for plotting
            x_coords = [point[0] for point in node_points]
            y_coords = [point[1] for point in node_points]
            z_coords = [point[2] for point in node_points]
            
            ax.plot(x_coords, y_coords, z_coords, linewidth=1.5, color=color, alpha=alpha)
    
    def _plot_curved_elements(self, ax, scale_factor=1.0, num_points=20, color='blue', alpha=1.0):
        """
        Plot elements with their deflected curved shape.
        """
        for element_id, element in self.frame.elements.items():
            # Get deflection data in local coordinates
            c, deflection = self.results.calculate_element_deflection(element_id, num_points)
            
            # Get nodes
            start_node = element.nodes[0]
            end_node = element.nodes[1]
            
            # Get node coordinates
            start_pos = start_node.coords
            end_pos = end_node.coords
            
            # Calculate element direction and length
            element_dir = end_pos - start_pos
            element_length = np.linalg.norm(element_dir)
            
            # Create array for deformed points
            global_points = np.zeros((num_points, 3))
            
            # Get local coordinate system (rotation matrix)
            R = element.R
            
            for i in range(num_points):
                # Normalized position along element
                xi = c[i] / element_length
                
                # Base point along undeformed element
                base_point = start_pos + xi * element_dir
                
                # Local displacement vector (uy, uz, ux) at this point
                local_disp = np.array([
                    deflection[i, 0],  # ux (local x)
                    deflection[i, 1],  # uy (local y)
                    deflection[i, 2]   # uz (local z)
                ])
                
                # Transform local displacement to global coordinates
                global_disp = R.T @ local_disp  # R.T transforms from local to global

                # Apply displacement scaling
                global_disp *= scale_factor
                
                # Final position = base point + displacement
                global_points[i] = base_point + global_disp
            
            # Plot the curved element
            ax.plot(global_points[:, 0], global_points[:, 1], global_points[:, 2], 
                    linewidth=2, color=color, alpha=alpha)