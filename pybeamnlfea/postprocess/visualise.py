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
    
    # def plot_deformed_shape(self, scale_factor=1.0, show_undeformed=True, num_points=20):
    #     """
    #     Plot the deformed and undeformed shape of the structure.
        
    #     Args:
    #         scale_factor: Factor to amplify displacements for better visualization
    #         show_undeformed: Whether to show the undeformed shape
    #         num_points: Number of points to use for curved elements
    #     """
    #     # Create a 3D plot
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')
        
    #     # Plot undeformed structure
    #     if show_undeformed:
    #         self._plot_nodes_and_elements(ax, deformed=False, color='gray', alpha=0.3)
        
    #     # Plot deformed structure with nodes
    #     self._plot_nodes_and_elements(ax, deformed=True, scale_factor=scale_factor, color='blue')
        
    #     # Plot deformed structure with curved elements
    #     self._plot_curved_elements(ax, scale_factor=scale_factor, num_points=num_points, color='blue')
        
    #     # Set labels and title
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.set_title(f'Structural Deformation (scale factor: {scale_factor})')
        
    #     # Add legend
    #     from matplotlib.lines import Line2D
    #     legend_elements = [
    #         Line2D([0], [0], color='gray', lw=1.5, alpha=0.3, label='Undeformed'),
    #         Line2D([0], [0], color='blue', lw=1.5, label='Deformed')
    #     ]
    #     ax.legend(handles=legend_elements, loc='best')
        
    #     plt.tight_layout()
    #     plt.show()
    #     return fig, ax
    
    # def _plot_nodes_and_elements(self, ax, deformed=False, scale_factor=1.0, color='blue', alpha=1.0):
    #     """
    #     Helper method to plot the nodes and elements.
    #     """
    #     # Get node coordinates (original and deformed)
    #     node_coords = {}
    #     for node_id, node in self.frame.nodes.items():
    #         original_coords = node.coords
            
    #         if deformed:
    #             # Get displacements
    #             w = self.results.get_nodal_displacement(node_id, 0) * scale_factor  # w - local z
    #             u_bar = self.results.get_nodal_displacement(node_id, 1) * scale_factor  # ū - local x
    #             v_bar = self.results.get_nodal_displacement(node_id, 2) * scale_factor  # v̄ - local y
    
    #             deformed_coords = original_coords + np.array([u_bar, v_bar, w])
                
    #             node_coords[node_id] = deformed_coords
    #         else:
    #             node_coords[node_id] = original_coords
        
    #     # Plot nodes
    #     for node_id, coords in node_coords.items():
    #         ax.scatter(coords[0], coords[1], coords[2], s=50, color=color, alpha=alpha)
        
    #     # Plot elements as straight lines between nodes
    #     for element_id, element in self.frame.elements.items():
    #         # Get coordinates of element nodes
    #         element_nodes = [node.id for node in element.nodes]
    #         node_points = [node_coords[node_id] for node_id in element_nodes]
            
    #         # Extract x, y, z coordinates for plotting
    #         x_coords = [point[0] for point in node_points]
    #         y_coords = [point[1] for point in node_points]
    #         z_coords = [point[2] for point in node_points]
            
    #         ax.plot(x_coords, y_coords, z_coords, linewidth=1.5, color=color, alpha=alpha)
    
    # def _plot_curved_elements(self, ax, scale_factor=1.0, num_points=20, color='blue', alpha=1.0):
    #     """
    #     Plot elements with their deflected curved shape.
    #     """
    #     for element_id, element in self.frame.elements.items():
    #         # Get deflection data in local coordinates
    #         c, deflection = self.results.calculate_element_deflection(element_id, num_points)
            
    #         # Get nodes
    #         start_node = element.nodes[0]
    #         end_node = element.nodes[1]
            
    #         # Get node coordinates
    #         start_pos = start_node.coords
    #         end_pos = end_node.coords
            
    #         # Calculate element direction and length
    #         element_dir = end_pos - start_pos
    #         element_length = np.linalg.norm(element_dir)
            
    #         # Create array for deformed points
    #         global_points = np.zeros((num_points, 3))
            
    #         # Get local coordinate system (rotation matrix)
    #         R = element.R
            
    #         for i in range(num_points):
    #             # Normalized position along element
    #             xi = c[i] / element_length
                
    #             # Base point along undeformed element
    #             base_point = start_pos + xi * element_dir
                
    #             # Local displacement vector (u, v, w) at this point
    #             local_disp = np.array([
    #                 deflection[i, 1],  # u (local x)
    #                 deflection[i, 2],  # v (local y)
    #                 deflection[i, 0]   # w (local z)
    #             ])
                
    #             # Transform local displacement to global coordinates
    #             global_disp = R.T @ local_disp  # R.T transforms from local to global

    #             # Apply displacement scaling
    #             global_disp *= scale_factor
                
    #             # Final position = base point + displacement
    #             global_points[i] = base_point + global_disp
            
    #         # Plot the curved element
    #         ax.plot(global_points[:, 0], global_points[:, 1], global_points[:, 2], 
    #                 linewidth=2, color=color, alpha=alpha)
                
    def plot_deformed_shape(self, scale_factor=1.0, show_undeformed=True, num_points=20):
        """
        Plot the deformed shape with both linear elements (straight lines) and 
        curved elements (using shape functions)
        """
        # Create a 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot undeformed structure if requested
        if show_undeformed:
            self._plot_nodes_and_elements(ax, deformed=False, color='gray', alpha=0.3)
        
        # Plot linearly deformed structure (straight lines between nodes)
        self._plot_nodes_and_elements(ax, deformed=True, scale_factor=scale_factor, color='red', alpha=0.7)
        
        # Plot curved deformed structure (using shape functions)
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
            Line2D([0], [0], color='red', lw=1.5, alpha=0.7, label='Linear Deformed'),
            Line2D([0], [0], color='blue', lw=1.5, label='Curved Deformed')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        # Set equal aspect ratio
        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])
        center = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        ax.set_xlim3d([center[0] - radius, center[0] + radius])
        ax.set_ylim3d([center[1] - radius, center[1] + radius])
        ax.set_zlim3d([center[2] - radius, center[2] + radius])
        
        plt.tight_layout()
        plt.show()
        return fig, ax

    def _plot_nodes_and_elements(self, ax, deformed=False, scale_factor=1.0, color='blue', alpha=1.0):
        """
        Helper method to plot the nodes and elements as straight lines.
        """
        # Get node coordinates (original and deformed)
        node_coords = {}
        for node_id, node in self.frame.nodes.items():
            original_coords = node.coords
            
            if deformed:
                # Get displacements in global coordinates
                # Note: These are already in global coordinates, so no transformation needed
                w = self.results.get_nodal_displacement(node_id, 0) * scale_factor
                u_bar = self.results.get_nodal_displacement(node_id, 1) * scale_factor
                v_bar = self.results.get_nodal_displacement(node_id, 2) * scale_factor

                deformed_coords = original_coords + np.array([u_bar, v_bar, w])
                
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
            n1, n2 = element.nodes
            p1 = n1.coords
            p2 = n2.coords
            
            # Element vector and length
            element_dir = p2 - p1
            L = np.linalg.norm(element_dir)
            
            # Get local coordinate system matrix
            R = element.R
            
            # Create array for deformed points
            deformed_points = np.zeros((num_points, 3))
            
            for i in range(num_points):
                # Base point along undeformed element
                xi = c[i] / L
                base_point = p1 + xi * element_dir
                
                # Local displacement vector (u, v, w)
                local_disp = np.array([
                    -deflection[i, 1],  # u (local x) - negated to match working solution
                    -deflection[i, 2],  # v (local y) - negated to match working solution
                    deflection[i, 0]    # w (local z)
                ])
                
                # Transform to global coordinates
                global_disp = R.T @ local_disp
                
                # Apply scale factor
                global_disp *= scale_factor
                
                # Final position
                deformed_points[i] = base_point + global_disp
            
            # Plot the curved element
            ax.plot(deformed_points[:, 0], deformed_points[:, 1], deformed_points[:, 2], 
                    linewidth=2, color=color, alpha=alpha)
                