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

    # def plot_deformed_shape(self, scale=1.0, show_undeformed=False, show_previous=False, 
    #                         show_local_axes=False, node_labels=True, npoints=20):
    #     """Visualize the model with deformation between previous and current states."""

    #     if self.model is None:
    #         raise ValueError("No model provided to visualiser")
        
    #     self.initialize_plot()
        
    #     if show_undeformed:
    #         self.plot_undeformed_model(nodes=True, dashed=True, show_local_axes=False, node_labels=False)
        
    #     # Plot elements
    #     for elem_id, element in self.model.elements.items():
            
    #         # Initial state 
    #         init_coords = element.initial_state['coords']

    #         # Current step
    #         prev_coords = element.previous_state['coords']
    #         prev_R = element.previous_state['R']
    #         prev_L = element.previous_state['L']
            
    #         curr_coords = element.current_state['coords']
    #         curr_R = element.current_state['R'] 
    #         curr_L = element.current_state['L']

    #         # Scale deformations 
    #         prev_coords = [
    #             init_coords[i] + scale * (prev_coords[i] - init_coords[i]) 
    #             for i in range(2)
    #         ]
            
    #         curr_coords = [
    #             init_coords[i] + scale * (curr_coords[i] - init_coords[i])
    #             for i in range(2)
    #         ]
            
    #         # For previous state
    #         prev_L = np.linalg.norm(prev_coords[1] - prev_coords[0])
    #         prev_R = prev_R.copy()
    #         if prev_L > 1e-10:
    #             # Update x-axis (tangent) to match scaled direction
    #             prev_R[0] = (prev_coords[1] - prev_coords[0]) / prev_L
                
    #             # Ensure y and z axes remain orthogonal to new x-axis
    #             y_temp = prev_R[1]
    #             y_local = y_temp - np.dot(y_temp, prev_R[0]) * prev_R[0]
    #             y_local = y_local / np.linalg.norm(y_local)
                
    #             z_local = np.cross(prev_R[0], y_local)
    #             z_local = z_local / np.linalg.norm(z_local)
                
    #             prev_R[1] = y_local
    #             prev_R[2] = z_local
            
    #         # For current state
    #         curr_L = np.linalg.norm(curr_coords[1] - curr_coords[0])
    #         curr_R = curr_R.copy()
    #         if curr_L > 1e-10:
    #             # Update x-axis (tangent) to match scaled direction
    #             curr_R[0] = (curr_coords[1] - curr_coords[0]) / curr_L
                
    #             # Ensure y and z axes remain orthogonal to new x-axis
    #             y_temp = curr_R[1]
    #             y_local = y_temp - np.dot(y_temp, curr_R[0]) * curr_R[0]
    #             y_local = y_local / np.linalg.norm(y_local)
                
    #             z_local = np.cross(curr_R[0], y_local)
    #             z_local = z_local / np.linalg.norm(z_local)
                
    #             curr_R[1] = y_local
    #             curr_R[2] = z_local

    #         # Plot previous state if requested
    #         if show_previous:
    #             # Plot previous state (dotted)
    #             self.ax.plot([prev_coords[0][0], prev_coords[1][0]],
    #                     [prev_coords[0][1], prev_coords[1][1]],
    #                     [prev_coords[0][2], prev_coords[1][2]],
    #                     'g--', lw=1.5, alpha=0.7)
                
    #             # Show nodes in previous state
    #             self.ax.scatter(prev_coords[0][0], prev_coords[0][1], prev_coords[0][2], 
    #                         color='g', s=25, alpha=0.7)
    #             self.ax.scatter(prev_coords[1][0], prev_coords[1][1], prev_coords[1][2], 
    #                         color='g', s=25, alpha=0.7)
                
    #             # Show local axes at previous state if requested
    #             if show_local_axes:
    #                 midpoint = (prev_coords[0] + prev_coords[1]) / 2
    #                 self.draw_local_axes(midpoint, prev_R, 
    #                                 scale=self.local_axes_scale*prev_L,
    #                                 linewidth=1, labels=False)
            
    #         # Plot current state (solid)
    #         self.ax.plot([curr_coords[0][0], curr_coords[1][0]],
    #                 [curr_coords[0][1], curr_coords[1][1]],
    #                 [curr_coords[0][2], curr_coords[1][2]],
    #                 'b-', lw=2)
            
    #         # Show nodes in current state
    #         self.ax.scatter(curr_coords[0][0], curr_coords[0][1], curr_coords[0][2], 
    #                     color='b', s=30)
    #         self.ax.scatter(curr_coords[1][0], curr_coords[1][1], curr_coords[1][2], 
    #                     color='b', s=30)
            
    #         # Show local axes for current state
    #         if show_local_axes:
    #             midpoint = (curr_coords[0] + curr_coords[1]) / 2
    #             self.draw_local_axes(midpoint, curr_R, 
    #                             scale=self.local_axes_scale*curr_L,
    #                             linewidth=2)
            
    #         # Add node labels if requested
    #         if node_labels:
    #             self.ax.text(curr_coords[0][0], curr_coords[0][1], curr_coords[0][2], 
    #                     f" {element.nodes[0].id}", fontsize=10)
    #             self.ax.text(curr_coords[1][0], curr_coords[1][1], curr_coords[1][2], 
    #                     f" {element.nodes[1].id}", fontsize=10)
        
    #     return self.fig, self.ax

    def plot_deformed_shape(self, scale=1.0, show_undeformed=False, show_previous=False, 
                            show_local_axes=False, node_labels=True, npoints=20):
        """Visualize the model with deformation between previous and current states."""

        if self.model is None:
            raise ValueError("No model provided to visualiser")
        
        self.initialize_plot()
        
        if show_undeformed:
            self.plot_undeformed_model(nodes=True, dashed=True, show_local_axes=False, node_labels=False)
        
        # Plot elements
        for elem_id, element in self.model.elements.items():
            
            # Initial state 
            init_coords = element.initial_state['coords']

            # Current step
            prev_coords = element.previous_state['coords']
            prev_R = element.previous_state['R']
            prev_L = element.previous_state['L']
            
            curr_coords = element.current_state['coords']
            curr_R = element.current_state['R'] 
            curr_L = element.current_state['L']

            # Debug - print the actual displacements being used
            print(f"Element {elem_id}:")
            print(f"  Initial coords: {init_coords}")
            print(f"  Current coords: {curr_coords}")
            print(f"  Displacement 1: {np.array(curr_coords[0]) - np.array(init_coords[0])}")
            print(f"  Displacement 2: {np.array(curr_coords[1]) - np.array(init_coords[1])}")
            
            # Scale deformations 
            scaled_coords = [
                init_coords[i] + scale * (curr_coords[i] - init_coords[i]) 
                for i in range(2)
            ]
            
            # Plot the scaled deformed element
            self.ax.plot([scaled_coords[0][0], scaled_coords[1][0]],
                    [scaled_coords[0][1], scaled_coords[1][1]],
                    [scaled_coords[0][2], scaled_coords[1][2]],
                    'b-', lw=2)
            
            # Show nodes
            self.ax.scatter(scaled_coords[0][0], scaled_coords[0][1], scaled_coords[0][2], 
                        color='b', s=30)
            self.ax.scatter(scaled_coords[1][0], scaled_coords[1][1], scaled_coords[1][2], 
                        color='b', s=30)
            
            # Add node labels if requested
            if node_labels:
                self.ax.text(scaled_coords[0][0], scaled_coords[0][1], scaled_coords[0][2], 
                        f" {element.nodes[0].id}", fontsize=10)
                self.ax.text(scaled_coords[1][0], scaled_coords[1][1], scaled_coords[1][2], 
                        f" {element.nodes[1].id}", fontsize=10)
        
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