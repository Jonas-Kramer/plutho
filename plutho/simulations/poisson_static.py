"""Module for the simulation of time domain piezoelectric systems."""


# Third party libraries
import numpy as np
import numpy.typing as npt
import scipy.sparse as sparse
import scipy.sparse.linalg as slin

# Local libraries
from .helpers import mat_apply_dbcs
from .integrals import integral_m, integral_ku, integral_kuv, \
    integral_kve
from ..enums import SolverType
from .solver import FEMSolver
from ..mesh.mesh import Mesh

__all__ = [
    "PoissonStatic"
]

class PoissonStatic(FEMSolver):
    """Class for the simulation of time domain piezoelectric systems.

    Attributes:
        m: Sparse mass matrix.
        c: Sparse damping matrix.
        k: Sparse stiffness matrix.
    """
    # FEM matrices
    m: sparse.lil_array
    c: sparse.lil_array
    k: sparse.lil_array

    def __init__(self, simulation_name: str, mesh: Mesh):
        super().__init__(simulation_name, mesh)

        self.solver_type = SolverType.PoissonStatic

    def assemble(self):
        """Assembles the FEM matrices based on the set mesh_data and
        material_data.
        The matrices are stored in self.m, self.c, self.k.
        """
        self.material_manager.initialize_materials()
        nodes = self.mesh_data.nodes
        element_order = self.mesh_data.element_order # TODO: my version only works with element_order=1

        number_of_nodes = len(nodes)
        
        m = np.zeros((number_of_nodes, number_of_nodes), dtype=np.float64)
        f = np.zeros(number_of_nodes, dtype=np.float64)

        number_of_elements = len(self.mesh_data.elements)
        positive_element = -1
        negative_element = -1

        for element_index, element in enumerate(self.mesh_data.elements):
            node_points = self.node_points[element_index]

            def integral_right(node_points: npt.NDArray, rho_volume_charge: float, epsilon: float) -> npt.NDArray:
                """Calculates the right integral.

                Parameters: 
                    node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
                        one triangle
                    rho_volume_charge: charge per volume of this element
                    epsilon: permittivity of this element

                Returns:
                    npt.NDArray: 3-element vector for the given element.
                """
                dn = np.array([
                                [-1, 1, 0],  # d_s
                                [-1, 0, 1]   # d_t
                                ])
                jacobian = np.dot(node_points, dn.T)
                jacobian_det = np.linalg.det(jacobian)
                
                return (np.ones(shape=(3,)) / 3.0) * (rho_volume_charge / epsilon) * (jacobian_det * 0.5)

            def integral_left(node_points: npt.NDArray) -> npt.NDArray:
                """Calculates the left integral.

                Parameters:
                    node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
                        one triangle.

                Returns:
                    npt.NDArray: 3x3 M matrix for the given element.
                """
                def inner(s: float, t: float) -> npt.NDArray:
                    dn = np.array([
                                    [-1, 1, 0],  # d_s
                                    [-1, 0, 1]   # d_t
                                  ])
                    jacobian = np.dot(node_points, dn.T)
                    jacobian_det = np.linalg.det(jacobian)
                    jacobian_inv = np.linalg.inv(jacobian)

                    n = np.array([1-s-t, s, t])

                    # Since the simulation is axisymmetric it is necessary
                    # to multiply with the radius in the integral
                    # (for the theta component (azimuth))
                    r = np.dot(node_points, n)
                    radius = 0.5 * (r[0] + r[1])

                    # # Get all combinations of shape function multiplied with each other
                    # return np.outer(n, n)*r*jacobian_det
                    derivative_of_N_vec = np.dot(jacobian_inv, dn)
                    return np.dot(derivative_of_N_vec.T, derivative_of_N_vec) * (jacobian_det * 0.5) * radius

                # return quadratic_quadrature(inner, 1)
                w1 = 1/6
                w2 = 2/3
                weights = np.array([w1, w1, w1])
                points = np.array([
                    [w1, w1],
                    [w2, w1],
                    [w1, w2]
                ])

                sum = np.zeros(shape=(3,3))
                for i in range(len(weights)):
                    sum = sum + weights[i] * inner(points[i][0], points[i][1])
                return sum

            m_e = integral_left(node_points) * 2 * np.pi
            f_e = integral_right(node_points, int(element_index == positive_element) - int(element_index == negative_element), 0.1) * 2 * np.pi
            # f_e = integral_right(node_points, 1000.0, 0.1) * 2 * np.pi

            # Now assemble all element matrices
            for local_p, global_p in enumerate(element):
                for local_q, global_q in enumerate(element):
                    m[global_p, global_q] += m_e[local_p][local_q]
                f[global_p] += f_e[local_p]

        self.m = m
        self.f = f

    def simulate(self):
        """
        Simulates the system by converting the dense matrix m to a sparse CSC matrix
        and solving the linear system m * phi = f.
        """
        m = self.m
        f = self.f

        # Assume self.m is a NumPy array and convert it to a SciPy CSC sparse matrix
        if not sparse.isspmatrix_csc(m):
            m = sparse.csc_matrix(m)

        self.phi = slin.spsolve(m, f)

        return self.phi

    def plot_potential(self):
        """Plots the calculated scalar potential (phi) over the 2D triangular mesh."""
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri

        if not hasattr(self, 'phi'):
            raise ValueError("Simulation must be run using .simulate() before plotting.")

        # Extract coordinates (assuming 2D: e.g., R and Z components)
        nodes = np.array(self.mesh_data.nodes)
        r = nodes[:, 0]
        z = nodes[:, 1]
        elements = self.mesh_data.elements

        # Create a matplotlib triangulation object
        triangulation = tri.Triangulation(r, z, elements)

        # Plot the potential using a filled contour plot
        plt.figure(figsize=(8, 6))
        contour = plt.tricontourf(triangulation, self.phi, cmap='viridis', levels=20)
        
        # Add a colorbar showing the potential scale
        cbar = plt.colorbar(contour)
        cbar.set_label('Potential ($\phi$)', rotation=270, labelpad=15)

        # Draw the underlying mesh wireframe faintly
        plt.triplot(triangulation, color='black', alpha=0.15, linewidth=0.5)

        # Labels and formatting
        plt.xlabel('Radius (r)')
        plt.ylabel('Height (z)')
        plt.title(f'Static Poisson Potential Distribution: {self.simulation_name}')
        plt.gca().set_aspect('equal')
        plt.grid(True, which='both', linestyle=':', alpha=0.5)
        
        plt.show()

    def add_neumann_boundary_conditions(self, V0: float, V1: float, height: float):
        nodes = np.array(self.mesh_data.nodes)

        for node_index, coord in enumerate(nodes):
            r, z = coord[0], coord[1]
            if abs(z - height) < 1e-6:
                self.f[node_index] = V1
                self.m[node_index, :] = np.zeros(shape=(len(nodes),))
                self.m[node_index,node_index] = 1
            elif abs(z) < 1e-6:
                self.f[node_index] = V0
                self.m[node_index, :] = np.zeros(shape=(len(nodes),))
                self.m[node_index,node_index] = 1