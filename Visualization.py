import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from TopologyErrors import DimensionalityError, validate_data_dimensionality
from ripser import ripser
import persim
import gudhi


class Visualization:

    @staticmethod
    def plot_data_cloud(data, color='crimson', alpha=0.7, marker='o', s=40):
        plt.scatter(
            data[:, 0], data[:, 1], color=color, alpha=alpha, marker=marker, s=s
        )
        plt.title(f'{data.shape[0]} points', fontsize=16)
        plt.axis('equal')

        plt.show()

    @staticmethod
    def plot_homologies(data, homology_boundaries, palette='gist_rainbow'):
        try:
            validate_data_dimensionality(data)
        except DimensionalityError as error:
            print(error)
            return

        plt.scatter(data[:, 0], data[:, 1], color='gray', alpha=0.5)

        cmap = plt.get_cmap(palette)
        colours = cmap(np.linspace(0, 1, len(homology_boundaries)))

        for homology_boundary, colour in zip(homology_boundaries, colours):
            lc = LineCollection(
                homology_boundary.boundaries, 
                colors=[colour]
            )
            plt.gca().add_collection(lc)

        plt.title('Homology Boundaries', fontsize=14)
        plt.axis('equal')

        plt.show()

    @staticmethod
    def plot_persistence_diagram(data):
        dgm = ripser(data)['dgms']
        title_pattern = ", ".join([
            f'|$H_{i}$|={len(d)}' for i, d in enumerate(dgm)
        ])

        persim.plot_diagrams(
            dgm, show=True,
            title=f'Persistence Diagram ({title_pattern})'
        )

    @staticmethod
    def plot_persistence_entropy(data):
        rips = gudhi.RipsComplex(points=data).create_simplex_tree(max_dimension=1)
        rips.compute_persistence()
        gudhi.plot_persistence_barcode(rips.persistence_intervals_in_dimension(0))
        plt.show()