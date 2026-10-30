import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.stats import gaussian_kde
from exceptions import validate_dimensionality
from ripser import ripser
import persim
import gudhi


def plot_data_cloud(data, color='crimson', alpha=0.7, marker='o', s=40):
    '''
    Функция для визуализации 2-мерного облака точек

    Args:
     * data (np.ndarray) - массив размерности n x 2 c координатами точек
     * color (str) - цвет маркеров точек
     * alpha (float) - коэффициент прозрачности точек (от 0 до 1)
     * marker (str) - форма маркера точек
     * s (int) - размер маркеров точек
    '''

    validate_dimensionality(data) 

    plt.scatter(
        data[:, 0], data[:, 1], color=color, alpha=alpha, marker=marker, s=s
    )
    plt.title(f'{data.shape[0]} points', fontsize=14)
    plt.axis('equal')

    plt.show()

def plot_density_heatmap(data, grid_size=100, palette='viridis', show_contours=True, show_points=True):
    '''
    Функция для визуализации плотности распределения данных на плоскости

    Args:
     * data (np.ndarray) - массив размерности n x 2 c координатами точек
     * palette (str) - название цветовой палитры matplotlib для для карты плотности,
       может быть либо 'viridis', либо 'coolwarm'
     * grid_size (int) - разрешение сетки для оценки плотности (чем оно выше, тем плавнее график)
     * show_contours (bool) - флаг необходимости нанесения изолиний
     * show_points (bool) - флаг необходимости отрисовки точек
    '''

    if palette != 'viridis' and palette != 'coolwarm':
        raise ValueError('Недопустимая палитра!')

    validate_dimensionality(data)

    xi, yi = np.mgrid[
        data[:, 0].min() - 0.5:data[:, 0].max() + 0.5:complex(grid_size),
        data[:, 1].min() - 0.5:data[:, 1].max() + 0.5:complex(grid_size)
    ]
    
    kernel = gaussian_kde(np.vstack(data.T))
    zi = kernel(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

    cf = plt.contourf(xi, yi, zi, levels=15, cmap=palette)
    plt.colorbar(cf)

    if show_contours: plt.contour(
        xi, yi, zi, levels=15, colors='black', linewidths=0.5, alpha=0.5
    )

    if show_points: plt.scatter(
        data[:, 0], data[:, 1], color='white', alpha=0.3, edgecolors='black', s=20
    )

    plt.title('Data Density', fontsize=14)
    plt.axis('equal')

    plt.show()

def plot_homologies(data, homology_boundaries, palette='gist_rainbow'):
    '''
    Функция для визуализации на облаке точек найденных в данных гомологий

    Args:
     * data (np.ndarray) - массив размерности n x 2 c координатами точек
     * homology_boundaries (List[Homology]) - коллекция объектов класса Homology,
       каждый из которых содержит контур топологической пустоты
     * palette (str) - название цветовой палитры matplotlib для окрашивания гомологий
    '''

    validate_dimensionality(data)

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

def plot_persistence_diagram(data):
    '''
    Функция для построения диаграммы персистентности

    Args:
     * data (np.ndarray) - массив размерности n x 2 c координатами точек
    '''

    validate_dimensionality(data) 

    dgm = ripser(data)['dgms']
    title_pattern = ", ".join([
        f'|$H_{i}$|={len(d)}' for i, d in enumerate(dgm)
    ])

    persim.plot_diagrams(
        dgm, show=True,
        title=f'Persistence Diagram ({title_pattern})'
    )

def plot_persistence_entropy(data):
    '''
    Функция для построения баркода персистентной энтропии для размерности H0

    Args:
     * data (np.ndarray) - массив размерности n x 2 c координатами точек
    '''

    validate_dimensionality(data) 

    rips = gudhi.RipsComplex(points=data).create_simplex_tree(max_dimension=1)
    rips.compute_persistence()
    gudhi.plot_persistence_barcode(rips.persistence_intervals_in_dimension(0))

    plt.show()
