import numpy as np
from sklearn.metrics import pairwise_distances
from exceptions import validate_dimensionality
from utils import execution_time, singleton
from abc import ABC, abstractmethod
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class Homology:
    boundaries: np.ndarray

    def __add__(self, other):
        return Homology(
            np.unique(np.vstack((self.boundaries, other.boundaries)), axis=0)
        )


class HomologyCalculator(ABC):
    def __init__(self):
        self.data = None
        self.n = None
        self.epsilon = None

        self.filtration = None
        self.boundary_matrix = None
        self.homology_boundaries = None

    @execution_time
    def compute_homologies(self, data, epsilon):
        validate_dimensionality(data)

        self.data = data
        self.n = data.shape[0]
        self.epsilon = epsilon

        self.filtration = self.compute_filtration()
        self.boundary_matrix = self.compute_boundary_matrix()
        self.homology_boundaries = self.reduce_boundary_matrix()

        return self.homology_boundaries

    def compute_filtration(self):
        dist_matrix = pairwise_distances(self.data)

        adj_matrix = np.zeros((self.n, self.n), dtype=np.int8)
        adj_matrix[dist_matrix[:, :] <= self.epsilon] = 1
        adj_matrix -= np.eye(self.n, dtype=np.int8)

        edges = np.array(np.where(adj_matrix == 1)).T
        filtration = np.array([
            tuple(edge) for edge in tqdm(edges[edges[:, 0] < edges[:, 1]], desc='Filtration computing')
        ])

        return filtration

    @abstractmethod
    def compute_boundary_matrix(self):
        pass

    @abstractmethod
    def reduce_boundary_matrix(self):
        pass

    def decode_homology_boundaries(self, homology_boundaries):
        decoded_homology_boundaries = dict()

        for key in list(homology_boundaries.keys()):
            decoded_homology_boundaries[key] = Homology(np.array([
                [
                    self.filtration[edge][0],
                    self.filtration[edge][1]
                ] for edge in homology_boundaries[key]
            ]))

        return decoded_homology_boundaries

    def get_homologies_sizes(self):
        homologies_sizes = [
            homology.boundaries.shape[0] for homology in self.homology_boundaries.values()
        ]
        return np.vstack(np.unique(homologies_sizes, return_counts=True)).T


@singleton
class MatrixHomologyCalculator(HomologyCalculator):
    def __init__(self):
        super().__init__()

    def compute_boundary_matrix(self):
        edges_nums = np.vstack((
            np.arange(self.filtration.shape[0]), np.arange(self.filtration.shape[0])
        )).T.reshape(-1,)

        boundary_matrix = np.zeros(
            (self.n, self.filtration.shape[0]), dtype=np.int8
        )

        boundary_matrix[self.filtration.reshape(-1,), edges_nums] = 1

        return boundary_matrix

    def reduce_boundary_matrix(self):
        lowest_ones = np.zeros(self.boundary_matrix.shape[1], dtype=int) - 1

        homology_boundaries, keys = dict(), set()

        for j in tqdm(range(self.boundary_matrix.shape[1]), desc='Boundary matrix reduction'):
            lowest_one_index_is_unique = False

            while lowest_one_index_is_unique == False:
                try:
                    lowest_one_index = np.where(self.boundary_matrix[:, j] == 1)[0][-1]
                except IndexError:
                    break

                if lowest_ones[lowest_one_index] == -1:
                    lowest_ones[lowest_one_index] = j
                    lowest_one_index_is_unique = True
                else:
                    self.boundary_matrix[:, j] += self.boundary_matrix[:, lowest_ones[lowest_one_index]]
                    self.boundary_matrix[self.boundary_matrix > 1] = 0

                    if j not in keys:
                        homology_boundaries[j] = [j]
                        keys.add(j)

                    if lowest_ones[lowest_one_index] in keys:
                        homology_boundaries[j] += homology_boundaries[lowest_ones[lowest_one_index]]
                    else:
                        homology_boundaries[j].append(lowest_ones[lowest_one_index])

        # keys &= set(np.where(np.sum(self.boundary_matrix, axis=0) == 0)[0])
        # homology_boundaries = {key: homology_boundaries[key] for key in keys}

        return self.decode_homology_boundaries(homology_boundaries)


class ListHomologyCalculator(HomologyCalculator):
    def __init__(self):
        super().__init__()

    def compute_boundary_matrix(self):
        return np.copy(self.filtration)

    def reduce_boundary_matrix(self):
        lowest_ones = np.zeros(len(self.boundary_matrix), dtype=int) - 1

        homology_boundaries, keys = dict(), set()

        for j in tqdm(range(self.boundary_matrix.shape[0]), desc='Boundary matrix reduction'):
            lowest_one_index_is_unique = False

            while lowest_one_index_is_unique == False:
                lowest_one_index = self.boundary_matrix[j][-1]
                if lowest_one_index == 0: break

                if lowest_ones[lowest_one_index] == -1:
                    lowest_ones[lowest_one_index] = j
                    lowest_one_index_is_unique = True
                else:
                    ones_indices = sorted(
                        set(self.boundary_matrix[j]) ^ set(self.boundary_matrix[lowest_ones[lowest_one_index]])
                    )

                    if len(ones_indices) == 0:
                        self.boundary_matrix[j] = np.array([0, 0])
                    else:
                        self.boundary_matrix[j] = np.array(ones_indices)

                    if j not in keys:
                        homology_boundaries[j] = [j]
                        keys.add(j)

                    if lowest_ones[lowest_one_index] in keys:
                        homology_boundaries[j] += homology_boundaries[lowest_ones[lowest_one_index]]
                    else:
                        homology_boundaries[j].append(lowest_ones[lowest_one_index])

        # keys &= set(np.where(np.sum(self.boundary_matrix, axis=1) == 0)[0])
        # homology_boundaries = {key: homology_boundaries[key] for key in keys}

        return self.decode_homology_boundaries(homology_boundaries)


class ColistHomologyCalculator(HomologyCalculator):
    def __init__(self):
        super().__init__()

    def compute_boundary_matrix(self):
        return np.copy(self.filtration)

    def reduce_boundary_matrix(self):
        homologies = dict()

        for j in tqdm(np.arange(self.n - 1, -1, step=-1), desc='Boundary matrix reduction'):
            identity_elements_indices = np.where(self.boundary_matrix[:, 1] == j)[0]

            if identity_elements_indices.shape[0] > 1:
                leading_element = self.boundary_matrix[identity_elements_indices[0]]
                reducing_elements_indices = identity_elements_indices[1:]

                new_voids = set(reducing_elements_indices) - set(homologies.keys())
                for num in new_voids:
                    homologies[num] = [np.copy(self.boundary_matrix[num])]

                for index in reducing_elements_indices:
                    homologies[index].append(1 * np.copy(leading_element))

                self.boundary_matrix[reducing_elements_indices, 1] = leading_element[0]
                self.boundary_matrix[reducing_elements_indices] = np.sort(
                    self.boundary_matrix[reducing_elements_indices]
                )

        return self.decode_homology_boundaries(homologies)

    def decode_homology_boundaries(self, homology_boundaries):
            decoded_homology_boundaries = dict()
    
            for key in list(homology_boundaries.keys()):
                decoded_homology_boundaries[key] = Homology(np.array([
                    [
                        self.data[edge][0],
                        self.data[edge][1]
                    ] for edge in homology_boundaries[key]
                ]))
    
            return decoded_homology_boundaries
