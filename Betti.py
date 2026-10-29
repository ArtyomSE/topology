import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from ripser import ripser

class BettiNumbersCalculator:
    def __init__(self):
        self.data = None
        self.dynamics = None

    def compute_betti_numbers_dynamics(self, data):
        self.data = data

        dynamics = ripser(
            pairwise_distances(self.data, metric='euclidean'),
            distance_matrix=True, maxdim=1
        )

        segments = dynamics['dgms'][1]

        epsilons = np.sort(list(set(segments.reshape(-1,))))

        indices = dict()
        for i in range(len(epsilons)):
            indices[epsilons[i]] = i

        homologies_num = np.zeros(len(epsilons))
        for l, r in segments:
            homologies_num[indices[l]] += 1
            homologies_num[indices[r]] -= 1

        for i in range(1, len(epsilons)):
            homologies_num[i] += homologies_num[i - 1]

        self.dynamics = np.vstack((epsilons, homologies_num))
        return self.dynamics

    def plot_betti_numbers_dynamics(self, epsilons=None, homologies_num=None):
        if epsilons == None or homologies_num == None:
            epsilons, homologies_num = self.dynamics

        fig, axes = plt.subplots(2, 1, figsize=(10, 12))

        axes[0].plot(epsilons, homologies_num, color='royalblue')
        axes[0].set_title(r'$H_1$ dynamics', fontsize=14)
        axes[0].set_xlabel(r'$\epsilon$', fontsize=12)
        axes[0].set_ylabel(r'$H_1$    ', rotation='horizontal', fontsize=12)
        axes[0].grid(axis='y', linestyle='--')

        axes[1].plot(np.arange(epsilons.shape[0]), homologies_num, color='forestgreen')
        axes[1].set_title(r'$H_1$ dynamics', fontsize=14)
        axes[1].set_xlabel(r'$ind(\epsilon$)', fontsize=12)
        axes[1].set_ylabel(r'$H_1$    ', rotation='horizontal', fontsize=12)
        axes[1].grid(axis='y', linestyle='--')

        plt.show()