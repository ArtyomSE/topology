import unittest
from sklearn.datasets import make_circles
from sklearn.utils import shuffle
from homology import ColistHomologyCalculator


class TestHomologyCalculator(unittest.TestCase):

    def setUp(self):
        '''
        Метод для подготовки тестовых наборов данных
        '''
        X, Y = make_circles(n_samples=22, noise=0.1, factor=0.99, random_state=0, shuffle=False)
        self.data_simple = shuffle(X[Y == 0], random_state=42)
        self.epsilon_simple = 0.79

        X, Y = make_circles(n_samples=1000, noise=0.1, factor=0.99, random_state=0, shuffle=False)
        self.data_complex = X[Y == 0]
        self.epsilon_complex = 0.21

    def test_on_simple_data(self):
        '''
        Тест, проверяющий корректность вычисления гомологий на простом
        наборе данных, состоящем из 11 точек
        '''

        print('\nTesting simple case...')

        model = ColistHomologyCalculator()
        model.compute_homologies(self.data_simple, self.epsilon_simple)

        self.assertEqual(model.get_homologies_sizes().T[-1, -1], 1)

    def test_on_complex_data(self):
        '''
        Тест, проверяющий корректность вычисления гомологий на объёмном
        наборе данных, состоящем из 500 точек
        '''

        print('\nTesting complex case...')

        model = ColistHomologyCalculator()
        model.compute_homologies(self.data_complex, self.epsilon_complex)

        self.assertEqual(model.get_homologies_sizes().T[-1, -1], 9)


if __name__ == '__main__':
    unittest.main()