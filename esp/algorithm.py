import numpy as np
from .neuron_population import NeuronPopulation


class ESPAlgorithm(object):
    def __init__(self,
                 hidden_layer_size: int,
                 population_size: int):
        self.population = NeuronPopulation(
            population_size=hidden_layer_size,
            subpopulation_size=population_size)

    def init(self, min_value: float, max_value: float):
        self.population.init(
            min_value=min_value,
            max_value=max_value)

    def train(self,
              generations_count: int,
              x_train: np.array,
              y_train: np.array):
        pass
