import numpy as np
from .neuron_population import NeuronPopulation
from .neural_network import NeuralNetwork
from .utils import mse


def forward_train(neural_network: NeuralNetwork,
                  x_train: np.array,
                  y_train: np.array) -> float:
    dataset_size = x_train.shape[0]
    errors = []
    for i in range(dataset_size):
        output = neural_network.forward(input_data=x_train[i])
        error = mse(y_true=y_train[i], y_pred=output)
        errors.append(error)
    return np.array(errors).mean()


class ESPAlgorithm(object):
    def __init__(self,
                 hidden_layer_size: int,
                 population_size: int,
                 input_count: int,
                 output_count: int):
        self.population = NeuronPopulation(
            population_size=hidden_layer_size,
            subpopulation_size=population_size,
            input_count=input_count,
            output_count=output_count)

    def init(self, min_value: float, max_value: float):
        self.population.init(
            min_value=min_value,
            max_value=max_value)

    def train(self,
              generations_count: int,
              x_train: np.array,
              y_train: np.array):
        for generation in range(generations_count):
            trials_count = 0
            while not self.population.is_trials_completed():
                selected_neurons = self.population.get_neurons()
                NeuronPopulation.increment_trials(selected_neurons)
                neural_network = NeuralNetwork(
                    hidden_neurons=selected_neurons)
                error = forward_train(
                    neural_network=neural_network,
                    x_train=x_train,
                    y_train=y_train)
                for neuron in selected_neurons:
                    neuron.cumulative_fitness += error
                trials_count += 1
            self.population.crossover()
            self.population.reset_trials()
