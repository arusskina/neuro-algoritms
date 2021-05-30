from .neuron import Neuron


class NeuronSubPopulation(object):
    def __init__(self, population_size: int):
        self.population = []
        for i in range(population_size):
            self.population.append(Neuron())

    def init(self, min_value: float, max_value: float):
        for i in range(len(self.population)):
            self.population[i].init(min_value=min_value, max_value=max_value)
