from .neuron_subpopulation import NeuronSubPopulation


class NeuronPopulation(object):
    def __init__(self, population_size: int, subpopulation_size: int):
        self.population = []
        for i in range(population_size):
            self.population.append(NeuronSubPopulation(population_size=subpopulation_size))

    def init(self, min_value: float, max_value: float):
        for i in range(len(self.population)):
            self.population[i].init(min_value=min_value, max_value=max_value)
