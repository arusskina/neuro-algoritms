import random
from .neuron import Neuron


class NeuronSubPopulation(object):
    def __init__(self,
                 population_size: int,
                 input_count: int,
                 output_count: int):
        self.population = []
        for i in range(population_size):
            self.population.append(Neuron(
                input_count=input_count,
                output_count=output_count))

    def init(self, min_value: float, max_value: float):
        for i in range(len(self.population)):
            self.population[i].init(
                min_value=min_value,
                max_value=max_value)

    def get_neuron(self) -> Neuron:
        return random.choice(self.population)

    def is_trials_completed(self) -> bool:
        trials = [neuron.trials for neuron in self.population]
        return min(trials) >= 10

    def reset_trials(self):
        for neuron in self.population:
            neuron.trials = 0
