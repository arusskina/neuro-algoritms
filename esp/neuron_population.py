from typing import List
from .neuron_subpopulation import NeuronSubPopulation
from .neuron import Neuron


class NeuronPopulation(object):
    def __init__(self,
                 population_size: int,
                 subpopulation_size: int,
                 input_count: int,
                 output_count: int):
        self.population = []
        for i in range(population_size):
            self.population.append(NeuronSubPopulation(
                population_size=subpopulation_size,
                input_count=input_count,
                output_count=output_count))

    def init(self, min_value: float, max_value: float):
        for i in range(len(self.population)):
            self.population[i].init(
                min_value=min_value,
                max_value=max_value)

    def get_neurons(self) -> List[Neuron]:
        return list(map(lambda x: x.get_neuron(), self.population))

    def is_trials_completed(self) -> bool:
        trials = [subpopulation.is_trials_completed() for subpopulation in self.population]
        return not (False in trials)

    def reset_trials(self):
        for subpopulation in self.population:
            subpopulation.reset_trials()

    @staticmethod
    def increment_trials(neurons: List[Neuron]):
        for n in neurons:
            n.trials += 1
