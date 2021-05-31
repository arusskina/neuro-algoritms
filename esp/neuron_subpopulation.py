import random
import numpy as np
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

    def crossover(self):
        for neuron in self.population:
            neuron.fit_avg_fitness()
        self.population.sort(key=lambda x: x.avg_fitness)
        selected_neurons_count = int(len(self.population) / 4)
        selected_neurons_count -= selected_neurons_count % 2
        for i in range(0, selected_neurons_count, 2):
            parent1 = self.population[i]
            parent2 = self.population[i + 1]
            child1, child2 = Neuron.crossover(
                parent1=parent1,
                parent2=parent2)
            self.population[-selected_neurons_count + i] = child1
            self.population[-selected_neurons_count + i + 1] = child2

    def get_best_neuron(self) -> Neuron:
        self.population.sort(key=lambda x: x.avg_fitness)
        return self.population[0]

    def burst_mutation(self):
        best_neuron = self.get_best_neuron()
        input_count = best_neuron.input_count
        output_count = best_neuron.output_count
        new_population = []
        for _ in self.population:
            new_neuron = Neuron(
                input_count=input_count,
                output_count=output_count)
            new_neuron.input_weights = \
                np.random.standard_cauchy(input_count) + best_neuron.input_weights
            new_neuron.output_weights = \
                np.random.standard_cauchy(output_count) + best_neuron.output_weights
            new_population.append(new_neuron)
        self.population = new_population
