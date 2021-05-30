import numpy as np


class Neuron(object):
    def __init__(self,
                 input_count: int,
                 output_count: int):
        self.input_count = input_count
        self.output_count = output_count
        self.input_weights = None
        self.output_weights = None
        self.fitness = 0.0
        self.trials = 0

    def init(self, min_value: float, max_value: float):
        self.input_weights = np.random.uniform(
            low=min_value,
            high=max_value,
            size=self.input_count)
        self.output_weights = np.random.uniform(
            low=min_value,
            high=max_value,
            size=self.output_count)
