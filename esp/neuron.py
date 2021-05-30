import random


class Neuron(object):
    def __init__(self):
        self.input_weight = 0.0
        self.output_weight = 0.0
        self.fitness = 0.0

    def init(self, min_value: float, max_value: float):
        self.input_weight = random.uniform(a=min_value, b=max_value)
        self.output_weight = random.uniform(a=min_value, b=max_value)
