from esp import ESPAlgorithm
from esp import Cancer1Dataset
import random


random.seed(1)

cancer1_dataset = Cancer1Dataset()
x_train, y_train = cancer1_dataset.get_train_data()

algorithm = ESPAlgorithm(
    hidden_layer_size=3,
    population_size=8,
    input_count=x_train.shape[1],
    output_count=y_train.shape[1],
    last_generations_count=5)

algorithm.init(
    min_value=-1.0,
    max_value=1.0)

algorithm.train(
    generations_count=30000,
    x_train=x_train,
    y_train=y_train)
