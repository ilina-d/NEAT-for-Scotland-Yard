import random


class NeuronGene:
    """ A gene to represent a single Neuron. """
    _ACTIVATION_FUNCTIONS = ("relu", "sigmoid", "tanh")

    def __init__(self, neuron_id: int, bias: float = None, activation: str = None) -> None:
        """
        Create a new NeuronGene. All of these will belong to the hidden layers.

        Arguments:
            neuron_id: The "ID" of the given NeuronGene (also called an Innovation Number).
            bias: The value of the shift after activation of this Neuron.
            activation: The type of activation function applied to the value fed to the Neuron.
                        (Can be one of the following: 'relu', 'tanh', and 'sigmoid')
        """

        self.neuron_id = neuron_id

        self.bias = bias
        self.activation = activation

        if bias is None:
            self.bias = random.normalvariate(mu = 0, sigma = 1)

        if activation is None:
            self.activation = random.choice(self._ACTIVATION_FUNCTIONS)


    def mutate_bias_value(self, nudge_thr: float, change_thr: float, power: float = 0.2, limit: float = 5.0) -> None:
        """ Mutate the bias by shifting its value or totally changing it. """

        val = random.random()

        if val < change_thr:
            self.bias = random.normalvariate(mu = 0, sigma = 1)
        elif val < change_thr + nudge_thr and nudge_thr != 0:
            self.bias += random.normalvariate(mu=0, sigma=power)

        self.bias = max(-limit, min(limit, self.bias))


    def mutate_activation(self, change_thr: float):
        """ Mutate the activation by choosing a random function from the pool. """

        val = random.random()

        if val < change_thr:
            self.activation = random.choice(self._ACTIVATION_FUNCTIONS)


__all__ = ["NeuronGene"]