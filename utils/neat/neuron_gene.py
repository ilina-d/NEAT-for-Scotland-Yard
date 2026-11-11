import random


class NeuronGene:
    """ A gene to represent a single Neuron. """
    _MAX_ID_VAL = 0
    _ACTIVATION_FUNCTIONS = ("relu", "sigmoid", "tanh")

    def __init__(self, neuron_id: int = None, bias: float = None, activation: str = None) -> None:
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

        if neuron_id is None:
            NeuronGene._MAX_ID_VAL += 1
            self.neuron_id = NeuronGene._MAX_ID_VAL
        elif neuron_id > NeuronGene._MAX_ID_VAL:
            NeuronGene._MAX_ID_VAL = neuron_id

        if bias is None:
            self.bias = random.normalvariate(mu = 0, sigma = 1)

        if activation is None:
            self.activation = random.choice(self._ACTIVATION_FUNCTIONS)


__all__ = ["NeuronGene"]