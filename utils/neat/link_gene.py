import random


class LinkGene:
    """ A gene to represent a single link/synaps. """

    def __init__(self, id_from: int, id_to: int, weight: float = None, is_active: bool = None) -> None:
        """
        Create a new LinkGene.

        Arguments:
            id_from: The ID of the Neuron from which this link passes values.
            id_to: The ID of the Neuron to which this link passes values.
            weight: The weight value applied to the signal.
            is_active: Whether the signal is active at all or not.
        """

        self.id_from = id_from
        self.id_to = id_to
        self.link_id = (self.id_from, self.id_to)

        self.weight = weight
        self.is_active = is_active

        if weight is None:
            self.weight = random.normalvariate(mu = 0, sigma = 1)

        if is_active is None:
            self.is_active = True


__all__ = ["LinkGene"]