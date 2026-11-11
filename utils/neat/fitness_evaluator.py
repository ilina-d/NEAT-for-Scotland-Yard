from .genome import Genome
from utils.players import BasePlayer, RandomPlayer
from utils.simulator import Simulator


class FitnessEvaluator:
    """ Evaluator to estimate the fitness of an individual or genome. """

    @classmethod
    def evaluate(cls, genome: Genome, opponent: str = "random") -> float:
        """ idhdi """
        # make net passing logic somewhere and make a player that can make moves
        # simulate games against opponent (random or greedy)
        pass


__all__ = ["FitnessEvaluator"]