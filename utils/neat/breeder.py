from .genome import Genome
from .fitness_evaluator import FitnessEvaluator

class Breeder:
    """ When a mommy genome and a daddy genome love each other very much... """

    @classmethod
    def crossover(cls, dominant_parent: Genome, recessive_parent: Genome) -> None:
        """ doijdow """

        parents_are_equal = False

        if recessive_parent.fitness > dominant_parent.fitness:
            dominant_parent, recessive_parent = recessive_parent, dominant_parent
        elif recessive_parent.fitness == dominant_parent.fitness:
            parents_are_equal = True
            # mutation diff

        # mix them
        # mutate