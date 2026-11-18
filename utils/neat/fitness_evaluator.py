from .genome import Genome
from .feed_forward_network import FeedForwardNet
from utils.players import RandomPlayer, GreedyPlayer, FFNPlayer
from utils.simulator import Simulator


class FitnessEvaluator:
    """ Evaluator to estimate the fitness of an individual or genome. """

    PLAYER_MAPS = {
        'random': RandomPlayer,
        'greedy': GreedyPlayer,
        'ffn': FFNPlayer
    }

    @classmethod
    def evaluate(cls, genome: Genome, opponent: str = 'random', num_sims = 1_000, num_workers = 4,
                 print_out: bool = False) -> float:
        """
        Give the Genome a fitness score, calculated by its win rate.

        Arguments:
            genome: The genome that need to be evaluated.
            opponent: The opponent it will face in the Simulations.
                      (could be one of the following: "random", "greedy")
            num_sims: How many games to simulate against the opponent.
            num_workers: How many workers to use when simulating.
            print_out: Whether to print the simulation progress and results.

        Returns:
            The win rate of the Genome.
        """

        x_player = opponent
        d_player = opponent

        x_args = {'name': 'x'}
        d_args = {}

        if genome.genome_type == "X":
            x_player = 'ffn'
            x_args['ffn'] = FeedForwardNet(genome)
        else:
            d_player = 'ffn'
            d_args['ffn'] = FeedForwardNet(genome)

        x_player = cls.PLAYER_MAPS[x_player]
        d_player = cls.PLAYER_MAPS[d_player]

        sim = Simulator(
            detective_r = d_player(name = 'r', **d_args),
            detective_g = d_player(name = 'g', **d_args),
            detective_b = d_player(name = 'b', **d_args),
            detective_o = d_player(name = 'o', **d_args),
            detective_p = d_player(name = 'p', **d_args),
            mr_x = x_player(**x_args),
            num_sims = num_sims,
            measure_performance = False,
            num_workers = num_workers,
            print_out = print_out
        )
        sim.start()

        if genome.genome_type == "X":
            wr = sim.x_wr
        else:
            wr = sim.d_wr

        genome.fitness = wr
        return wr


__all__ = ["FitnessEvaluator"]