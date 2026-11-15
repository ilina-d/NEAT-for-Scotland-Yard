import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # FORCES CPU USAGE!
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

from multiprocessing import freeze_support
from utils.game import Game
from utils.simulator import Simulator

from utils.players import \
    RandomPlayer, UserPlayer, ExplorerPlayer, GreedyPlayer, \
    XNetPlayer, XNetSemiRandomPlayer, TrainingXNetPlayer, \
    DNetPlayer, TrainingDNetPlayer, FFNPlayer

from utils.neural_nets.trainers import MrXTrainer, DetectiveTrainer

from utils.neural_nets.detective_nets import DNet3L, DNet5L, DNet5LW, DNet5LRes
from utils.neural_nets.mrx_nets import XNet3L, XNet5LRes, XNet5L, XNet5LW

from utils.neat import Genome, NeuronGene, LinkGene, FeedForwardNet


def main():
    # genome = Genome('X', neuron_genes=[NeuronGene(5)],
    #                 link_genes=[LinkGene(1,5), LinkGene(5, 4), LinkGene(2, 4), LinkGene(3, 4)])

    # for var, value in vars(genome).items():
    #     print(var, value)

    genome = Genome('X')
    FFN = FeedForwardNet(genome)
    ffn_x = FFNPlayer(FFN, 'x')

    genome_d = Genome('D')
    FFN_D = FeedForwardNet(genome_d)

    simulator = Simulator(
        detective_r=RandomPlayer('r'),
        detective_g=RandomPlayer('g'),
        detective_b=RandomPlayer('b'),
        detective_o=RandomPlayer('o'),
        detective_p=RandomPlayer('p'),
        mr_x=ffn_x,
        num_sims=100_000,
        measure_performance=False,
        num_workers=4
    )
    simulator.start()

    # game = Game(
    #     detective_r = FFNPlayer(FFN_D, 'r'),
    #     detective_g = FFNPlayer(FFN_D, 'g'),
    #     detective_b = FFNPlayer(FFN_D, 'b'),
    #     detective_o = FFNPlayer(FFN_D, 'o'),
    #     detective_p = FFNPlayer(FFN_D, 'p'),
    #     mr_x = RandomPlayer('x'),
    #     wait_after_move = 'input',
    #     always_show_x = True
    # )
    # game.play()

if __name__ == '__main__':
    freeze_support()
    main()
    input('... preventing program from closing by waiting for input ...')
