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
    DNetPlayer, TrainingDNetPlayer

from utils.neural_nets.trainers import MrXTrainer, DetectiveTrainer

from utils.neural_nets.detective_nets import DNet3L, DNet5L, DNet5LW, DNet5LRes
from utils.neural_nets.mrx_nets import XNet3L, XNet5LRes, XNet5L, XNet5LW

from utils.neat import Genome, NeuronGene, LinkGene, FeedForwardNet


def main():
    genome = Genome('X', neuron_genes=[NeuronGene(5)],
                    link_genes=[LinkGene(1,5), LinkGene(5, 4), LinkGene(2, 4), LinkGene(3, 4)])

    # for var, value in vars(genome).items():
    #     print(var, value)

    FFN = FeedForwardNet(genome)

    print(FFN.forward([1.7, 2.2, 1]))

if __name__ == '__main__':
    freeze_support()
    main()
    input('... preventing program from closing by waiting for input ...')
