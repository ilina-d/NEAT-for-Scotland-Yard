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

from utils.neat.fitness_evaluator import FitnessEvaluator
from utils.neat.offspring_generator import OffspringGenerator


def main():
    osg = OffspringGenerator()
    g1 = Genome('X')
    g2 = Genome('X')
    g3 = osg.cross_parents(g1, g2)
    print(f'G1: {", ".join([f"{link.id_from}--{link.weight:.2f}->{link.id_to}" for link in g1.link_genes])}')
    print(f'G2: {", ".join([f"{link.id_from}--{link.weight:.2f}->{link.id_to}" for link in g2.link_genes])}')
    print(f'G3: {", ".join([f"{link.id_from}--{link.weight:.2f}->{link.id_to}" for link in g3.link_genes])}')
    print(f'G3: {[f"ID: {n.neuron_id}, ACT: {n.activation}, BIAS: {n.bias}" for n in g3.neuron_genes]}')



if __name__ == '__main__':
    freeze_support()
    main()
    input('... preventing program from closing by waiting for input ...')
