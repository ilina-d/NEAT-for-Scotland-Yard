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



def main():
    dnet = DNet5LW()
    dnet.load_weights('5LW_BestTrained')
    xnet = XNet3L()
    xnet.load_weights('3L_BestTrained')

    game = Game(
        detective_r = DNetPlayer('r', dnet),
        detective_g = DNetPlayer('g', dnet),
        detective_b = DNetPlayer('b', dnet),
        detective_o = DNetPlayer('o', dnet),
        detective_p = DNetPlayer('p', dnet),
        mr_x = XNetPlayer(xnet),
        wait_after_move = 'input',
        visuals = True,
        always_show_x = True
    )

    game.play()


if __name__ == '__main__':
    freeze_support()
    main()
    input('... preventing program from closing by waiting for input ...')
