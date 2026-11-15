from .base_player import BasePlayer
from utils.neat import FeedForwardNet
from ..helpers import State


class FFNPlayer(BasePlayer):
    """ Player agent that plays by an FFN's policy. """

    def __init__(self, ffn: FeedForwardNet, name: str) -> None:
        """
        Create a player instance.

        Arguments:
            ffn: A pretrained Mr. X FFN.
            name: Name of the player ("x", "r", "g", "b", "o", or "p").
        """

        super().__init__(name = name)
        self.ffn = ffn


    def make_move(self, state: State) -> tuple[str, int | None]:
        legal_moves = self.get_legal_moves(state)

        if not legal_moves:
            return '', None

        scores = {}

        for ticket, station in legal_moves:
            scores[(ticket, station)] = self.ffn.get_score_for_move(state, self.name, ticket, station)

        return max(scores, key=scores.get)