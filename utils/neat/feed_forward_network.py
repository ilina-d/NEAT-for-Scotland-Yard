import math
from .genome import Genome
from utils.helpers import State, Graph


class FeedForwardNet:
    """ Our Phenotype (the runnable FeedForward Network that is built from a Genome)."""

    _ACTIVATION = {
        'relu' : lambda x: x if x > 0 else 0.0,
        'sigmoid': lambda x: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))),
        'tanh': lambda x: math.tanh(x),
        'identity': lambda x: x
    }

    def __init__(self, genome: Genome | str) -> None:
        """
        Build an FFN from the given Genome.

        Genome Options:
            - A pre-made Genome object.
            - Path to a json file located at 'saved_genomes/path_name.json'.
            - "X" or "D" to create a randomized fully-connected input output Genome.

        Arguments:
            genome: The genome option to use.
        """

        if isinstance(genome, str):
            if genome == "X" or genome == "D":
                genome = Genome(genome)
            else:
                genome = Genome.load_from_file(genome)

        self.genome = genome

        # Build lookup (which neurons wait on which?):

        hidden_neurons = {n.neuron_id: n for n in genome.neuron_genes}

        incoming = {nid: [] for nid in hidden_neurons}
        incoming[genome.output_id] = []

        for link in genome.link_genes:
            if link.is_active and link.id_to in incoming:
                incoming[link.id_to].append((link.id_from, link.weight))

        # Keep only the ones that influence the output:

        required_nids = {genome.output_id}
        stack = [genome.output_id]
        while stack:
            for nid_in, _ in incoming[stack.pop()]:
                if nid_in in hidden_neurons and nid_in not in required_nids:
                    required_nids.add(nid_in)
                    stack.append(nid_in)

        # Order the neurons so every neuron comes after the ones feeding into it:

        waiting_on = {nid: {src for src, _ in incoming[nid] if src in required_nids} for nid in required_nids}
        ready = [nid for nid in genome.input_ids]
        order = []
        while ready:
            nid = ready.pop()
            order.append(nid)
            for other_nid, dependants in waiting_on.items():
                dependants.discard(nid)
                if not dependants and other_nid not in order and other_nid not in ready:
                    ready.append(other_nid)

        # Make an eval list containing all ordered data for a forward pass:

        evals = []
        for nid in order:
            if nid == genome.output_id:
                bias, activation = 0.0, 'identity'
            elif nid in genome.input_ids:
                continue
            else:
                neuron = hidden_neurons[nid]
                bias, activation = neuron.bias, neuron.activation
            evals.append((nid, bias, activation, incoming[nid]))

        self.evals = evals
        self.input_ids = genome.input_ids
        self.output_id = genome.output_id

        self.graph = Graph()


    def forward(self, inputs: list[float]) -> float:
        """
        Feed an input through the network and get an output.

        Arguments:
            inputs: A list of input values.

        Returns:
            The result of the processed input.
        """

        values = dict(zip(self.input_ids, inputs))
        for nid, bias, activation_name, links in self.evals:
            total = bias
            for source_id, weight in links:
                total += values[source_id] * weight
            values[nid] = FeedForwardNet._ACTIVATION[activation_name](total)

        return values[self.output_id]


    def get_score_for_move(self, state: State, name: str, ticket: str, station: int) -> float:
        """
        Score a move according to the FFN's policy.

        Arguments:
            state: The current game state.
            name: The name of the player whose turn it is.
            ticket: The ticket used in the move.
            station: The destination of the move.

        Returns:
            The score of the given move.
        """

        features = self.graph.get_ffn_features_for_x(state, ticket, station) if name == 'x' \
            else self.graph.get_ffn_features_for_d(state, name, ticket, station)

        return self.forward(features)


__all__ = ["FeedForwardNet"]