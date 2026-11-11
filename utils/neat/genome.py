from .link_gene import LinkGene
from .neuron_gene import NeuronGene


class Genome:
    """ A genome/chromosome representing the Genotype of the Neural Network. """

    _GENOME_COUNTER = 0
    _INPUT_COUNTS = {'X' : 3, 'D' : 1}  # filler, will replace with real values

    def __init__(self, genome_type: str, neuron_genes: list[NeuronGene] = None,
                 link_genes: list[LinkGene] = None) -> None:
        """
        Create a new Genome.

        Arguments:
            genome_type: Whether the Genome is used to represent a Mr. X or Detective individual.
                         (Can be either "X" or "D")
            neuron_genes: The (hidden) Neuron Genes that are part of this Genotype.
            link_genes: The Link Genes that are part of this Genotype.
        """

        Genome._GENOME_COUNTER += 1
        self.genome_id = Genome._GENOME_COUNTER

        self.num_inputs = Genome._INPUT_COUNTS[genome_type]
        # num_outputs is always 1

        self.output_id = self.num_inputs + 1
        self.input_ids = [i for i in range(1, self.output_id)]

        self.neuron_genes = neuron_genes
        self.link_genes = link_genes

        self.fitness = None

        if neuron_genes is None:
            self.neuron_genes = []
            self.link_genes = []
            self.initialize_genome()
            # self.set_fitness()


    def initialize_genome(self) -> None:
        """ Initialize a starter fully-connected Genome. """

        for in_id in self.input_ids:
            link = LinkGene(id_from = in_id, id_to = self.output_id)
            self.link_genes.append(link)


    def set_fitness(self, fitness: float) -> None:
        """ Evaluate and set the fitness of the genome. """

        self.fitness = fitness # kill this


__all__ = ["Genome"]