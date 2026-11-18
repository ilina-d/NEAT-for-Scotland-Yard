from .link_gene import LinkGene
from .neuron_gene import NeuronGene
import json
import os


class Genome:
    """ A genome/chromosome representing the Genotype of the Neural Network. """

    _INPUT_COUNTS = {'X' : 20, 'D' : 16}

    def __init__(self, genome_type: str, neuron_genes: list[NeuronGene] = None,
                 link_genes: list[LinkGene] = None) -> None:
        """
        Create a new Genome.

        Arguments:
            genome_type: Whether the Genome is used to represent a Mr. X or Detective individual.
                         (Can be either "X" or "D")
            neuron_genes: The (hidden) Neuron Genes that are part of this Genotype.
                         (If no genes are passed, the Genome is preset to a random fully connected minimal individual)
            link_genes: The Link Genes that are part of this Genotype.
        """

        self.genome_type = genome_type
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

        self.nid_dict = dict()
        for n_gene in self.neuron_genes:
            self.nid_dict[n_gene.neuron_id] = n_gene

        self.lid_dict = dict()
        for l_gene in self.link_genes:
            self.lid_dict[l_gene.link_id] = l_gene


    def initialize_genome(self) -> None:
        """ Initialize a starter fully-connected Genome. """

        for in_id in self.input_ids:
            link = LinkGene(id_from = in_id, id_to = self.output_id)
            self.link_genes.append(link)


    def add_link(self, start: int, end: int) -> None:
        """ Add a new link gene to Genome. """

        link = LinkGene(start, end)
        self.link_genes.append(link)
        self.lid_dict[(start, end)] = link


    def add_link_obj(self, link: LinkGene) -> None:
        """ Add a new link gene to the Genome. """

        self.link_genes.append(link)
        self.lid_dict[link.link_id] = link


    def add_neuron_obj(self, neuron: NeuronGene) -> None:
        """ Add a new neuron gene to the Genome. """

        self.neuron_genes.append(neuron)
        self.nid_dict[neuron.neuron_id] = neuron


    def remove_neuron_obj(self, neuron: NeuronGene) -> None:
        """ Remove the given neuron from the Genome, and all the links that are connected to it. """

        n_id = neuron.neuron_id
        self.neuron_genes.remove(neuron)
        self.nid_dict.pop(n_id)

        for link_id in list(self.lid_dict):
            if n_id in link_id:
                self.remove_link(link_id)


    def remove_link(self, link_id: tuple[int, int]) -> None:
        """ Remove a link from the Genome. """

        link = self.lid_dict.pop(link_id)
        self.link_genes.remove(link)


    @classmethod
    def load_from_file(cls, file_path) -> 'Genome':
        """
        Load a Genome object from a save file at 'saved_genomes/folder_name/file_name.json'.
        Folder name should be 'mrx_genomes' or 'detective_genomes'.

        Arguments:
            file_path: The path of the file containing the json data.
        """

        with open(f'saved_genomes/{file_path}.json', 'r') as file:
            data = json.load(file)

        neuron_genes = []
        for n_gene in data['neuron_genes']:
            neuron_genes.append(NeuronGene(n_gene['neuron_id'], n_gene['bias'], n_gene['activation']))

        link_genes = []
        for l_gene in data['link_genes']:
            link_genes.append(LinkGene(l_gene['id_from'], l_gene['id_to'], l_gene['weight'], l_gene['is_active']))

        genome = Genome(data['genome_type'], neuron_genes, link_genes)
        genome.fitness = data['fitness']

        return genome


    def save_to_file(self, file_name):
        """
        Save a Genome as a json file located at 'saved_genomes/folder_name/file_name.json'.
        Folder name will be 'mrx_genomes' or 'detective_genomes' depending on the genome type.

        Arguments:
            file_name: The name of the file that will contain the json data.
        """

        folder_name = 'mrx_genomes' if self.genome_type == 'X' else 'detective_genomes'
        os.makedirs(f'saved_genomes/{folder_name}', exist_ok=True)

        data = dict()
        data['genome_type'] = self.genome_type

        neuron_genes = []
        for n_gene in self.neuron_genes:
            neuron_genes.append({'neuron_id': n_gene.neuron_id, 'bias': n_gene.bias, 'activation': n_gene.activation})

        link_genes = []
        for l_gene in self.link_genes:
            link_genes.append({'id_from': l_gene.id_from, 'id_to': l_gene.id_to, 'weight': l_gene.weight, 'is_active': l_gene.is_active})

        data['neuron_genes'] = neuron_genes
        data['link_genes'] = link_genes
        data['fitness'] = self.fitness

        with open(f'saved_genomes/{folder_name}/{file_name}.json', 'w') as file:
            json.dump(data, file)


__all__ = ["Genome"]