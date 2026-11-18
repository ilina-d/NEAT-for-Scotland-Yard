from .link_gene import LinkGene
from .neuron_gene import NeuronGene
from .genome import Genome
from .fitness_evaluator import FitnessEvaluator
import random
import json
import os


class OffspringGenerator:
    """ A class that covers the mating behaviour of individuals. """

    _MUTATION_CHANCE = {
        'add_link': 0.1,
        'remove_link': 0.1,
        'add_neuron': 0.1,
        'remove_neuron': 0.1,

        'nudge_weight': 0.1,
        'change_weight': 0.1,
        'nudge_bias': 0.1,
        'change_bias': 0.1,
        'change_activation': 0.1,
        'toggle_enabled': 0.1
    }

    _HIDDEN_ID_MAP_X : dict[tuple[int, int], int] = {}
    _HIDDEN_ID_MAP_D : dict[tuple[int, int], int] = {}
    _last_hidden_id_x : int = 0
    _last_hidden_id_d : int = 0
    _output_id_x = 21
    _output_id_d = 17

    def __init__(self) -> None:
        """ Set up the OffspringGenerator. """
        OffspringGenerator._load_id_dict()

    def __del__(self) -> None:
        """ Save the id dictionary before discarding the Generator. """
        OffspringGenerator._save_id_dict()


    @staticmethod
    def cross_parents(parent1: Genome, parent2: Genome, merge_type: str = 'per_gene') -> Genome:
        """
        Combine the genes of two parents to generate an offspring.

        Merge Type options:
            - 'per_gene': each gene is inherited as a whole from one of the parents.
            - 'per_feature': features of a given gene can be inherited from either parent.

        Arguments:
            parent1: The first parents Genome.
            parent2: The second parents Genome.
            merge_type: When merging parents, what variant of inheritance to use.
        """

        if parent1.fitness is None:
            FitnessEvaluator.evaluate(parent1)
        if parent2.fitness is None:
            FitnessEvaluator.evaluate(parent2)

        dominant_parent = parent1
        recessive_parent = parent2

        if recessive_parent.fitness > dominant_parent.fitness:
            dominant_parent, recessive_parent = recessive_parent, dominant_parent

        offspring = OffspringGenerator._mix_genes(dominant_parent, recessive_parent, merge_type)
        mutated_offspring = OffspringGenerator._mutate_offspring(offspring)

        return mutated_offspring


    @classmethod
    def _mix_genes(cls, dominant_parent: Genome, recessive_parent: Genome, merge_type: str) -> Genome:
        """ Take genes from both parents and create the offspring Genome. """

        dn_dict = dominant_parent.nid_dict
        rn_dict = recessive_parent.nid_dict
        child_neurons = []

        for n_id in dn_dict:
            if n_id in rn_dict:              # Common neuron gene
                dn = dn_dict[n_id]
                rn = rn_dict[n_id]

                if merge_type == 'per_gene':
                    parent = random.choice([dn, rn])
                    bias = parent.bias
                    activation = parent.activation
                else:
                    bias = random.choice([dn.bias, rn.bias])
                    activation = random.choice([dn.activation, rn.activation])

                child_neurons.append(NeuronGene(n_id, bias, activation))

            else:                          # Inherit dominant disjoint/excess neuron gene
                parent = dn_dict[n_id]
                child_neurons.append(NeuronGene(n_id, parent.bias, parent.activation))

        dl_dict = dominant_parent.lid_dict
        rl_dict = recessive_parent.lid_dict
        child_links = []

        for l_id in dl_dict:
            if l_id in rl_dict:             # Common link gene
                dl = dl_dict[l_id]
                rl = rl_dict[l_id]

                if merge_type == 'per_gene':
                    parent = random.choice([dl, rl])
                    weight = parent.weight
                    is_active = parent.is_active
                else:
                    weight = random.choice([dl.weight, rl.weight])
                    is_active = random.choice([dl.is_active, rl.is_active])

                child_links.append(LinkGene(l_id[0], l_id[1], weight, is_active))

            else:                           # Inherit dominant disjoint/excess link gene
                parent = dl_dict[l_id]
                child_links.append(LinkGene(l_id[0], l_id[1], parent.weight, parent.is_active))

        return Genome(dominant_parent.genome_type, child_neurons, child_links)


    @classmethod
    def _mutate_offspring(cls, offspring: Genome) -> Genome:
        """ Apply mutations with a set chance to the offspring Genome. """

        # Structural mutations:
        cls._add_link_mutation(offspring)
        cls._remove_link_mutation(offspring)
        cls._add_neuron_mutation(offspring)
        cls._remove_neuron_mutation(offspring)

        # Non-structural mutations:
        for neuron in offspring.neuron_genes:
            neuron.mutate_bias_value(cls._MUTATION_CHANCE['nudge_bias'], cls._MUTATION_CHANCE['change_bias'])
            neuron.mutate_activation(cls._MUTATION_CHANCE['change_activation'])

        for link in offspring.link_genes:
            link.mutate_weight_value(cls._MUTATION_CHANCE['nudge_weight'], cls._MUTATION_CHANCE['change_weight'])
            link.mutate_active_flag(cls._MUTATION_CHANCE['toggle_enabled'])

        return offspring


    @classmethod
    def _add_link_mutation(cls, offspring: Genome) -> Genome:
        """ Mutate the offspring by adding a link gene that does not create a cycle. """

        if random.random() >= cls._MUTATION_CHANCE['add_link']:
            return offspring

        start_pool = list(offspring.nid_dict)
        start_pool.extend(offspring.input_ids)

        end_pool = list(offspring.nid_dict)
        end_pool.append(offspring.output_id)

        start = random.choice(start_pool)
        end = random.choice(end_pool)

        if (start, end) in offspring.lid_dict:
            return offspring

        if cls._link_causes_cycle(start, end, list(offspring.lid_dict), offspring.output_id):
            return offspring

        offspring.add_link(start, end)

        return offspring


    @classmethod
    def _link_causes_cycle(cls, start: int, end: int, links: list[tuple[int, int]], output_id: int):
        """ Does this new link make a cycle in the network? """

        lookup = {}
        for l_start, l_end in links:
            if l_start not in lookup:
                lookup[l_start] = []
            lookup[l_start].append(l_end)

        queue = [i for i in lookup[end]]
        idx = 0
        while idx < len(queue):
            n = queue[idx]
            if n == output_id:
                idx += 1
                continue

            tail = lookup[n]
            if start in tail:
                return True

            queue.extend(tail)
            idx += 1

        return False


    @classmethod
    def _remove_link_mutation(cls, offspring: Genome) -> Genome:
        """ Mutate the offspring by removing a link gene. """

        if random.random() >= cls._MUTATION_CHANCE['remove_link'] or len(offspring.link_genes) == 0:
            return offspring

        links = list(offspring.lid_dict)
        link = random.choice(links)
        offspring.remove_link(link)

        return offspring


    @classmethod
    def _add_neuron_mutation(cls, offspring: Genome) -> Genome:
        """ Mutate the offspring by splitting a link, and adding a new neuron gene. """

        if random.random() >= cls._MUTATION_CHANCE['add_neuron']:
            return offspring

        link_ids = list(offspring.lid_dict)
        link_id = random.choice(link_ids)

        old_link = offspring.lid_dict[link_id]
        old_link.is_active = False

        hidden_map = cls._HIDDEN_ID_MAP_X if offspring.genome_type == "X" else cls._HIDDEN_ID_MAP_D

        if link_id in hidden_map:
            neuron = NeuronGene(hidden_map[link_id])
        else:
            if offspring.genome_type == "X":
                cls._last_hidden_id_x += 1
                num = cls._last_hidden_id_x
            else:
                cls._last_hidden_id_d += 1
                num = cls._last_hidden_id_d

            hidden_map[link_id] = num
            neuron = NeuronGene(num)

        old_start, old_end = link_id
        link1 = LinkGene(old_start, neuron.neuron_id, weight = 1.0)
        link2 = LinkGene(neuron.neuron_id, old_end, weight = old_link.weight)

        offspring.add_neuron_obj(neuron)
        offspring.add_link_obj(link1)
        offspring.add_link_obj(link2)

        return offspring


    @classmethod
    def _remove_neuron_mutation(cls, offspring: Genome) -> Genome:
        """ Mutate the offspring by removing a neuron gene and all links connected to it. """

        if random.random() >= cls._MUTATION_CHANCE['remove_neuron'] or len(offspring.neuron_genes) == 0:
            return offspring

        neuron = random.choice(offspring.neuron_genes)
        offspring.remove_neuron_obj(neuron)

        return offspring


    @classmethod
    def _load_id_dict(cls) -> None:
        """ Load the dictionaries for id allocation of newly created neurons. """

        os.makedirs('neuron_id_dicts/', exist_ok = True)

        try:
            with open('neuron_id_dicts/x_dict.json', 'r') as file:
                data_x = json.load(file)
        except:
            data_x = {}

        x_map = {}
        for l_id in data_x:
            parts = l_id.split(", ")
            x_map[(int(parts[0]), int(parts[1]))] = data_x[l_id]

        cls._HIDDEN_ID_MAP_X = x_map
        cls._last_hidden_id_x = len(x_map) + cls._output_id_x

        try:
            with open(f'neuron_id_dicts/d_dict.json', 'r') as file:
                data_d = json.load(file)
        except:
            data_d = {}

        d_map = {}
        for l_id in data_d:
            parts = l_id.split(", ")
            d_map[(int(parts[0]), int(parts[1]))] = data_d[l_id]

        cls._HIDDEN_ID_MAP_D = d_map
        cls._last_hidden_id_d = len(d_map) + cls._output_id_d


    @classmethod
    def _save_id_dict(cls) -> None:
        """ Save the id allocation dictionaries. """

        os.makedirs(f'neuron_id_dicts', exist_ok=True)

        data_x = {}
        for start, end in cls._HIDDEN_ID_MAP_X:
            data_x[f'{start}, {end}'] = cls._HIDDEN_ID_MAP_X[(start, end)]

        with open(f'neuron_id_dicts/x_dict.json', 'w') as file:
            json.dump(data_x, file)

        data_d = {}
        for start, end in cls._HIDDEN_ID_MAP_D:
            data_d[f'{start}, {end}'] = cls._HIDDEN_ID_MAP_D[(start, end)]

        with open(f'neuron_id_dicts/d_dict.json', 'w') as file:
            json.dump(data_d, file)


__all__ = ["OffspringGenerator"]