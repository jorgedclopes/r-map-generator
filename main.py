import uuid
from dataclasses import asdict
from itertools import cycle, islice
from random import randint, uniform
from typing import Tuple, List

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx import Graph
import names

from classes import Population, Individual
from utils import read_populations, transform_number_of_nodes, flatten


def generate_nodes(total_nodes: int, populations: Tuple[Population]) -> Tuple[Individual]:
    default_populations = tuple(el for el in populations if el.amount == 0)

    def generate_individual(population: Population):
        args = asdict(population)
        args['name'] = names.get_full_name()
        args['id'] = uuid.uuid4().hex
        return Individual.from_dict(args)

    def func(population: Population, total: int) -> Tuple[Individual]:
        return tuple(generate_individual(population)
                     for _ in range(transform_number_of_nodes(population.amount, total)))

    base_nodes: Tuple[Individual] = tuple(flatten((func(e, total_nodes) for e in populations)))
    complement_nodes: Tuple[Individual] = tuple(generate_individual(population)
                                                for population in
                                                islice(cycle(default_populations), total_nodes - len(base_nodes)))
    output: Tuple[Individual] = tuple((*base_nodes, *complement_nodes))
    return output


# TODO: add weight to populations
def connect_node(node: Individual,
                 nodes: List[Individual],
                 populations: Tuple[Population]):
    population = next(el for el in populations if el.status == node.status)
    [first, second] = [population.connection_number.first, population.connection_number.second]
    number_of_connections = randint(first, second)
    valid_connects = [sub_node for sub_node in nodes
                      if sub_node.id not in sub_node.connections and
                      node.id not in sub_node.connections]
    candidate_connections = [sub_node for sub_node in valid_connects
                             if not sub_node.id == node.id and
                             sub_node.id not in node.connections]

    if len(candidate_connections) < number_of_connections:
        raise Exception('Not possible to generate connections with this configuration')

    connections: List[str] = [el.id for el in np.random.choice(candidate_connections,
                                                               size=number_of_connections,
                                                               replace=False)]

    args = asdict(node)
    args['connections'] = connections
    return Individual.from_dict(args)


def generate_connections(nodes: Tuple[Individual],
                         populations: Tuple[Population]) -> Tuple[Individual]:
    connected_nodes = list(nodes)
    for i, node in enumerate(connected_nodes):
        connected_nodes[i] = connect_node(node, connected_nodes, populations)
    return tuple(connected_nodes)


def populate_nodes(graph: Graph,
                   node_list):
    g = graph.copy()
    for el in node_list:
        g.add_node(el.name, status=el.status)
    return g


def populate_edges(graph: Graph,
                   individual_list: Tuple[Individual]):
    g = graph.copy()

    def get_individual_from_uid(uid: str) -> Individual:
        individual = next(filter(lambda ind: ind.id == uid, individual_list))
        return individual

    connection_pairs = [(individual.name, get_individual_from_uid(connection).name)
                        for individual in individual_list for connection in individual.connections]
    g.add_edges_from(connection_pairs)
    return g


def color(status, populations):
    return next(filter(lambda population: population.status == status, populations)).color


if __name__ == '__main__':
    population_list = read_populations()
    number_of_nodes = 20
    nodes = generate_nodes(number_of_nodes, population_list)
    connected_nodes = generate_connections(nodes, population_list)

    empty_G = nx.MultiDiGraph()
    G_nodes = populate_nodes(empty_G, nodes)
    G_full = populate_edges(G_nodes, connected_nodes)

    edge_number = sum(map(lambda node: len(node.connections), connected_nodes))

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G_full, scale=100, iterations=300)
    labels = {node.name: node.name for node in nodes}
    nx.draw_networkx_labels(G_full, pos, labels=labels)
    colors: tuple = tuple(color(el.status, population_list) for el in nodes)
    nx.draw_networkx_nodes(G_full, pos, node_color=colors)
    for node in connected_nodes:
        print(node)
    for edge in G_full.edges:
        style = f'arc3,rad={round(uniform(0., .3), 2)}'
        local_G = empty_G.copy()
        local_G.add_edge(edge[0], edge[1])
        nx.draw_networkx_edges(local_G, pos=pos, arrowsize=10, connectionstyle=style)
    plt.show()
