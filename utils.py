from jsonschema import validate
import json
from typing import Tuple

from classes import Population, Pair, Individual


def flatten(arg_list):
    return (item for sublist in arg_list for item in sublist)


def transform_number_of_nodes(number, total):
    return round(number * total) if 0 < number < 1 else number


# TODO: add defaults
def read_populations():
    with open('schema.json') as s, open('populations.json') as f:
        data_array = json.load(f)
        schema = json.load(s)
        print(schema)
        print(data_array)
        validate(instance=data_array, schema=schema)
        for data in data_array:
            data['connection_number'] = Pair(data['connection_number'][0], data['connection_number'][1])
        return tuple(Population(**data) for data in data_array)


def print_nodes(nodes):
    for node in nodes:
        print(node)


def print_connections(nodes: Tuple[Individual]):
    print("|{:<20}| {:<35}| {:<60}|".format('name', 'uid', 'connections'))
    print('-' * 111)
    for el in nodes:
        print("|{:<20}| {:<35}| {:<60}|".format(el.name, el.id, str(el.connections)))
