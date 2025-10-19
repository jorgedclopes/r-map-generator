import os
from jsonschema import validate
import json
from typing import Tuple

from classes import Population, Pair, Individual


def flatten(arg_list):
    return (item for sublist in arg_list for item in sublist)


def transform_number_of_nodes(number, total):
    return round(number * total) if 0 < number < 1 else number


# TODO: add defaults
<<<<<<< Updated upstream
def read_populations(schema_file="./schema.json", population_file="./populations.json"):
    cwd = os.getcwd()
    with open(os.path.abspath(os.path.join(cwd, schema_file))) as s, open(
        os.path.abspath(os.path.join(cwd, population_file))
    ) as f:
        data = json.load(f)
        schema = json.load(s)
        print(json.dumps(data, indent=2))
        validate(instance=data, schema=schema)
        for entry in data:
            entry["connection_number"] = Pair(
                entry["connection_number"][0], entry["connection_number"][1]
            )
        return tuple(Population(**entry) for entry in data)
=======
def read_populations():
    with open("schema.json") as s, open("populations.json") as f:
        data_array = json.load(f)
        schema = json.load(s)
        print(schema)
        print(data_array)
        validate(instance=data_array, schema=schema)
        for data in data_array:
            data["connection_number"] = Pair(
                data["connection_number"][0], data["connection_number"][1]
            )
        return tuple(Population(**data) for data in data_array)
>>>>>>> Stashed changes


def read_playbooks(playbook_file="./playbooks.json"):
    cwd = os.getcwd()
    with open(os.path.abspath(os.path.join(cwd, playbook_file))) as pb:
        data = json.load(pb)
        return data
        


def print_nodes(nodes) -> None:
    print(json.dumps(nodes))


def print_connections(nodes: Tuple[Individual]):
    print("|{:<20}| {:<35}| {:<60}|".format("name", "uid", "connections"))
    print("-" * 111)
    for el in nodes:
        print("|{:<20}| {:<35}| {:<60}|".format(el.name, el.id, str(el.connections)))
