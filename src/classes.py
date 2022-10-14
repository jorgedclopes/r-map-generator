from dataclasses import dataclass
import inspect
from typing import Union, Tuple


@dataclass(frozen=True)
class Pair:
    first: any
    second: any


@dataclass(frozen=True)
class NodeAttributes:
    status: str
    color: str


@dataclass(frozen=True)
class Individual(NodeAttributes):
    id: str
    name: str
    connections: Tuple = ()

    @classmethod
    def from_dict(cls, env):
        return cls(**{
            k: v for k, v in env.items()
            if k in inspect.signature(cls).parameters
        })


@dataclass(frozen=True)
class Population(NodeAttributes):
    amount: Union[int, float]
    connection_number: Pair
