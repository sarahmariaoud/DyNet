from typing import List
import numpy as np

from Code.System import System
from InterAction import InterAction


class Model:
    def __init__(self, interactions: List[InterAction], nodes: np.ndarray, adj: np.ndarray = None,
                 method='gillespie_direct'):
        if not all(isinstance(interaction, InterAction) for interaction in interactions):
            raise TypeError("All elements in interactions should be an instance of InterAction")

        self.time = 0
        self.system = System(nodes, adj)
        self.interactions = interactions

        self.method = method
        self.simulation = self._initialize_simulation()

    def _initialize_simulation(self):
        method_maps = {'gillespie_direct': GillespieSimulation}
        if self.method in method_maps:
            return method_maps[self.method](self,self.system,self.interactions)
        else:
            raise ValueError(f"Unknown simulation method: {self.method}")

    def __str__(self):
        node_str = np.array2string(self.system.nodes)
        adj_str = np.array2string(self.system.adj)
        return f"System\nNodes:\n{node_str}\nAdjacency matrix:\n{adj_str}"



#how can we ensure that the signature is always the same (is all this switching back and forth costing me time)
# the simulator, needs to :
# 1- initialize
# 2- search and select
# 3- update it's values if needed

class GillespieSimulation:
    def __init__(self, model:Model ,system: System, interactions : List[InterAction]):
        x=0