import numpy as np


class System:
    """
    A class representing a symmetric network with pairwise interactions.
    """

    def __init__(self, nodes: np.ndarray, adj: np.ndarray = None):
        """
        Initialize the system with nodes and an adjacency matrix.

        :param nodes: A 2D numpy array representing the nodes (NxD).
        :param adj: A 2D numpy array representing the adjacency matrix (NxN). Defaults to an identity matrix.
        """
        self.nodes = np.array(nodes)
        n = len(self.nodes)

        # Assert that it is a 2D array
        assert len(self.nodes.shape) == 2, "Nodes should be a 2D array (NxD)"

        if adj is None:
            self.adj = np.eye(n)
        else:
            self.adj = np.array(adj)
            # Assert that adj is a square matrix and its size matches the number of nodes
            assert self.adj.shape == (
                n,
                n,
            ), "Adjacency matrix should be of size NxN and match the number of nodes"

        self.maps = {
            "+": self._add_edge,
            "-": self._remove_edge,
            "np": self._change_node_property,
        }

    def apply_maps(self, maps):
        """
        Apply a series of maps to the system.

        :param maps: A list of lists where each inner list contains the map key and its parameters.
        """
        # Assert that maps is a list
        assert isinstance(maps, list), "maps should be a list"

        for mi in maps:
            # Assert that each item in maps is a list
            assert isinstance(mi, list), "Each map instruction should be a list"
            assert len(mi) == 2, "Each map instruction should contain two elements"

            # Assert that the first element of each inner list is a string and is a key in self.maps
            assert (
                isinstance(mi[0], str) and mi[0] in self.maps
            ), f"The first element of each map instruction should be a string and a key in the available maps. Found:\
                 {mi[0]}"

            assert isinstance(
                mi[1], tuple
            ), "the second element in the map instruction is a tuple"
            # Assert that the rest of the elements in the inner list match the expected arguments for the map function
            # This is a basic check; for more complex validation, you might need custom validation per map type
            expected_args_count = (
                self.maps[mi[0]].__code__.co_argcount - 1
            )  # -1 to exclude 'self'
            actual_args_count = len(mi[1])  # -1 to exclude the map key
            assert (
                actual_args_count == expected_args_count
            ), f"Map function '{mi[0]}' expects {expected_args_count} arguments, but {actual_args_count} were given"

            # Apply the map
            self.maps[mi[0]](*mi[1])

    def get_size(self):
        return self.nodes.shape[0]

    def _add_edge(self, n1: int, n2: int):
        """
        Add an edge between two nodes.

        :param n1: Index of the first node.
        :param n2: Index of the second node.
        """
        if n1 == n2:
            raise ValueError("Cannot connect a node to itself")
        if self.adj[n1][n2] == 1 or self.adj[n2][n1] == 1:
            raise ValueError("Attempting to add an edge that is already present")
        self.adj[n1][n2] += 1
        self.adj[n2][n1] += 1

    def _remove_edge(self, n1: int, n2: int):
        """
        Remove an edge between two nodes.

        :param n1: Index of the first node.
        :param n2: Index of the second node.
        """
        if n1 == n2:
            raise ValueError("Cannot disconnect a node from itself")
        if self.adj[n1][n2] == 0 or self.adj[n2][n1] == 0:
            raise ValueError("Attempting to remove an edge that doesn't exist")
        self.adj[n1][n2] -= 1
        self.adj[n2][n1] -= 1

    def _change_node_property(self, n1: int, y):
        """
        Change the property of a node.

        :param n1: Index of the node.
        :param y: A numpy array representing the new property of the node.
        """
        # Ensure y has the correct dimensions
        y = np.array(y)
        assert (
            y.shape[0] == self.nodes.shape[1]
        ), "New property dimensions must match the existing property dimensions"
        self.nodes[n1] = y
