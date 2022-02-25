# pylint: disable=invalid-name
import networkx as nx


def read_graph(filename):
    """
    read a graph file from filename, and return a graph object (as in networkx)
    """
    with open(filename, 'r') as my_graph:
        num_line = my_graph.readline()

        num_data = list(map(int, num_line.split()))
        assert len(num_data) == 3

        n, m = num_data[0], num_data[1]

        G = nx.Graph()
        for i, line in enumerate(my_graph, start=1):
            edge_end = list(map(int, line.split()))
            i_edges = [(i, v) for v in edge_end]
            G.add_edges_from(i_edges)
    return G


if __name__ == '__main__':
    default_filename = 'karate.graph'
    default_G = read_graph(default_filename)
    print(nx.number_of_nodes(default_G), nx.number_of_edges(default_G))
    print(default_G.edges(1))
