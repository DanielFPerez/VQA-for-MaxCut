import pickle
import networkx as nx
from typing import List, Dict


def load_pickle(src_dir: str) -> object:
    with open(src_dir, 'rb') as f:
        return pickle.load(f)   
    

def save_pickle(data: object, dest_dir: str) -> int:
    with open(dest_dir, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {dest_dir}.")    
    return 0


# Generate random graphs
def generate_random_graphs(num_graphs: int, num_nodes: int, edge_prob: float = 0.5) -> List[nx.Graph]:
    graphs = [nx.erdos_renyi_graph(num_nodes, edge_prob) for _ in range(num_graphs)]
    # set random positions for nodes
    for elem in graphs:
        nx.set_node_attributes(elem, nx.spring_layout(elem), 'pos')
    return graphs

def show_graph(graph: nx.Graph, font: str = 'white', node_size: int = 600):
    return nx.draw(graph,pos=graph.nodes(data='pos'), with_labels=True, font_color=font, node_size=node_size)



def get_case(data: Dict, index: int):
    if index < 0 or index >= len(data['graphs']):
        raise IndexError("Index out of bounds")
    return data['graphs'][index], data['optimized_parameters'][index]