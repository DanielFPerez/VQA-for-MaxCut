from typing import Dict, List

import networkx as nx
from ortools.sat.python import cp_model


# add enum data type for selecting the graph type
GRAPHTYPE = ['CUSTOM', 'STAR', 'CYCLE', 'PATH', 'COMPLETE', 'ERDOS_RENYI']

# Function for creating a graph depending on the enum type
def create_graph(graph_type: str, n_nodes: int, verbose: bool =False):
    if graph_type == 'CUSTOM':
        G = nx.Graph()
        # node_list = list(range(n_nodes))
        # G.add_edges_from([(int(u), int(v)) for u, v in np.random.choice(node_list, (n_nodes, 2))])
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    elif graph_type == 'STAR':
        G = nx.star_graph(n_nodes)
    elif graph_type == 'CYCLE':
        G = nx.cycle_graph(n_nodes)
    elif graph_type == 'PATH':
        G = nx.path_graph(n_nodes)
    elif graph_type == 'COMPLETE':  
        G = nx.complete_graph(n_nodes)
    elif graph_type == 'ERDOS_RENYI':
        G = nx.erdos_renyi_graph(n_nodes, 0.5)
        # Check if erdos renyi graph is connected, if not, create random edges between isolated nodes
        if not nx.is_connected(G):
            isolated_nodes = list(nx.isolates(G))
            for u in isolated_nodes:
                v = np.random.choice(list(G.nodes()))
                G.add_edge(u, v)
    else:
        raise ValueError("Invalid graph type")
    
    G = G.to_undirected()
    if verbose:
        print("Nodes of the graph:", G.nodes())
        print("Edges of the graph:", G.edges()) 

    return G


def multiply_two_node_variables(model: cp_model.CpModel, node_vars: Dict[str, cp_model.IntVar], u: str, v: str): 
    """Multiply two boolean variables in CP-SAT 
    Args:
        cp_model (cp_model.CpModel): CP-SAT model
        node_vars (Dict[str, cp_model.IntVar]): dictionary of node variables
        u, v (str): node names
    Returns:
        cp_model.IntVar: result of the multiplication
    """
    prod_uv = model.new_bool_var(f"prod_{u}_{v}")
    model.add_implication(~node_vars[u], ~prod_uv)
    model.add_implication(~node_vars[v], ~prod_uv)
    model.add_bool_and(~node_vars[u], ~node_vars[v], prod_uv)

    return model, prod_uv


