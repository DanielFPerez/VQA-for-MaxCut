import click
import cirq
import numpy as np
import networkx as nx
import scipy.optimize


from src.utils import *
from src.qaoa import create_qaoa_circuit

# Optimize QAOA parameters with gradient-based method
def objective_function(params, graph, p):
    gamma = params[:p]
    beta = params[p:]
    circuit = create_qaoa_circuit(graph, p, gamma, beta)
    
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1000)
    measurements = result.measurements['result']
    
    maxcut_value = 0
    for bitstring in measurements:
        cut_value = sum(1 for edge in graph.edges if bitstring[edge[0]] != bitstring[edge[1]])
        maxcut_value += cut_value
    
    return -maxcut_value / 1000

# ### PARAMETER SHIFT RULE
def parameter_shift_rule(params, graph, p, i, shift=np.pi/8):
    params_shifted_forward = np.copy(params)
    params_shifted_forward[i] += shift
    obj_forward = objective_function(params_shifted_forward, graph, p)
    
    params_shifted_backward = np.copy(params)
    params_shifted_backward[i] -= shift
    obj_backward = objective_function(params_shifted_backward, graph, p)
    
    return (obj_forward - obj_backward) / (2 * np.sin(shift))

def gradient(params, graph, p, shift=np.pi/8):
    grad = np.zeros_like(params)
    for i in range(len(params)):
        grad[i] = parameter_shift_rule(params, graph, p, i, shift)
    return grad


def optimize_qaoa(graph: nx.Graph, p: int, method: str):
    initial_params = np.random.uniform(0, np.pi, 2 * p)
    if method == 'gradient':
        result = scipy.optimize.minimize(
            fun=objective_function,
            x0=initial_params,
            args=(graph, p),
            method='BFGS',
            jac=gradient
        )
    else:
        result = scipy.optimize.minimize(
            fun=objective_function,
            x0=initial_params,
            args=(graph, p),
            method='Nelder-Mead'
        )
    return result.x


@click.command()
@click.option('--graph_file', type=click.Path(exists=True), required=True, help="Path for reading the graph.")
@click.option('--optimizer', type=click.Choice(['gradient', 'non-gradient']), required=True, default='non-gradient', help="Optimizer used for parameter optimization.")
@click.option('--depth', required=True, type=int, default=1, help="QAOA circuit depth parameter p.")
@click.option('--out_path', type=str, required=True, help="Destination path to save the optimized parameters list of the respective graphs together with the graphs.")
def main(graph_file, optimizer, depth, out_path):
    # Load graphs from pickle file
    graphs = load_pickle(graph_file)
    
    optimized_parameters = []
    
    for i, graph in enumerate(graphs):
        print(f"Optimizing QAOA for graph {i + 1}/{len(graphs)} with depth {depth} using {optimizer} optimizer", flush=True)
        optimal_params = optimize_qaoa(graph, depth, optimizer)
        optimized_parameters.append(optimal_params)
        print(f"Optimal parameters for graph {i + 1}: {optimal_params}", flush=True)
    

    result_data = {
        'graphs': graphs,
        'optimized_parameters': optimized_parameters,
        'p': depth,
        'optimizer': optimizer
    }
    
    save_pickle(result_data, out_path)

if __name__ == '__main__':
    main()
