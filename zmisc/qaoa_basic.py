import cirq
import numpy as np
import scipy.optimize
from src.utils import load_pickle

# read random graphs from path
src_dir = "./data/graphs-10nodes_100elems.pkl"
random_graphs = load_pickle(src_dir)


# Define QAOA circuit
def maxcut_cost_operator(graph: nx.Graph, gamma: float):
    cost_ops = []
    for u, v in graph.edges:
        qubit_u = cirq.LineQubit(u)
        qubit_v = cirq.LineQubit(v)
        cost_ops.append(cirq.ZZ(qubit_u, qubit_v) ** gamma)
    return cost_ops

def mixing_operator(graph: nx.Graph, beta: float):
    mix_ops = []
    for node in graph.nodes:
        qubit = cirq.LineQubit(node)
        mix_ops.append(cirq.X(qubit) ** beta)
    return mix_ops

def create_qaoa_circuit(graph: nx.Graph, p: int, gamma: List[float], beta: List[float]) -> cirq.Circuit:
    qubits = cirq.LineQubit.range(len(graph.nodes))
    circuit = cirq.Circuit()
    
    # Initialize in superposition
    circuit.append(cirq.H.on_each(qubits))
    
    for i in range(p):
        circuit.append(maxcut_cost_operator(graph, gamma[i]))
        circuit.append(mixing_operator(graph, beta[i]))

    # Add measurements
    circuit.append(cirq.measure(*qubits, key='result'))
    
    return circuit

# Optimize QAOA parameters
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

def optimize_qaoa(graph: nx.Graph, p: int):
    initial_params = np.random.uniform(0, np.pi, 2 * p)
    result = scipy.optimize.minimize(objective_function, initial_params, args=(graph, p), method='Nelder-Mead')
    return result.x

# Run optimization for each graph
p = 1  # Number of layers in QAOA, adjust as needed
optimized_parameters = []

for i, graph in enumerate(random_graphs):
    print(f"Optimizing QAOA for graph {i + 1}/{num_graphs}")
    optimal_params = optimize_qaoa(graph, p)
    optimized_parameters.append(optimal_params)
    print(f"Optimal parameters for graph {i + 1}: {optimal_params}")