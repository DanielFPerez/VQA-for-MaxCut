import cirq
import networkx as nx
from typing import List

# Define QAOA circuit
def maxcut_cost_operator(graph: nx.Graph, qubits, gamma: float):
    cost_ops = []
    for u, v in graph.edges:
        qubit_u = qubits[u]
        qubit_v = qubits[v]
        cost_ops.append(cirq.ZZ(qubit_u, qubit_v) ** gamma)
    return cost_ops

def mixing_operator(graph: nx.Graph, qubits, beta: float):
    mix_ops = []
    for node in graph.nodes:
        qubit = qubits[node]
        mix_ops.append(cirq.X(qubit) ** beta)
    return mix_ops

def create_qaoa_circuit(graph: nx.Graph, p: int, gamma: List[float], beta: List[float]) -> cirq.Circuit:
    qubits = cirq.LineQubit.range(len(graph.nodes))
    circuit = cirq.Circuit()
    
    # Initialize in superposition
    circuit.append(cirq.H.on_each(qubits))
    
    for i in range(p):
        circuit.append(maxcut_cost_operator(graph, qubits, gamma[i]))
        circuit.append(mixing_operator(graph, qubits, beta[i]))

    # Add measurements
    circuit.append(cirq.measure(*qubits, key='result'))
    
    return circuit



# ### FINITE DIFERENCE GRADIENT
# def gradient(params, graph, p, epsilon=1e-4):
#     grad = np.zeros_like(params)
#     for i in range(len(params)):
#         params_shifted = np.copy(params)
#         params_shifted[i] += epsilon
#         obj_plus = objective_function(params_shifted, graph, p)
        
#         params_shifted[i] -= 2 * epsilon
#         obj_minus = objective_function(params_shifted, graph, p)
        
#         grad[i] = (obj_plus - obj_minus) / (2 * epsilon)
    
#     return grad