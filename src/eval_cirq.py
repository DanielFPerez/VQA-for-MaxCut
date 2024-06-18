import matplotlib.pyplot as plt
import networkx as nx
import cirq

from src.qaoa import create_qaoa_circuit

# Evaluate the QAOA solution
def evaluate_qaoa(graph, p, gamma, beta, reps=1000):
    circuit = create_qaoa_circuit(graph, p, gamma, beta)
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=reps)
    measurements = result.measurements['result']
    
    maxcut_value = 0
    for bitstring in measurements:
        cut_value = sum(1 for edge in graph.edges if bitstring[edge[0]] != bitstring[edge[1]])
        maxcut_value += cut_value
    
    average_cut_value = maxcut_value / reps
    return average_cut_value

# Classical MaxCut algorithm (Goemans-Williamson approximation)
def classical_algorithm(graph):
    cut_value, partition = nx.algorithms.approximation.maxcut.one_exchange(graph)
    return cut_value

# Compare QAOA with classical algorithm
def compare_qaoa_with_classical(graphs, optimized_parameters, p):
    qaoa_cut_values = []
    classical_cut_values = []
    
    for i, graph in enumerate(graphs):
        gamma, beta = optimized_parameters[i][:p], optimized_parameters[i][p:]
        qaoa_cut_value = evaluate_qaoa(graph, p, gamma, beta)
        qaoa_cut_values.append(qaoa_cut_value)
        
        classical_cut_value = classical_algorithm(graph)
        classical_cut_values.append(classical_cut_value)
    
    return qaoa_cut_values, classical_cut_values

# Plot the results
def plot_results(qaoa_cut_values, classical_cut_values, in_size=(6, 4), in_title='QAOA vs Classical Algorithm for MaxCut'):
    plt.figure(figsize=in_size)
    plt.plot(qaoa_cut_values, label='QAOA', marker='o')
    plt.plot(classical_cut_values, label='Classical', marker='x')
    plt.xlabel('Graph Index')
    plt.ylabel('Cut Value')
    plt.title(in_title)
    plt.legend()
    plt.show()