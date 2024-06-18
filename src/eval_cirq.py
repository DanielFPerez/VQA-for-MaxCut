import matplotlib.pyplot as plt
import networkx as nx
import cirq
import numpy as np

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
def classical_algorithm(graph, seed:int = 42):
    cut_value, partition = nx.algorithms.approximation.maxcut.one_exchange(graph, seed=seed)
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
    fig, ax = plt.subplots(figsize=in_size)

    ax.plot(qaoa_cut_values, label='QAOA', marker='o')
    ax.plot(classical_cut_values, label='Classical', marker='x')
    ax.set_xlabel('Graph Index')
    ax.set_ylabel('Cut Value')
    ax.set_title(in_title)
    ax.legend()
    
    return fig


def compare_qaoa_with_classical_multi_p(graphs, optimized_parameters_list, depths):
    qaoa_results = {depth: [] for depth in depths}
    classical_results = []

    for i, graph in enumerate(graphs):
        classical_cut_value = classical_algorithm(graph)
        classical_results.append(classical_cut_value)

        for depth, optimized_parameters in zip(depths, optimized_parameters_list):
            gamma, beta = optimized_parameters[i][:depth], optimized_parameters[i][depth:]
            qaoa_cut_value = evaluate_qaoa(graph, depth, gamma, beta)
            qaoa_results[depth].append(qaoa_cut_value)

    return qaoa_results, classical_results


def plot_multiple_p_results(qaoa_results, classical_results, depths, in_size=(10, 6), title='QAOA vs Classical Algorithm for MaxCut'):
    fig, ax = plt.subplots(figsize=in_size)
    
    for depth in depths:
        ax.plot(qaoa_results[depth], label=f'QAOA p={depth}', marker='o')
    
    ax.plot(classical_results, label='Classical', marker='x')
    ax.set_xlabel('Graph Index')
    ax.set_ylabel('Cut Value')
    ax.set_title(title)
    ax.legend()
    return fig

    
def normalize_qaoa_to_classical(graphs, optimized_parameters_list, depths):
    normalized_results = {depth: [] for depth in depths}
    classical_results = []

    for i, graph in enumerate(graphs):
        classical_cut_value = classical_algorithm(graph)
        classical_results.append(classical_cut_value)

        for depth, optimized_parameters in zip(depths, optimized_parameters_list):
            gamma, beta = optimized_parameters[i][:depth], optimized_parameters[i][depth:]
            qaoa_cut_value = evaluate_qaoa(graph, depth, gamma, beta)
            normalized_value = qaoa_cut_value / classical_cut_value
            normalized_results[depth].append(normalized_value)

    return normalized_results, classical_results


def plot_normalized_bar_chart(normalized_results, depths, in_size=(10, 6), title='Normalized MaxCut Values and Standard Deviations'):
    mean_values = []
    std_values = []
    labels = ['Classical'] + [f'QAOA p={depth}' for depth in depths]

    # Calculate mean and standard deviation for each depth
    for depth in depths:
        mean_values.append(np.mean(normalized_results[depth]))
        std_values.append(np.std(normalized_results[depth]))

    # Add the classical method bar
    mean_values.insert(0, 1.0)
    std_values.insert(0, 0.0)

    # Plot the bar chart
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=in_size)
    ax.bar(x, mean_values, yerr=std_values, capsize=5)
    ax.set_xticks(x)
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Method')
    ax.set_ylabel('Normalized Cut Value')
    ax.set_title(title)
    ax.grid(axis='y')

    return fig


