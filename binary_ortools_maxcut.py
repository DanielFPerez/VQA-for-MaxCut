from enum import Enum
from typing import Dict, List

import click

import networkx as nx
import numpy as np
from ortools.sat.python import cp_model

from utils import create_graph, GRAPHTYPE

class Solution:
    def __init__(self, node_vars: Dict[str, int], graph: nx.Graph):
        self.node_vars = node_vars
        self.graph = graph
        self.__cut = sum(node_vars[u] != node_vars[v] for u, v in self.graph.edges())
    
    @property
    def cut(self):
        return self.__cut

class MaxCutSolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""
    def __init__(self, node_vars, cut_vars, graph):
        cp_model.CpSolverSolutionCallback.__init__(self)
        
        self.__node_vars = node_vars
        self.__cut_vars = cut_vars
        self.__graph = graph

        self.__solution_count = 0
        self.solutions = []

    def OnSolutionCallback(self):
        self.__solution_count += 1
        sol_nodes = {i: self.Value(self.__node_vars[i]) for i in self.__node_vars.keys()}
        solution = Solution(sol_nodes, self.__graph)
        cut = sum(self.Value(cut_var) for cut_var in self.__cut_vars.values()) # division by 2 to get the number of edges
        print(f"Solution {self.__solution_count}: {solution.node_vars}, CUT VALUE: {solution.cut}, SUM_CUT-VARS: {cut}")
        self.solutions.append(solution)

    @property
    def solution_count(self):
        return self.__solution_count


def build_maxcut_model_OTHER(graph: nx.Graph):
    # Create the CP-SAT model
    model = cp_model.CpModel()
    binary_domain = cp_model.Domain.from_values([-1, 1])
    node_vars = {}
    for node in graph.nodes():
        node_vars[node] = model.new_int_var_from_domain(binary_domain, f'node_{node}')
    # Add constraints to maximize the number of edges between the two partitions
    cut_var, prod_uv = {}, {}    
    for u, v in graph.edges():
        cut_var[(u, v)] = model.new_int_var(0, 2, f'cut_{u}_{v}')
        prod_uv[(u,v)] = model.new_int_var_from_domain(binary_domain, f'prod_{u}_{v}')
        model.add_multiplication_equality(prod_uv[(u, v)], [node_vars[u], node_vars[v]])
        model.add(cut_var[(u, v)] == (1 - prod_uv[(u, v)]))
    
    return model, node_vars, cut_var   


def build_maxcut_model(graph: nx.Graph):
    # Create the CP-SAT model
    model = cp_model.CpModel()
    
    node_vars = {}
    for node in graph.nodes():
        node_vars[node] = model.new_bool_var(f'node_{node}')

    
    # Add constraints to maximize the number of edges between the two partitions
    prod_uv = {}
    cut_var = {}    
    for u, v in graph.edges():
        cut_var[(u,v)] = model.new_bool_var(f'cut_{u}_{v}')
        prod_uv[(u, v)] = model.new_int_var(0, 2, f"prod_{u}_{v}")
        model.add_multiplication_equality(prod_uv[(u, v)], [node_vars[u], node_vars[v]])
        model.add(cut_var[(u, v)] == (node_vars[u] + node_vars[v] - 2 * prod_uv[(u, v)]))
    
    return model, node_vars, cut_var   


def get_optimum_obj_value(graph: nx.Graph) -> int:
    """Get the optimum value for the maxcut problem
    Args:   
        graph (nx.Graph): graph
    Returns:
        int: optimum value
    """
    model, node_vars, cut_var = build_maxcut_model(graph)
    model.maximize(sum(cut_var.values()))

    tmp_solver = cp_model.CpSolver()
    tmp_status = tmp_solver.Solve(model)
    print(f"\TMP Status: {tmp_solver.StatusName(tmp_status)}")
    
    if tmp_status == cp_model.OPTIMAL:
        print(f"TMP OPTIMUM Objective value: {tmp_solver.ObjectiveValue()}")
        print()
        return int(tmp_solver.ObjectiveValue())
    else:
        print()
        return None
    

@click.command()
@click.option("--graph_type", type=click.Choice(GRAPHTYPE, case_sensitive=False), default='CUSTOM', help="Type of graph to create")
@click.option("--n_nodes", type=int, default=4, help="Number of nodes in the graph")
def solve_maxcut(graph_type: str, n_nodes: int):
    """Given a nx.Graph, solve the maxcut problem and deliver all solutions
    Args:
        graph (nx.Graph): graph
    """
    # Create a graph
    print(f"Graph type: {graph_type}")

    graph = create_graph(graph_type, n_nodes, verbose=True)

    # Get the optimum value for the maxcut problem
    opt_value = get_optimum_obj_value(graph)
    print(f"Optimum value: {opt_value}")

    # Create the CP-SAT model
    model, node_vars, cut_var = build_maxcut_model(graph)

    # Add constraints to maximize the number of edges between the two partitions
    obj_value = model.new_int_var(-2*len(graph.edges()), 2*len(graph.edges()), 'obj_value')
    model.add(obj_value == sum(cut_var.values()))
    model.add(obj_value == opt_value)
    
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solution_printer = MaxCutSolutionPrinter(node_vars, cut_var, graph)
    status = solver.solve(model, solution_printer)
    print(f"Status: {solver.StatusName(status)}")

    # If an optimal solution is found, print the results
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:   
        print(f'Number of solutions found: {solution_printer.solution_count}')
        # print("All solutions:")
        # for solution in solution_printer.solutions:
        #     print(solution)
        return solution_printer.solutions
    else:
        print("No solution found.")
        print(f"Status: {solver.StatusName(status)}")
        return None


if __name__ == "__main__":
    
    # Solve the MaxCut problem
    all_solutions = solve_maxcut()
    print(f"These are the solutions:  {all_solutions}")
    if len(all_solutions): 
        print("All feasible solutions:")
        for solution in all_solutions:
            print(solution)


    # print('\nStatistics')
    # print(f'  conflicts      : {solver.NumConflicts()}')
    # print(f'  branches       : {solver.NumBranches()}')
    # print(f'  wall time      : {solver.WallTime()} s')
    # print(f'  solutions found: {solution_printer.solution_count}')
    