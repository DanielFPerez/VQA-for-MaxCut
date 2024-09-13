import networkx as nx
from ortools.sat.python import cp_model 

class MaxCutSolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""
    def __init__(self, node_vars, cut_vars, graph):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__node_vars = node_vars
        self.__cut_vars = cut_vars
        self.__graph = graph
        self.__solution_count = 0

    def OnSolutionCallback(self):
        self.__solution_count += 1
        sol_nodes = {i: self.Value(self.__node_vars[i]) for i in self.__node_vars.keys()}
        cut = sum(sol_nodes[u] != sol_nodes[v] for u, v in self.__graph.edges())
        cut_vars_sum = sum(self.Value(cut_var) for cut_var in self.__cut_vars.values())
        print(f"Solution {self.__solution_count}: {sol_nodes}, CUT VALUE: {cut}, SUM CUT_VARS: {cut_vars_sum}")

    @property
    def solution_count(self):
        return self.__solution_count


graph = nx.Graph()
graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
print("Nodes of the graph:", graph.nodes())
print("Edges of the graph:", graph.edges())

model = cp_model.CpModel()

node_vars = {}
for node in graph.nodes():
    node_vars[node] = model.new_bool_var(f'node_{node}')

# Add constraints to maximize the number of edges between the two partitions
cut_var, prod_uv = {}, {}
for u, v in graph.edges():
    cut_var[(u,v)] = model.new_bool_var(f'cut_{u}_{v}')
    
    # ### OPTION 1
    # model.Add(node_vars[u] != node_vars[v]).OnlyEnforceIf(cut_var[(u, v)])
    # model.Add(node_vars[u] == node_vars[v]).OnlyEnforceIf(cut_var[(u, v)].Not())

    # ### OPTION 2 
    prod_uv[(u, v)] = model.new_int_var(0, 2, f"prod_{u}_{v}")
    model.add_multiplication_equality(prod_uv[(u, v)], [node_vars[u], node_vars[v]])
    model.add(cut_var[(u, v)] == (node_vars[u] + node_vars[v] - 2 * prod_uv[(u, v)]))

obj_value = model.new_int_var(1, len(graph.edges()), 'obj_value')
model.add(obj_value == sum(cut_var.values()))

model.maximize(obj_value)

solver = cp_model.CpSolver()
solver.parameters.enumerate_all_solutions = True
solution_printer = MaxCutSolutionPrinter(node_vars, cut_var, graph)
status = solver.solve(model, solution_printer)

if status == cp_model.OPTIMAL:   
    print(f'Number of solutions found: {solution_printer.solution_count}')
else:
    print("No solution found.")
    print(f"Status: {solver.StatusName(status)}")