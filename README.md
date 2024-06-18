# Variational Quanutm Circuits for MaxCut Problem

This repository implements the QAOA algorithm for solving the MaxCut problem on Erdos-Renyi random graphs.


## Installation

It is recommended to create a new Python virtual environment first: 

```
$ python3 -m venv /path/to/vnevs/quantum_cirq
```

Activate the python environment and install the required packages with the provided `quantum_requirements.txt`file:
```
$ source /path/to/vnevs/quantum_cirq/bin/activate
$ cd /path/to/this/repo
$ pip install -r ./quantum_requirements.txt
```

## Run files

1. **Generate random Graphs**: go to the Jupyter script `main_generate_random_graphs.ipynb`
2. **Optimize QAOA Circuit**: this is done with the python file `main_qaoa.py`. Usage:
    - How to run the file (e.g., using a file for 20-node graphs with 100 elements, the gradient optimizer for a QAOA circuit of depth 1 and saving results disk):
      ```
      $ cd /path/to/this/repo
      $ python main_qaoa.py --graph_file "./data/graphs-20nodes_100elems.pkl" --optimizer "gradient" --depth 1 --out_path "./results/grad_p1_20nodes.pkl"
      ```
    - Help guide:
      ```
      $ python main_qaoa.py --help
      ```

3. **Evaluate circuit**: load an optimized circuit and compare it to a classical MaxCut algorithm, make some plots comparing their performance. This is done in the jupyter notebook `main_evaluate_circuits.ipynb`. 


## Folder structure

- `data`: path where the random graphs are saved. Each file here is a pickled object of Python list containing the graphs.
- `logs`: text log files.
- `results`: here the results from quantum circuit optimizations are saved (output of running `main_qaoa.py`). The structure of the pyckled object is a dictionary with following keys:
    - `p`: QAOA circuit depth.
    - `optimizer`: string, either "gradient" (BFGS) or "non-gradient" (Nelder-Mead).
    - `graphs`: list of graphs.
    - `optimized_parameters`: list of optimal parameters of the quantum circuit (size `2p`).
- `src`: source code for building circuit, utils, and evaluating circuits.
- `plots`: figures after post-processing.