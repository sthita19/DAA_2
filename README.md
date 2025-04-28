# Densest Subgraph Algorithms for h-Cliques

## Project Website

For more information or to access the hosted version of the project, visit the [Project Website](https://sthita19.github.io/DAA_2/).
This is the github link for codes and other stuff visit the [github repository](https://github.com/sthita19/DAA_2/)

## Algorithm Overview

The goal of this project is to solve the **Densest Subgraph Problem for h-cliques**, which involves finding the subgraph with the maximum ratio of h-cliques to vertices. We provide two algorithms that efficiently compute densest subgraphs using different approaches.

### 1. Algorithm1: Parametric Flow Approach

The **Parametric Flow Approach** solves the Densest Subgraph Problem by finding the optimal density using a binary search method. This is done by constructing a flow network and using the **Edmonds-Karp algorithm** to compute the maximum flow. The minimum cut identifies the densest subgraph.

#### File Descriptions

- **`Preprocess.cpp`**: A utility to preprocess raw graph datasets by remapping node IDs, detecting graph directionality, removing duplicate edges and self-loops, and creating clean datasets.
- **`Algorithm1.cpp`**: The main algorithm implementation that parses cleaned datasets, finds h-cliques using optimized algorithms for h=2 and h=3, and calculates the densest subgraph using the flow-based approach.
- **`Algorithm1.exe`**: Executable for running the algorithm.
- **Datasets**: Graph datasets in `.txt` format (e.g., `As-733.txt`, `As-Caida.txt`, `Ca-HepTh.txt`).
- **Output Folder**: Contains results for different values of h with the densest subgraph details.

#### Execution Steps

1. **Preprocessing**:

   - Run `Preprocess.exe` to create cleaned versions of datasets.
   - Usage: `Preprocess.exe [dataset_files]` (If no files are specified, default datasets will be processed.)
   - Outputs `*-clean.txt` and `*-mapping.txt`.

2. **Running the Algorithm**:
   - Run `Algorithm1.exe` with a dataset and an h-value to determine the densest subgraph.
   - Example: `Algorithm1.exe 3 As-733-clean.txt` (h-value represents the clique size, e.g., h=2 for edges, h=3 for triangles).
   - Results are written to `Output/<dataset>-output-h<h_value>.txt`.

---

### 2. Algorithm4: CoreExact with Core Decomposition

**CoreExact** enhances the Densest Subgraph Problem solution by combining **core decomposition** with a flow-based exact algorithm. The algorithm performs **k-ψ core decomposition** to filter the graph based on clique density, reducing the search space. This is followed by a flow-based parametric approach to find the densest subgraph.

#### File Descriptions

- **`Preprocess.cpp`**: Same as in Algorithm1, remapping nodes, cleaning datasets, and creating mapping files.
- **`Algorithm4.cpp` (CoreExact)**: Implements the CoreExact algorithm, using core decomposition and pruning to improve performance. This algorithm applies a parametric flow approach on the filtered graph components to find the densest subgraph.
- **`Algorithm4.exe`**: Executable for the CoreExact algorithm.
- **`Algorithm3.cpp`**: Implements the k-ψ core decomposition used by CoreExact.
- **Datasets**: Same as in Algorithm1, with clean datasets and mapping files.
- **Output Folder**: Contains results from CoreExact with densest subgraph information.

#### Execution Steps

1. **Preprocessing**:

   - Run `Preprocess.exe` to clean datasets (same as in Algorithm1).
   - Outputs `*-clean.txt` and `*-mapping.txt`.

2. **Running CoreExact (Algorithm4)**:
   - Run `Algorithm4.exe` with a dataset, h-value, pruning threshold, and optional parameters for graph size.
   - Example: `Algorithm4.exe 3 As-733-clean.txt 0.7` (where 0.7 is the pruning threshold).
   - Results are written to `Output/<dataset>-coreexact-h<h_value>.txt`.

---

## Features of Both Algorithms

### Preprocessing

- Handles both directed and undirected graphs.
- Removes duplicate edges and self-loops.
- Creates mapping files for tracing results back to the original graph.
- Standardizes input format for both algorithms.

### Algorithm Optimizations

- **Algorithm1** uses a flow-based parametric approach with binary search to find optimal subgraphs.
- **Algorithm4 (CoreExact)** applies core decomposition to filter the graph, significantly improving performance on large graphs, and includes configurable pruning to balance speed and solution quality.

### Output Format

Both algorithms output:

- Dataset information and h-value
- Graph statistics (node count, edge count, directed/undirected)
- Final density, execution runtime, and vertices in the densest subgraph (both remapped and original node IDs).

---

## Group Contributions

| Name                | Id            | Contribution                                                                                             |
| ------------------- | ------------- | -------------------------------------------------------------------------------------------------------- |
| Sthitaprajna        | 2021B3A71082H | Contributed to the implementation of the **CORE-EXACT** algorithm. Implemented the **OUTPUT** algorithm. |
| Kushagra Mishra     | 2021B5A72970H | Contributed to the implementation of the **OUTPUT** algorithm. Authored the report                       |
| Riya Agrawal        | 2021B3A70996H | Contributed to the implementation of the **CORE-EXACT** algorithm. Helped develop and host the website   |
| Dhruv Choudhary     | 2021B3A73142H | Contributed to the implementation of the **OUTPUT** algorithm. Authored the report                       |
| Waleed Iqbal Shaikh | 2021B3A70559H | Contributed to the implementation of the **OUTPUT** algorithm. Helped develop and host the website       |

---

---

This version includes a clean integration of both algorithms, their file descriptions, execution steps, and group contributions, with an added link to the project website.
