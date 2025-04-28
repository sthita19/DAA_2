#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <queue>
#include <limits>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>  // For setprecision
#include <cmath>    // For isfinite
#include <functional> // For std::function
#include <unordered_map> // For node ID mapping

using namespace std;
using namespace std::chrono;

// Type aliases for clarity
using Graph = vector<vector<int>>;
using Clique = vector<int>;

const double EPSILON = 1e-9;  // For double comparisons
const double INF = numeric_limits<double>::infinity();

// Function for detailed logging
void log(const string& message) {
    cout << message << endl;
    cout.flush(); // Ensure output appears immediately
}

// Parse cleaned dataset files
bool parseCleanDataset(const string& filename, Graph& adj, int& numNodes, int& numEdges, bool& isDirected) {
    ifstream file(filename);
    if (!file.is_open()) {
        log("Error: Cannot open file " + filename);
        return false;
    }

    log("Parsing dataset: " + filename);
    
    string line;
    
    // Parse metadata line
    if (getline(file, line)) {
        if (line[0] == '#') {
            istringstream ss(line);
            string temp;
            
            // Parse "# nodes: X edges: Y directed: Z"
            ss >> temp; // #
            ss >> temp; // nodes:
            ss >> numNodes;
            ss >> temp; // edges:
            ss >> numEdges;
            ss >> temp; // directed:
            int dirFlag;
            ss >> dirFlag;
            isDirected = (dirFlag == 1);
            
            log("  Metadata: " + to_string(numNodes) + " nodes, " + 
                to_string(numEdges) + " edges, " + 
                (isDirected ? "directed" : "undirected"));
        }
    }
    
    if (numNodes <= 0) {
        log("Error: Invalid node count in file");
        return false;
    }
    
    // Initialize adjacency list
    adj.resize(numNodes);
    
    // Reserve space to reduce reallocations
    for (auto& neighbors : adj) {
        neighbors.reserve(50); // Reserve for average degree
    }
    
    // Parse edges
    int edgeCount = 0;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        istringstream ss(line);
        int from, to;
        if (ss >> from >> to) {
            if (from >= 0 && from < numNodes && to >= 0 && to < numNodes) {
                adj[from].push_back(to);
                // Always add the reverse edge for Algorithm 1, since it requires undirected graph
                adj[to].push_back(from);
                edgeCount++;
            } else {
                log("Warning: Edge (" + to_string(from) + "," + to_string(to) + 
                    ") refers to nodes outside valid range [0," + to_string(numNodes-1) + "]");
            }
        }
    }
    
    // Sort adjacency lists and remove duplicates
    for (auto& neighbors : adj) {
        sort(neighbors.begin(), neighbors.end());
        neighbors.erase(unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }
    
    // We still read the isDirected flag for informational purposes, but internally 
    // we always build an undirected graph as required by Algorithm 1
    log("Successfully parsed " + to_string(numNodes) + " nodes and " + 
        to_string(edgeCount) + " edges into an undirected graph representation");
    
    return true;
}

// Load node ID mapping from file
unordered_map<int, int> loadNodeMapping(const string& filename) {
    unordered_map<int, int> mapping;
    string mappingFile = filename.substr(0, filename.find("-clean")) + "-mapping.txt";
    ifstream mapFile(mappingFile);
    
    if (!mapFile.is_open()) {
        log("Note: No mapping file found at " + mappingFile);
        return mapping;
    }
    
    log("Loading node ID mapping from " + mappingFile);
    
    string line;
    int lineCount = 0;
    
    while (getline(mapFile, line)) {
        lineCount++;
        if (line.empty() || line[0] == '#') continue;
        
        istringstream ss(line);
        int remappedId;
        int originalId;
        char comma;
        
        if (ss >> remappedId >> comma >> originalId) {
            mapping[remappedId] = originalId;
        } else {
            log("Warning: Invalid format in mapping file at line " + to_string(lineCount));
        }
    }
    
    log("Loaded " + to_string(mapping.size()) + " node mappings");
    return mapping;
}

// Function to check if two vertices are connected
bool areConnected(int u, int v, const Graph& adj) {
    if (u < 0 || u >= adj.size() || v < 0 || v >= adj.size()) return false;
    return binary_search(adj[u].begin(), adj[u].end(), v);
}

// Find all edges (h=2 cliques) in the graph
vector<Clique> findEdges(const Graph& adj) {
    vector<Clique> edges;
    int n = adj.size();
    edges.reserve(n * 5); // Reserve space for average degree * nodes
    
    auto start_time = high_resolution_clock::now();
    int progress_interval = max(1, n / 20); // Report progress 20 times
    
    log("Finding all edges (h=2 cliques)...");
    
    for (int u = 0; u < n; ++u) {
        for (int v : adj[u]) {
            if (u < v) { // Count each edge only once for undirected graph
                edges.push_back({u, v});
            }
        }
        
        // Progress reporting
        if (u % progress_interval == 0 || u == n - 1) {
            auto current_time = high_resolution_clock::now();
            double elapsed = duration_cast<milliseconds>(current_time - start_time).count() / 1000.0;
            double percentage = (u + 1) * 100.0 / n;
            log("  Progress: " + to_string(static_cast<int>(percentage)) + "% (" + 
                to_string(u + 1) + "/" + to_string(n) + " nodes, " + 
                to_string(edges.size()) + " edges found, " +
                to_string(elapsed) + " seconds)");
        }
    }
    
    auto end_time = high_resolution_clock::now();
    double total_time = duration_cast<milliseconds>(end_time - start_time).count() / 1000.0;
    log("Found " + to_string(edges.size()) + " edges in " + to_string(total_time) + " seconds");
    
    return edges;
}

// Find all triangles (h=3 cliques) in the graph using an optimized algorithm
vector<Clique> findTriangles(const Graph& adj) {
    vector<Clique> triangles;
    int n = adj.size();
    
    auto start_time = high_resolution_clock::now();
    int progress_interval = max(1, n / 20); // Report progress 20 times
    
    log("Finding all triangles (h=3 cliques)...");
    log("This may take some time for large graphs.");
    
    // Track progress
    int nodes_processed = 0;
    int triangles_found = 0;
    
    // Find all triangles (u,v,w) where u < v < w
    for (int u = 0; u < n; ++u) {
        const auto& u_neighbors = adj[u];
        
        // For each pair of neighbors of u
        for (size_t i = 0; i < u_neighbors.size(); ++i) {
            int v = u_neighbors[i];
            if (v <= u) continue; // Ensure u < v
            
            for (size_t j = i + 1; j < u_neighbors.size(); ++j) {
                int w = u_neighbors[j];
                if (w <= v) continue; // Ensure v < w
                
                // Check if v and w are connected to form a triangle
                if (areConnected(v, w, adj)) {
                    triangles.push_back({u, v, w});
                    triangles_found++;
                }
            }
        }
        
        nodes_processed++;
        
        // Progress reporting
        if (nodes_processed % progress_interval == 0 || nodes_processed == n) {
            auto current_time = high_resolution_clock::now();
            double elapsed = duration_cast<milliseconds>(current_time - start_time).count() / 1000.0;
            double percentage = nodes_processed * 100.0 / n;
            
            log("  Progress: " + to_string(static_cast<int>(percentage)) + "% (" + 
                to_string(nodes_processed) + "/" + to_string(n) + " nodes, " + 
                to_string(triangles_found) + " triangles found, " +
                to_string(elapsed) + " seconds)");
            
            // Estimate remaining time
            if (percentage > 0 && elapsed > 0) {
                double estimated_total = elapsed * 100.0 / percentage;
                double remaining = estimated_total - elapsed;
                log("  Estimated time remaining: " + to_string(static_cast<int>(remaining)) + " seconds");
            }
        }
    }
    
    auto end_time = high_resolution_clock::now();
    double total_time = duration_cast<milliseconds>(end_time - start_time).count() / 1000.0;
    log("Found " + to_string(triangles.size()) + " triangles in " + to_string(total_time) + " seconds");
    
    return triangles;
}

// Flow network structures
struct FlowEdge {
    int to;
    double capacity;
    int reverseEdgeIndex;
};

using FlowGraph = vector<vector<FlowEdge>>;

// Add an edge to the flow network
void addFlowEdge(int u, int v, double capacity, FlowGraph& flow) {
    flow[u].push_back({v, capacity, static_cast<int>(flow[v].size())});
    flow[v].push_back({u, 0.0, static_cast<int>(flow[u].size() - 1)});
}

// Find augmenting path using BFS
bool findAugmentingPath(int s, int t, const FlowGraph& flow, vector<int>& parentEdge, vector<int>& parentNode) {
    fill(parentEdge.begin(), parentEdge.end(), -1);
    fill(parentNode.begin(), parentNode.end(), -1);
    parentNode[s] = -2;

    queue<pair<int, double>> q;
    q.push({s, INF});

    while (!q.empty()) {
        int u = q.front().first;
        double flow_val = q.front().second;
        q.pop();

        for (int i = 0; i < flow[u].size(); ++i) {
            const auto& edge = flow[u][i];
            if (parentNode[edge.to] == -1 && edge.capacity > EPSILON) {
                parentNode[edge.to] = u;
                parentEdge[edge.to] = i;
                double new_flow = min(flow_val, edge.capacity);
                if (edge.to == t) return true;
                q.push({edge.to, new_flow});
            }
        }
    }
    return false;
}

// Compute maximum flow using Edmonds-Karp algorithm
double computeMaxFlow(int s, int t, FlowGraph& flow) {
    double total_flow = 0;
    vector<int> parentEdge(flow.size());
    vector<int> parentNode(flow.size());

    while (findAugmentingPath(s, t, flow, parentEdge, parentNode)) {
        double path_flow = INF;
        
        // Find bottleneck capacity
        for (int v = t; v != s; v = parentNode[v]) {
            int u = parentNode[v];
            int edge_idx = parentEdge[v];
            path_flow = min(path_flow, flow[u][edge_idx].capacity);
        }

        if (!isfinite(path_flow) || path_flow <= EPSILON) {
            log("Warning: Invalid path flow detected");
            break;
        }

        // Update residual capacities
        for (int v = t; v != s; v = parentNode[v]) {
            int u = parentNode[v];
            int edge_idx = parentEdge[v];
            int rev_edge_idx = flow[u][edge_idx].reverseEdgeIndex;

            flow[u][edge_idx].capacity -= path_flow;
            flow[v][rev_edge_idx].capacity += path_flow;
        }
        
        total_flow += path_flow;
    }
    
    return total_flow;
}

// Find the nodes on the source side of the min-cut
vector<int> findMinCut(int s, int t, const FlowGraph& flow) {
    vector<bool> visited(flow.size(), false);
    vector<int> min_cut_nodes;
    queue<int> q;

    q.push(s);
    visited[s] = true;
    min_cut_nodes.push_back(s);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (const auto& edge : flow[u]) {
            if (edge.capacity > EPSILON && !visited[edge.to]) {
                visited[edge.to] = true;
                q.push(edge.to);
                min_cut_nodes.push_back(edge.to);
            }
        }
    }
    
    return min_cut_nodes;
}

// Find all cliques of size h in the graph (for h > 3)
// This is a fallback function that's less optimized
vector<Clique> findCliquesGeneral(int h, const Graph& adj) {
    vector<Clique> cliques;
    Clique current;
    
    // Helper recursive function for finding cliques
    function<void(int)> findCliquesRecursive = [&](int start) {
        // Base case: if we have a clique of size h
        if (current.size() == h) {
            cliques.push_back(current);
            return;
        }
        
        // Try adding each vertex starting from 'start'
        for (int v = start; v < adj.size(); ++v) {
            // Check if v can be added to the current clique
            bool canAdd = true;
            for (int u : current) {
                if (!areConnected(u, v, adj)) {
                    canAdd = false;
                    break;
                }
            }
            
            if (canAdd) {
                current.push_back(v);
                findCliquesRecursive(v + 1);
                current.pop_back(); // Backtrack
            }
        }
    };
    
    log("Finding all " + to_string(h) + "-cliques using generic algorithm...");
    log("Note: This could be very slow for h > 3 and large graphs");
    
    auto start_time = high_resolution_clock::now();
    findCliquesRecursive(0);
    auto end_time = high_resolution_clock::now();
    
    double total_time = duration_cast<milliseconds>(end_time - start_time).count() / 1000.0;
    log("Found " + to_string(cliques.size()) + " " + to_string(h) + "-cliques in " + 
        to_string(total_time) + " seconds");
    
    return cliques;
}

// Find all h-cliques, dispatching to specialized functions for h=2 and h=3
vector<Clique> findAllCliques(int h, const Graph& adj) {
    if (h == 2) {
        return findEdges(adj);
    } else if (h == 3) {
        return findTriangles(adj);
    } else {
        return findCliquesGeneral(h, adj);
    }
}

// Main algorithm: Exact Densest Subgraph for h-Cliques
// Returns {subgraph vertices, density, total h-cliques count}
tuple<vector<int>, double, int> findDensestSubgraph(const Graph& adj, int h) {
    int n = adj.size();
    if (n == 0 || h < 2) {
        log("Error: Empty graph or invalid h value");
        return {{}, 0.0, 0};
    }

    // Find all h-cliques
    auto start_clique = high_resolution_clock::now();
    vector<Clique> h_cliques = findAllCliques(h, adj);
    auto end_clique = high_resolution_clock::now();
    double clique_time = duration_cast<milliseconds>(end_clique - start_clique).count() / 1000.0;
    
    int totalCliques = h_cliques.size();
    log("Found " + to_string(totalCliques) + " " + to_string(h) + "-cliques in " + 
        to_string(clique_time) + " seconds");

    if (h_cliques.empty()) {
        log("No " + to_string(h) + "-cliques found in the graph");
        return {{}, 0.0, 0};
    }

    // Calculate clique degrees for each vertex
    vector<double> cliqueDegrees(n, 0.0);
    double maxCliqueDegree = 0.0;
    
    for (const auto& clique : h_cliques) {
        for (int vertex : clique) {
            cliqueDegrees[vertex]++;
            maxCliqueDegree = max(maxCliqueDegree, cliqueDegrees[vertex]);
        }
    }
    
    log("Maximum " + to_string(h) + "-clique degree: " + to_string(maxCliqueDegree));

    // Find (h-1)-cliques if h > 2
    vector<Clique> h_minus_1_cliques;
    map<Clique, int> h_minus_1_clique_map;
    
    if (h > 2) {
        auto start_small = high_resolution_clock::now();
        h_minus_1_cliques = findAllCliques(h - 1, adj);
        auto end_small = high_resolution_clock::now();
        double small_time = duration_cast<milliseconds>(end_small - start_small).count() / 1000.0;
        
        log("Found " + to_string(h_minus_1_cliques.size()) + " " + to_string(h-1) + 
            "-cliques in " + to_string(small_time) + " seconds");
        
        // Map each (h-1)-clique to its index in the flow network
        log("Building clique mapping...");
        for (int i = 0; i < h_minus_1_cliques.size(); ++i) {
            h_minus_1_clique_map[h_minus_1_cliques[i]] = i + n + 2;
        }
    }

    // Binary search for the optimal density
    double low = 0.0, high = maxCliqueDegree;
    vector<int> bestSubgraph;
    double precision = 1.0 / (n * (n-1) + 1);
    int iteration = 0;
    int maxIterations = 30;
    
    log("Starting binary search for optimal density...");
    
    while (high - low > precision && iteration < maxIterations) {
        double mid = low + (high - low) / 2.0;
        log("  Iteration " + to_string(++iteration) + ": Testing density " + 
            to_string(mid) + " (range [" + to_string(low) + ", " + to_string(high) + "])");

        // Build flow network
        int flow_size = 2 + n + (h > 2 ? h_minus_1_cliques.size() : 0);
        FlowGraph flow(flow_size);
        
        // Pre-allocate capacity in flow adjacency lists
        for (auto& edges : flow) {
            edges.reserve(20); // Reasonable capacity estimate
        }
        
        int s = 0, t = 1;

        // Add edges s -> v with capacity = clique_degree(v)
        for (int i = 0; i < n; ++i) {
            if (cliqueDegrees[i] > EPSILON) {
                addFlowEdge(s, i + 2, cliqueDegrees[i], flow);
            }
        }

        // Add edges v -> t with capacity = mid * h (our density guess)
        for (int i = 0; i < n; ++i) {
            addFlowEdge(i + 2, t, mid * h, flow);
        }

        // For h > 2, add additional edges to represent clique structure
        if (h > 2) {
            log("    Building clique structure edges...");
            set<pair<int, int>> added_edges;
            int edges_added = 0;
            
            for (const auto& h_clique : h_cliques) {
                for (int i = 0; i < h; ++i) {
                    int v = h_clique[i];
                    
                    // Create an (h-1)-clique by removing vertex v
                    Clique h_minus_1;
                    for (int j = 0; j < h; ++j) {
                        if (i != j) h_minus_1.push_back(h_clique[j]);
                    }
                    sort(h_minus_1.begin(), h_minus_1.end());
                    
                    // Add edge v -> (h-1)-clique with capacity 1
                    if (h_minus_1_clique_map.count(h_minus_1)) {
                        int psi_node = h_minus_1_clique_map[h_minus_1];
                        int v_node = v + 2;
                        
                        // Avoid duplicate edges
                        pair<int, int> edge_pair(v_node, psi_node);
                        if (added_edges.find(edge_pair) == added_edges.end()) {
                            addFlowEdge(v_node, psi_node, 1.0, flow);
                            added_edges.insert(edge_pair);
                            edges_added++;
                        }
                    }
                }
            }
            log("    Added " + to_string(edges_added) + " clique structure edges");
        }

        // Compute max-flow
        auto start_flow = high_resolution_clock::now();
        log("    Computing max flow...");
        double flow_value = computeMaxFlow(s, t, flow);
        auto end_flow = high_resolution_clock::now();
        double flow_time = duration_cast<milliseconds>(end_flow - start_flow).count() / 1000.0;
        
        log("    Max flow: " + to_string(flow_value) + " (computed in " + 
            to_string(flow_time) + " seconds)");

        // --- NEW FEASIBILITY CHECK BASED ON FLOW VALUE ---
        double totalSourceCapacity = 0.0;
        if (h == 2) {
            // For Goldberg network: Compare flow_value against m (total edges)
            totalSourceCapacity = static_cast<double>(totalCliques); // m = totalCliques when h=2
        } else {
            // For h > 2 general network: Compare flow_value against sum of cliqueDegrees
            for(int i=0; i<n; ++i) { // Sum over ALL n potential nodes
                totalSourceCapacity += cliqueDegrees[i];
            }
        }
        log("    Total source capacity calculated: " + to_string(totalSourceCapacity));

        // The check: Compare flow value to source capacity
        if (flow_value >= totalSourceCapacity - EPSILON) { // Use EPSILON for float comparison
             // If flow equals source capacity, it implies no subgraph with density >= mid exists
             high = mid;
             log("    No subgraph with density >= " + to_string(mid) + " exists. Reducing density bound to " + to_string(high));
        } else {
             // Flow is less than source capacity, indicating a non-trivial cut exists
             low = mid;

             // Find the actual min-cut nodes from the *residual graph* after max flow
             log("    Finding min cut nodes from residual graph...");
             // **Ensure computeMaxFlow returned the residual graph or was called on a copy**
             // Assuming 'flow' is the residual graph AFTER max-flow computation:
             vector<int> min_cut_nodes = findMinCut(s, t, flow); // Use the residual graph

             vector<int> currentSubgraphNodes; // Store nodes for this 'low' value
             for (int node : min_cut_nodes) {
                 // Ensure node index is within the range of graph vertices in the flow network
                 if (node >= 2 && node < n + 2) {
                     currentSubgraphNodes.push_back(node - 2); // Map back to original 0-based index
                 }
             }
             // Update bestSubgraph found so far for the current 'low'
             bestSubgraph = currentSubgraphNodes; // Store the subgraph corresponding to the current 'low'
             log("    Found potential subgraph with " + to_string(bestSubgraph.size()) +
                 " nodes. Increasing density bound to " + to_string(low));
        }
        // --- END OF NEW FEASIBILITY CHECK ---
    }
    
    log("Binary search completed after " + to_string(iteration) + " iterations");
    log("Final density: " + to_string(low));
    
    sort(bestSubgraph.begin(), bestSubgraph.end());
    return {bestSubgraph, low, totalCliques};
}

// Write results to output file
void writeOutputFile(const string& inputFilename, int h, const vector<int>& result, 
                    double runTime, const unordered_map<int, int>& nodeMapping,
                    int totalNodes, int totalEdges, bool isDirected,
                    int totalCliques, double finalDensity) {
    // Create output directory if it doesn't exist
    string outputDir = "Output";
    // Use Windows-specific command to create directory
    system(("mkdir \"" + outputDir + "\" 2>nul").c_str());
    
    // Create output filename with h-value
    string baseName = inputFilename.substr(inputFilename.find_last_of("/\\") + 1);
    baseName = baseName.substr(0, baseName.find("-clean"));
    // Use backslash for Windows
    string outputFilename = outputDir + "\\" + baseName + "-output-h" + to_string(h) + ".txt";
    
    ofstream outFile(outputFilename);
    if (!outFile.is_open()) {
        log("Error: Cannot create output file " + outputFilename);
        return;
    }
    
    // Write comprehensive output information
    outFile << "Dataset: " << inputFilename << endl;
    outFile << "h-value: " << h << endl;
    outFile << "Graph information:" << endl;
    outFile << "  Nodes: " << totalNodes << endl;
    outFile << "  Edges: " << totalEdges << endl;
    outFile << "  Directed: " << (isDirected ? "Yes" : "No") << endl;
    outFile << endl;
    
    outFile << "Algorithm execution:" << endl;
    outFile << "  Total h-cliques found: " << totalCliques << endl;
    outFile << "  Final density: " << fixed << setprecision(6) << finalDensity << endl;
    outFile << "  Total execution time: " << fixed << setprecision(2) << runTime << " seconds" << endl;
    outFile << endl;
    
    outFile << "Results:" << endl;
    outFile << "  Densest subgraph size: " << result.size() << " nodes" << endl;
    
    if (result.empty()) {
        outFile << "  Densest subgraph vertices: (empty)" << endl;
    } else {
        outFile << "  Densest subgraph vertices (remapped IDs): ";
        for (int i = 0; i < result.size(); ++i) {
            outFile << result[i] << (i == result.size() - 1 ? "" : ", ");
        }
        outFile << endl;
        
        // If mapping exists, show original IDs as well
        if (!nodeMapping.empty()) {
            outFile << "  Densest subgraph vertices (original IDs): ";
            for (int i = 0; i < result.size(); ++i) {
                int remappedID = result[i];
                int originalID = nodeMapping.count(remappedID) ? nodeMapping.at(remappedID) : -1;
                outFile << originalID << (i == result.size() - 1 ? "" : ", ");
            }
            outFile << endl;
        }
    }
    
    log("Results written to " + outputFilename);
}

int main(int argc, char* argv[]) {
    // Validate command-line arguments
    if (argc < 3 || argc > 4) {
        cout << "Usage: " << argv[0] << " <h-value> <dataset-file> [max-nodes]" << endl;
        cout << "Example: " << argv[0] << " 3 As-733-clean.txt" << endl;
        cout << "Example with node limit: " << argv[0] << " 3 As-733-clean.txt 200" << endl;
        return 1;
    }
    
    try {
        // Parse arguments
        int h = stoi(argv[1]);
        if (h < 2) {
            log("Error: h-value must be at least 2");
            return 1;
        }
        
        string filename = argv[2];
        
        // Optional max nodes parameter
        int maxNodes = 0; // 0 means no limit
        if (argc == 4) {
            maxNodes = stoi(argv[3]);
            if (maxNodes <= 0) {
                log("Error: max-nodes must be positive");
                return 1;
            }
        }
        
        log("Algorithm 1: Exact Densest Subgraph for h-Cliques");
        log("h = " + to_string(h) + ", dataset = " + filename);
        if (maxNodes > 0) {
            log("Using only the first " + to_string(maxNodes) + " nodes");
        }
        
        // Parse the dataset
        auto start_parse = high_resolution_clock::now();
        Graph adj;
        int numNodes = 0, numEdges = 0;
        bool isDirected = false;
        if (!parseCleanDataset(filename, adj, numNodes, numEdges, isDirected)) {
            log("Error: Failed to parse dataset or empty graph");
            return 1;
        }
        auto end_parse = high_resolution_clock::now();
        double parse_time = duration_cast<milliseconds>(end_parse - start_parse).count() / 1000.0;
        
        log("Parsing completed in " + to_string(parse_time) + " seconds");
        
        // Limit the graph if maxNodes is specified
        if (maxNodes > 0 && maxNodes < adj.size()) {
            log("Limiting graph to first " + to_string(maxNodes) + " nodes");
            
            // Create a subgraph with only the first maxNodes nodes
            Graph limitedAdj(maxNodes);
            for (int i = 0; i < maxNodes; i++) {
                for (int neighbor : adj[i]) {
                    if (neighbor < maxNodes) {
                        limitedAdj[i].push_back(neighbor);
                    }
                }
            }
            adj = limitedAdj;
            numNodes = maxNodes;
            
            log("Limited graph has " + to_string(adj.size()) + " nodes");
        }
        
        // Load node ID mapping
        unordered_map<int, int> nodeMapping = loadNodeMapping(filename);
        
        // Run the algorithm (which will find all cliques internally)
        auto start_algo = high_resolution_clock::now();
        
        // Get result tuple containing {vertices, density, clique count}
        tuple<vector<int>, double, int> result_tuple = findDensestSubgraph(adj, h);
        vector<int> result = std::get<0>(result_tuple);
        double finalDensity = std::get<1>(result_tuple);
        int totalCliques = std::get<2>(result_tuple);
        
        auto end_algo = high_resolution_clock::now();
        double algo_time = duration_cast<milliseconds>(end_algo - start_algo).count() / 1000.0;
        
        // Calculate total runtime
        double total_time = parse_time + algo_time;
        
        // Print results
        log("\nResults:");
        log("  Densest subgraph size: " + to_string(result.size()) + " nodes");
        log("  Final density: " + to_string(finalDensity));
        log("  Total " + to_string(h) + "-cliques: " + to_string(totalCliques));
        
        if (result.size() <= 20) {
            string vertices = "  Vertices (remapped IDs): ";
            for (int i = 0; i < result.size(); ++i) {
                vertices += to_string(result[i]) + (i == result.size() - 1 ? "" : ", ");
            }
            log(vertices);
            
            // If mapping exists, show original IDs as well
            if (!nodeMapping.empty()) {
                string originalVertices = "  Vertices (original IDs): ";
                for (int i = 0; i < result.size(); ++i) {
                    int remappedID = result[i];
                    int originalID = nodeMapping.count(remappedID) ? nodeMapping.at(remappedID) : -1;
                    originalVertices += to_string(originalID) + (i == result.size() - 1 ? "" : ", ");
                }
                log(originalVertices);
            }
        } else {
            log("  (Too many vertices to display)");
        }
        
        log("  Algorithm runtime: " + to_string(algo_time) + " seconds");
        log("  Total runtime: " + to_string(total_time) + " seconds");
        
        // Write results to output file
        writeOutputFile(filename, h, result, total_time, nodeMapping, 
                      numNodes, numEdges, isDirected, totalCliques, finalDensity);
        
        return 0;
    } catch (const exception& e) {
        log("Error: " + string(e.what()));
        return 1;
    } catch (...) {
        log("Unknown error occurred");
        return 1;
    }
}