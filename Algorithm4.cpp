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
#include <iomanip>       // For setprecision
#include <cmath>         // For isfinite
#include <functional>    // For std::function
#include <unordered_map> // For node ID mapping
#include <list>

using namespace std;
using namespace std::chrono;

// Type aliases for clarity
using Graph = vector<vector<int>>;
using Clique = vector<int>;

const double EPSILON = 1e-9; // For double comparisons
const double INF = numeric_limits<double>::infinity();

// Flow network structures
struct FlowEdge
{
    int to;
    double capacity;
    int reverseEdgeIndex;
};

using FlowGraph = vector<vector<FlowEdge>>;

// Function for detailed logging
void log(const string &message)
{
    cout << message << endl;
    cout.flush(); // Ensure output appears immediately
}

// Parse cleaned dataset files (from Algorithm1.cpp)
bool parseCleanDataset(const string &filename, Graph &adj, int &numNodes, int &numEdges, bool &isDirected)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        log("Error: Cannot open file " + filename);
        return false;
    }

    log("Parsing dataset: " + filename);

    string line;

    // Parse metadata line
    if (getline(file, line))
    {
        if (line[0] == '#')
        {
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

    if (numNodes <= 0)
    {
        log("Error: Invalid node count in file");
        return false;
    }

    // Initialize adjacency list
    adj.resize(numNodes);

    // Reserve space to reduce reallocations
    for (auto &neighbors : adj)
    {
        neighbors.reserve(50); // Reserve for average degree
    }

    // Parse edges
    int edgeCount = 0;
    while (getline(file, line))
    {
        if (line.empty() || line[0] == '#')
            continue;

        istringstream ss(line);
        int from, to;
        if (ss >> from >> to)
        {
            if (from >= 0 && from < numNodes && to >= 0 && to < numNodes)
            {
                adj[from].push_back(to);
                // For undirected graphs, add the reverse edge
                if (!isDirected)
                {
                    adj[to].push_back(from);
                }
                edgeCount++;
            }
            else
            {
                log("Warning: Edge (" + to_string(from) + "," + to_string(to) +
                    ") refers to nodes outside valid range [0," + to_string(numNodes - 1) + "]");
            }
        }
    }

    // Sort adjacency lists and remove duplicates
    for (auto &neighbors : adj)
    {
        sort(neighbors.begin(), neighbors.end());
        neighbors.erase(unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }

    log("Successfully parsed " + to_string(numNodes) + " nodes and " +
        to_string(edgeCount) + " edges");

    return true;
}

// Function to check if two vertices are connected (from Algorithm1.cpp)
bool areConnected(int u, int v, const Graph &adj)
{
    if (u < 0 || u >= adj.size() || v < 0 || v >= adj.size())
        return false;
    return binary_search(adj[u].begin(), adj[u].end(), v);
}

// Load node ID mapping from file (from Algorithm1.cpp)
unordered_map<int, int> loadNodeMapping(const string &filename)
{
    unordered_map<int, int> mapping;
    string mappingFile = filename.substr(0, filename.find("-clean")) + "-mapping.txt";
    ifstream mapFile(mappingFile);

    if (!mapFile.is_open())
    {
        log("Note: No mapping file found at " + mappingFile);
        return mapping;
    }

    log("Loading node ID mapping from " + mappingFile);

    string line;
    int lineCount = 0;

    while (getline(mapFile, line))
    {
        lineCount++;
        if (line.empty() || line[0] == '#')
            continue;

        istringstream ss(line);
        int remappedId;
        int originalId;
        char comma;

        if (ss >> remappedId >> comma >> originalId)
        {
            mapping[remappedId] = originalId;
        }
        else
        {
            log("Warning: Invalid format in mapping file at line " + to_string(lineCount));
        }
    }

    log("Loaded " + to_string(mapping.size()) + " node mappings");
    return mapping;
}

// Enumerate h-1 cliques for a vertex (based on Algorithm3.cpp)
void enumCliques(int v, vector<int> &path, int pos, vector<vector<int>> &cliques, const Graph &adj, int h)
{
    if (path.size() == h - 1)
    {
        cliques.push_back(path);
        return;
    }
    for (int i = pos; i < adj[v].size(); ++i)
    {
        int u = adj[v][i];
        bool ok = true;
        for (int x : path)
        {
            if (!binary_search(adj[u].begin(), adj[u].end(), x))
            {
                ok = false;
                break;
            }
        }
        if (ok)
        {
            path.push_back(u);
            enumCliques(v, path, i + 1, cliques, adj, h);
            path.pop_back();
        }
    }
}

// Find all h-cliques containing a vertex (based on Algorithm3.cpp)
vector<vector<int>> findHCliquesForVertex(int v, const Graph &adj, int h)
{
    vector<vector<int>> cliques;
    vector<int> path;
    enumCliques(v, path, 0, cliques, adj, h);
    return cliques;
}

// Calculate clique-degree for a vertex (number of h-cliques it belongs to)
int calculateCliqueDegree(int v, const Graph &adj, int h)
{
    return findHCliquesForVertex(v, adj, h).size();
}

// Core decomposition algorithm (based on Algorithm3.cpp)
vector<int> coreDecomposition(const Graph &adj, int h)
{
    int n = adj.size();
    vector<int> deg(n), core(n, 0);
    vector<bool> removed(n, false);

    log("Computing clique degrees for all vertices...");
    // Calculate clique degrees for each vertex
    for (int v = 0; v < n; ++v)
    {
        deg[v] = calculateCliqueDegree(v, adj, h);
        if (v % 100 == 0 || v == n - 1)
        {
            log("  Progress: Computed clique degree for vertex " + to_string(v + 1) + "/" + to_string(n));
        }
    }

    log("Performing core decomposition...");
    // Implement the bucketing method for core decomposition
    int max_deg = *max_element(deg.begin(), deg.end());
    vector<list<int>> bins(max_deg + 1);
    vector<list<int>::iterator> pos(n);

    for (int v = 0; v < n; ++v)
    {
        bins[deg[v]].push_front(v);
        pos[v] = bins[deg[v]].begin();
    }

    // Process vertices in non-decreasing order of degrees
    for (int d = 0; d <= max_deg; ++d)
    {
        while (!bins[d].empty())
        {
            int v = bins[d].front();
            bins[d].pop_front();
            core[v] = d;
            removed[v] = true;

            // Update degrees of neighbors
            vector<vector<int>> cliques = findHCliquesForVertex(v, adj, h);

            for (const auto &clique : cliques)
            {
                for (int u : clique)
                {
                    if (!removed[u] && deg[u] > d)
                    {
                        bins[deg[u]].erase(pos[u]);
                        deg[u]--;
                        bins[deg[u]].push_front(u);
                        pos[u] = bins[deg[u]].begin();
                    }
                }
            }
        }
    }

    return core;
}

// Identify connected components in a subgraph
vector<vector<int>> findConnectedComponents(const Graph &adj, const vector<int> &vertices)
{
    int n = adj.size();
    vector<bool> visited(n, false);
    vector<vector<int>> components;

    for (int start : vertices)
    {
        if (visited[start])
            continue;

        // BFS to find connected component
        vector<int> component;
        queue<int> q;
        q.push(start);
        visited[start] = true;
        component.push_back(start);

        while (!q.empty())
        {
            int u = q.front();
            q.pop();

            for (int v : adj[u])
            {
                // Check if v is in the vertices set
                if (!visited[v] && find(vertices.begin(), vertices.end(), v) != vertices.end())
                {
                    visited[v] = true;
                    q.push(v);
                    component.push_back(v);
                }
            }
        }

        components.push_back(component);
    }

    return components;
}

// Extract a subgraph induced by a set of vertices
Graph extractSubgraph(const Graph &adj, const vector<int> &vertices)
{
    // Create a mapping from original vertex IDs to new vertex IDs
    unordered_map<int, int> vertexMapping;
    for (int i = 0; i < vertices.size(); ++i)
    {
        vertexMapping[vertices[i]] = i;
    }

    // Create the subgraph
    Graph subgraph(vertices.size());

    for (int i = 0; i < vertices.size(); ++i)
    {
        int u = vertices[i];
        for (int v : adj[u])
        {
            // Check if neighbor is in the subgraph
            if (vertexMapping.find(v) != vertexMapping.end())
            {
                subgraph[i].push_back(vertexMapping[v]);
            }
        }

        // Sort neighbors
        sort(subgraph[i].begin(), subgraph[i].end());
    }

    return subgraph;
}

// Find all h-cliques in a graph
vector<Clique> findAllCliques(const Graph &adj, int h)
{
    vector<Clique> cliques;
    int n = adj.size();

    log("Finding all " + to_string(h) + "-cliques...");

    // For each vertex, find all h-cliques it belongs to
    for (int v = 0; v < n; ++v)
    {
        vector<vector<int>> vertexCliques = findHCliquesForVertex(v, adj, h);

        // Add the vertex to each clique and ensure uniqueness
        for (auto &clique : vertexCliques)
        {
            clique.push_back(v);
            sort(clique.begin(), clique.end());

            // Only add unique cliques
            if (find(cliques.begin(), cliques.end(), clique) == cliques.end())
            {
                cliques.push_back(clique);
            }
        }

        if (v % 100 == 0 || v == n - 1)
        {
            log("  Progress: Processed vertex " + to_string(v + 1) + "/" + to_string(n) +
                ", found " + to_string(cliques.size()) + " cliques so far");
        }
    }

    log("Found " + to_string(cliques.size()) + " " + to_string(h) + "-cliques");
    return cliques;
}

// Calculate density of a subgraph (number of h-cliques / number of vertices)
double calculateDensity(const vector<Clique> &cliques, const vector<int> &vertices)
{
    if (vertices.empty())
        return 0.0;

    // Count cliques that are fully contained in the vertex set
    int cliqueCount = 0;
    set<int> vertexSet(vertices.begin(), vertices.end());

    for (const auto &clique : cliques)
    {
        bool fullyContained = true;
        for (int v : clique)
        {
            if (vertexSet.find(v) == vertexSet.end())
            {
                fullyContained = false;
                break;
            }
        }
        if (fullyContained)
        {
            cliqueCount++;
        }
    }

    return static_cast<double>(cliqueCount) / vertices.size();
}

// Add an edge to the flow network (from Algorithm1.cpp)
void addFlowEdge(int u, int v, double capacity, FlowGraph &flow)
{
    flow[u].push_back({v, capacity, static_cast<int>(flow[v].size())});
    flow[v].push_back({u, 0.0, static_cast<int>(flow[u].size() - 1)});
}

// Find augmenting path using BFS (from Algorithm1.cpp)
bool findAugmentingPath(int s, int t, const FlowGraph &flow, vector<int> &parentEdge, vector<int> &parentNode)
{
    fill(parentEdge.begin(), parentEdge.end(), -1);
    fill(parentNode.begin(), parentNode.end(), -1);
    parentNode[s] = -2;

    queue<pair<int, double>> q;
    q.push({s, INF});

    while (!q.empty())
    {
        int u = q.front().first;
        double flow_val = q.front().second;
        q.pop();

        for (int i = 0; i < flow[u].size(); ++i)
        {
            const auto &edge = flow[u][i];
            if (parentNode[edge.to] == -1 && edge.capacity > EPSILON)
            {
                parentNode[edge.to] = u;
                parentEdge[edge.to] = i;
                double new_flow = min(flow_val, edge.capacity);
                if (edge.to == t)
                    return true;
                q.push({edge.to, new_flow});
            }
        }
    }
    return false;
}

// Compute maximum flow using Edmonds-Karp algorithm (from Algorithm1.cpp)
double computeMaxFlow(int s, int t, FlowGraph &flow)
{
    double total_flow = 0;
    vector<int> parentEdge(flow.size());
    vector<int> parentNode(flow.size());

    while (findAugmentingPath(s, t, flow, parentEdge, parentNode))
    {
        double path_flow = INF;

        // Find bottleneck capacity
        for (int v = t; v != s; v = parentNode[v])
        {
            int u = parentNode[v];
            int edge_idx = parentEdge[v];
            path_flow = min(path_flow, flow[u][edge_idx].capacity);
        }

        if (!isfinite(path_flow) || path_flow <= EPSILON)
        {
            log("Warning: Invalid path flow detected");
            break;
        }

        // Update residual capacities
        for (int v = t; v != s; v = parentNode[v])
        {
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

// Find the nodes on the source side of the min-cut (from Algorithm1.cpp)
vector<int> findMinCut(int s, int t, const FlowGraph &flow)
{
    vector<bool> visited(flow.size(), false);
    vector<int> min_cut_nodes;
    queue<int> q;

    q.push(s);
    visited[s] = true;
    min_cut_nodes.push_back(s);

    while (!q.empty())
    {
        int u = q.front();
        q.pop();

        for (const auto &edge : flow[u])
        {
            if (edge.capacity > EPSILON && !visited[edge.to])
            {
                visited[edge.to] = true;
                q.push(edge.to);
                min_cut_nodes.push_back(edge.to);
            }
        }
    }

    return min_cut_nodes;
}

// Build flow network for the exact algorithm (from Algorithm1.cpp, modified for subgraphs)
FlowGraph buildFlowNetwork(const vector<int> &component, const Graph &adj, double alpha, int h,
                           vector<Clique> &h_cliques, vector<Clique> &h_minus_1_cliques)
{
    int n = component.size();

    // Calculate clique degrees for each vertex in the component
    vector<double> cliqueDegrees(n, 0.0);
    for (const auto &clique : h_cliques)
    {
        // Check if clique is fully within the component
        bool fullyContained = true;
        for (int v : clique)
        {
            // Check if v is in the component
            if (find(component.begin(), component.end(), v) == component.end())
            {
                fullyContained = false;
                break;
            }
        }

        if (fullyContained)
        {
            for (int v : clique)
            {
                // Map vertex to its index within component
                auto it = find(component.begin(), component.end(), v);
                int idx = distance(component.begin(), it);
                cliqueDegrees[idx]++;
            }
        }
    }

    // Build a map of (h-1)-cliques for the flow network
    map<vector<int>, int> h_minus_1_clique_map;
    int next_clique_id = n + 2; // Start after nodes for s, t, and component vertices

    if (h > 2)
    {
        for (int i = 0; i < h_minus_1_cliques.size(); ++i)
        {
            // Check if clique is fully within the component
            bool fullyContained = true;
            for (int v : h_minus_1_cliques[i])
            {
                if (find(component.begin(), component.end(), v) == component.end())
                {
                    fullyContained = false;
                    break;
                }
            }

            if (fullyContained)
            {
                h_minus_1_clique_map[h_minus_1_cliques[i]] = next_clique_id++;
            }
        }
    }

    // Build flow network
    int flow_size = 2 + n + (h > 2 ? h_minus_1_clique_map.size() : 0);
    FlowGraph flow(flow_size);

    // Pre-allocate capacity in flow adjacency lists
    for (auto &edges : flow)
    {
        edges.reserve(20); // Reasonable capacity estimate
    }

    int s = 0, t = 1;

    // Add edges s -> v with capacity = clique_degree(v)
    for (int i = 0; i < n; ++i)
    {
        if (cliqueDegrees[i] > EPSILON)
        {
            addFlowEdge(s, i + 2, cliqueDegrees[i], flow);
        }
    }

    // Add edges v -> t with capacity = alpha
    for (int i = 0; i < n; ++i)
    {
        addFlowEdge(i + 2, t, alpha, flow);
    }

    // For h > 2, add additional edges to represent clique structure
    if (h > 2)
    {
        set<pair<int, int>> added_edges;

        for (const auto &h_clique : h_cliques)
        {
            // Check if clique is fully within the component
            bool fullyContained = true;
            for (int v : h_clique)
            {
                if (find(component.begin(), component.end(), v) == component.end())
                {
                    fullyContained = false;
                    break;
                }
            }

            if (fullyContained)
            {
                for (int i = 0; i < h_clique.size(); ++i)
                {
                    int v = h_clique[i];

                    // Create an (h-1)-clique by removing vertex v
                    vector<int> h_minus_1;
                    for (int j = 0; j < h_clique.size(); ++j)
                    {
                        if (i != j)
                            h_minus_1.push_back(h_clique[j]);
                    }
                    sort(h_minus_1.begin(), h_minus_1.end());

                    // Add edge v -> (h-1)-clique with capacity 1
                    if (h_minus_1_clique_map.count(h_minus_1))
                    {
                        // Map vertex to its index within component
                        auto it = find(component.begin(), component.end(), v);
                        int idx = distance(component.begin(), it);

                        int v_node = idx + 2; // +2 for s and t
                        int psi_node = h_minus_1_clique_map[h_minus_1];

                        // Avoid duplicate edges
                        pair<int, int> edge_pair(v_node, psi_node);
                        if (added_edges.find(edge_pair) == added_edges.end())
                        {
                            addFlowEdge(v_node, psi_node, 1.0, flow);
                            added_edges.insert(edge_pair);
                        }
                    }
                }
            }
        }
    }

    return flow;
}

// Find densest subgraph in a component using flow-based approach
vector<int> findDensestSubgraphInComponent(const vector<int> &component, const Graph &adj, int h,
                                           double l, double u, vector<Clique> &h_cliques,
                                           vector<Clique> &h_minus_1_cliques)
{
    int n = component.size();
    if (n == 0)
    {
        return {};
    }

    // Binary search for the optimal density
    double low = l, high = u;
    vector<int> bestSubgraph;
    double precision = 1.0 / (n * (n - 1) + 1);
    int maxIterations = 30;

    for (int iteration = 0; iteration < maxIterations && high - low > precision; ++iteration)
    {
        double mid = low + (high - low) / 2.0;
        log("  Iteration " + to_string(iteration + 1) + ": Testing density " +
            to_string(mid) + " (range [" + to_string(low) + ", " + to_string(high) + "])");

        // Build flow network
        FlowGraph flow = buildFlowNetwork(component, adj, mid, h, h_cliques, h_minus_1_cliques);

        // Compute max-flow
        int s = 0, t = 1;
        double flow_value = computeMaxFlow(s, t, flow);

        // Find min-cut
        vector<int> min_cut = findMinCut(s, t, flow);

        // Check if cut is non-trivial (more than just source)
        if (min_cut.size() <= 1)
        {
            high = mid; // Density too high
        }
        else
        {
            low = mid; // Found a candidate subgraph

            // Extract vertices from min-cut (convert from flow network IDs back to component IDs)
            bestSubgraph.clear();
            for (int node : min_cut)
            {
                if (node >= 2 && node < n + 2)
                {
                    // Convert back to original vertex ID
                    bestSubgraph.push_back(component[node - 2]);
                }
            }
        }
    }

    return bestSubgraph;
}

// CoreExact algorithm implementation
tuple<vector<int>, double> coreExact(const Graph &adj, int h, double prune_threshold)
{
    int n = adj.size();

    // Step 1: Perform core decomposition
    log("Step 1: Performing core decomposition...");
    vector<int> core_numbers = coreDecomposition(adj, h);

    // Find k'' value (maximum core number)
    int k_double_prime = *max_element(core_numbers.begin(), core_numbers.end());
    log("Maximum core number (k''): " + to_string(k_double_prime));

    // Step 2: Locate the (k'', Ψ)-core using pruning criteria
    log("Step 2: Locating (k'', Ψ)-core with pruning threshold " + to_string(prune_threshold));
    vector<int> k_double_prime_core;
    for (int v = 0; v < n; ++v)
    {
        if (core_numbers[v] >= k_double_prime * prune_threshold)
        {
            k_double_prime_core.push_back(v);
        }
    }

    log("Size of (k'', Ψ)-core: " + to_string(k_double_prime_core.size()) + " vertices");

    // Step 3: Find all connected components in the k''-core
    log("Step 3: Finding connected components in the (k'', Ψ)-core");
    vector<vector<int>> components = findConnectedComponents(adj, k_double_prime_core);
    log("Found " + to_string(components.size()) + " connected components");

    // Step 4: Initialize variables
    vector<int> C, D, U;
    double l = prune_threshold * k_double_prime;
    double u = k_double_prime;

    // Find all h-cliques in the graph (needed for density calculations)
    log("Finding all h-cliques in the graph...");
    vector<Clique> h_cliques = findAllCliques(adj, h);

    // If needed for h > 2, find (h-1)-cliques
    vector<Clique> h_minus_1_cliques;
    if (h > 2)
    {
        log("Finding all (h-1)-cliques in the graph...");
        h_minus_1_cliques = findAllCliques(adj, h - 1);
    }

    // Step 5: Process each connected component
    vector<int> best_subgraph;
    double best_density = 0.0;

    log("Step 5: Processing each connected component...");
    for (const auto &component : components)
    {
        log("  Processing component with " + to_string(component.size()) + " vertices");

        // Check if component size is larger than threshold
        if (component.size() > h)
        {
            log("  Component size > h, applying flow-based algorithm");

            // Build a flow network and find min st-cut
            FlowGraph flow = buildFlowNetwork(component, adj, 0, h, h_cliques, h_minus_1_cliques);
            int s = 0, t = 1;
            double flow_value = computeMaxFlow(s, t, flow);
            vector<int> min_cut = findMinCut(s, t, flow);

            // Skip if min-cut is empty
            if (min_cut.size() <= 1)
            {
                log("  Empty min-cut, skipping component");
                continue;
            }

            // Loop for binary search
            double component_l = l;
            double component_u = u;
            while (component_u - component_l >= 1.0 / (component.size() * (component.size() - 1)))
            {
                double alpha = (component_l + component_u) / 2.0;
                log("  Testing density threshold alpha = " + to_string(alpha));

                // Build flow network with current alpha
                FlowGraph flow = buildFlowNetwork(component, adj, alpha, h, h_cliques, h_minus_1_cliques);
                double flow_value = computeMaxFlow(s, t, flow);
                vector<int> min_cut = findMinCut(s, t, flow);

                if (min_cut.size() <= 1)
                {
                    // Cut is trivial (only source)
                    component_u = alpha;
                }
                else
                {
                    // Found a candidate subgraph
                    component_l = alpha;

                    // Extract vertices from min-cut
                    U.clear();
                    for (int node : min_cut)
                    {
                        if (node >= 2 && node < component.size() + 2)
                        {
                            U.push_back(component[node - 2]);
                        }
                    }

                    // If alpha > l, remove some vertices (pruning)
                    if (alpha > l && !U.empty())
                    {
                        // Implement vertex pruning if needed
                    }
                }
            }

            // Check if U has better density than current best
            double U_density = calculateDensity(h_cliques, U);
            double D_density = calculateDensity(h_cliques, D);

            log("  Subgraph found with " + to_string(U.size()) + " vertices and density " + to_string(U_density));

            if (U_density > D_density)
            {
                D = U;
                log("  New best subgraph found with density " + to_string(U_density));
            }
        }

        // Track overall best subgraph
        double component_density = calculateDensity(h_cliques, D);
        if (component_density > best_density || best_subgraph.empty())
        {
            best_density = component_density;
            best_subgraph = D;
        }
    }

    log("Best subgraph found with " + to_string(best_subgraph.size()) +
        " vertices and density " + to_string(best_density));

    return {best_subgraph, best_density};
}

// Write results to output file (similar to Algorithm1.cpp)
void writeOutputFile(const string &inputFilename, int h, const vector<int> &result,
                     double runTime, const unordered_map<int, int> &nodeMapping,
                     int totalNodes, int totalEdges, bool isDirected,
                     double finalDensity)
{
    // Create output directory if it doesn't exist
    string outputDir = "Output";
    system(("mkdir \"" + outputDir + "\" 2>nul").c_str());

    // Create output filename with h-value
    string baseName = inputFilename.substr(inputFilename.find_last_of("/\\") + 1);
    baseName = baseName.substr(0, baseName.find("-clean"));
    string outputFilename = outputDir + "\\" + baseName + "-coreexact-h" + to_string(h) + ".txt";

    ofstream outFile(outputFilename);
    if (!outFile.is_open())
    {
        log("Error: Cannot create output file " + outputFilename);
        return;
    }

    // Write comprehensive output information
    outFile << "Dataset: " << inputFilename << endl;
    outFile << "Algorithm: CoreExact" << endl;
    outFile << "h-value: " << h << endl;
    outFile << "Graph information:" << endl;
    outFile << "  Nodes: " << totalNodes << endl;
    outFile << "  Edges: " << totalEdges << endl;
    outFile << "  Directed: " << (isDirected ? "Yes" : "No") << endl;
    outFile << endl;

    outFile << "Algorithm execution:" << endl;
    outFile << "  Final density: " << fixed << setprecision(6) << finalDensity << endl;
    outFile << "  Total execution time: " << fixed << setprecision(2) << runTime << " seconds" << endl;
    outFile << endl;

    outFile << "Results:" << endl;
    outFile << "  Densest subgraph size: " << result.size() << " nodes" << endl;

    if (result.empty())
    {
        outFile << "  Densest subgraph vertices: (empty)" << endl;
    }
    else
    {
        outFile << "  Densest subgraph vertices (remapped IDs): ";
        for (int i = 0; i < result.size(); ++i)
        {
            outFile << result[i] << (i == result.size() - 1 ? "" : ", ");
        }
        outFile << endl;

        // If mapping exists, show original IDs as well
        if (!nodeMapping.empty())
        {
            outFile << "  Densest subgraph vertices (original IDs): ";
            for (int i = 0; i < result.size(); ++i)
            {
                int remappedID = result[i];
                int originalID = nodeMapping.count(remappedID) ? nodeMapping.at(remappedID) : -1;
                outFile << originalID << (i == result.size() - 1 ? "" : ", ");
            }
            outFile << endl;
        }
    }

    log("Results written to " + outputFilename);
}

int main(int argc, char *argv[])
{
    // Validate command-line arguments
    if (argc < 3 || argc > 5)
    {
        cout << "Usage: " << argv[0] << " <h-value> <dataset-file> [pruning-threshold] [max-nodes]" << endl;
        cout << "Example: " << argv[0] << " 3 As-733-clean.txt" << endl;
        cout << "Example with pruning: " << argv[0] << " 3 As-733-clean.txt 0.7" << endl;
        cout << "Example with node limit: " << argv[0] << " 3 As-733-clean.txt 0.7 200" << endl;
        return 1;
    }

    try
    {
        // Parse arguments
        int h = stoi(argv[1]);
        if (h < 2)
        {
            log("Error: h-value must be at least 2");
            return 1;
        }

        string filename = argv[2];

        // Optional pruning threshold parameter (default: 0.7)
        double pruneThreshold = 0.7;
        if (argc >= 4)
        {
            pruneThreshold = stod(argv[3]);
            if (pruneThreshold <= 0.0 || pruneThreshold > 1.0)
            {
                log("Error: pruning threshold must be in (0, 1]");
                return 1;
            }
        }

        // Optional max nodes parameter
        int maxNodes = 0; // 0 means no limit
        if (argc == 5)
        {
            maxNodes = stoi(argv[4]);
            if (maxNodes <= 0)
            {
                log("Error: max-nodes must be positive");
                return 1;
            }
        }

        log("Algorithm 4: CoreExact Densest Subgraph for h-Cliques");
        log("h = " + to_string(h) + ", dataset = " + filename);
        log("Pruning threshold = " + to_string(pruneThreshold));
        if (maxNodes > 0)
        {
            log("Using only the first " + to_string(maxNodes) + " nodes");
        }

        // Parse the dataset
        auto start_parse = high_resolution_clock::now();
        Graph adj;
        int numNodes = 0, numEdges = 0;
        bool isDirected = false;
        if (!parseCleanDataset(filename, adj, numNodes, numEdges, isDirected))
        {
            log("Error: Failed to parse dataset or empty graph");
            return 1;
        }
        auto end_parse = high_resolution_clock::now();
        double parse_time = duration_cast<milliseconds>(end_parse - start_parse).count() / 1000.0;

        log("Parsing completed in " + to_string(parse_time) + " seconds");

        // Limit the graph if maxNodes is specified
        if (maxNodes > 0 && maxNodes < adj.size())
        {
            log("Limiting graph to first " + to_string(maxNodes) + " nodes");

            // Create a subgraph with only the first maxNodes nodes
            Graph limitedAdj(maxNodes);
            for (int i = 0; i < maxNodes; i++)
            {
                for (int neighbor : adj[i])
                {
                    if (neighbor < maxNodes)
                    {
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

        // Run the CoreExact algorithm
        auto start_algo = high_resolution_clock::now();

        auto [result, finalDensity] = coreExact(adj, h, pruneThreshold);

        auto end_algo = high_resolution_clock::now();
        double algo_time = duration_cast<milliseconds>(end_algo - start_algo).count() / 1000.0;

        // Calculate total runtime
        double total_time = parse_time + algo_time;

        // Print results
        log("\nResults:");
        log("  Densest subgraph size: " + to_string(result.size()) + " nodes");
        log("  Final density: " + to_string(finalDensity));

        if (result.size() <= 20)
        {
            string vertices = "  Vertices (remapped IDs): ";
            for (int i = 0; i < result.size(); ++i)
            {
                vertices += to_string(result[i]) + (i == result.size() - 1 ? "" : ", ");
            }
            log(vertices);

            // If mapping exists, show original IDs as well
            if (!nodeMapping.empty())
            {
                string originalVertices = "  Vertices (original IDs): ";
                for (int i = 0; i < result.size(); ++i)
                {
                    int remappedID = result[i];
                    int originalID = nodeMapping.count(remappedID) ? nodeMapping.at(remappedID) : -1;
                    originalVertices += to_string(originalID) + (i == result.size() - 1 ? "" : ", ");
                }
                log(originalVertices);
            }
        }
        else
        {
            log("  (Too many vertices to display)");
        }

        log("  Algorithm runtime: " + to_string(algo_time) + " seconds");
        log("  Total runtime: " + to_string(total_time) + " seconds");

        // Write results to output file
        writeOutputFile(filename, h, result, total_time, nodeMapping,
                        numNodes, numEdges, isDirected, finalDensity);

        return 0;
    }
    catch (const exception &e)
    {
        log("Error: " + string(e.what()));
        return 1;
    }
    catch (...)
    {
        log("Unknown error occurred");
        return 1;
    }
}