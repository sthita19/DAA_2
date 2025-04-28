#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <algorithm>

using namespace std;

// Process a dataset file and output standardized format
void preprocessDataset(const string& inputFile, const string& outputFile) {
    ifstream inFile(inputFile);
    if (!inFile.is_open()) {
        cerr << "Error: Cannot open input file " << inputFile << endl;
        return;
    }
    
    cout << "Processing " << inputFile << "..." << endl;
    
    string line;
    vector<pair<int, int>> edges;
    unordered_map<int, int> nodeIdMap;
    bool isDirected = false;
    int originalNodeCount = 0;
    int originalEdgeCount = 0;
    
    // Read input file
    while (getline(inFile, line)) {
        if (line.empty()) continue;
        
        // Process comment lines to extract metadata
        if (line[0] == '#') {
            if (line.find("Directed") != string::npos) {
                isDirected = true;
            }
            
            // Try to extract node and edge counts
            if (line.find("Nodes") != string::npos && line.find("Edges") != string::npos) {
                size_t nodesPos = line.find("Nodes");
                size_t edgesPos = line.find("Edges");
                
                if (nodesPos != string::npos && edgesPos != string::npos) {
                    string nodesPart = line.substr(nodesPos);
                    string edgesPart = line.substr(edgesPos);
                    
                    // Extract numbers using stringstream
                    istringstream nodesSS(nodesPart);
                    string temp;
                    nodesSS >> temp; // Skip "Nodes:"
                    nodesSS >> originalNodeCount;
                    
                    istringstream edgesSS(edgesPart);
                    edgesSS >> temp; // Skip "Edges:"
                    edgesSS >> originalEdgeCount;
                }
            }
            continue;
        }
        
        // Process edge lines
        istringstream ss(line);
        int from, to;
        if (ss >> from >> to) {
            // Track unique node IDs
            if (nodeIdMap.find(from) == nodeIdMap.end()) {
                nodeIdMap[from] = nodeIdMap.size();
            }
            if (nodeIdMap.find(to) == nodeIdMap.end()) {
                nodeIdMap[to] = nodeIdMap.size();
            }
            
            // Store the edge with original IDs
            edges.push_back({from, to});
        }
    }
    
    inFile.close();
    
    // Create output file
    ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        cerr << "Error: Cannot create output file " << outputFile << endl;
        return;
    }
    
    int remappedNodeCount = nodeIdMap.size();
    int remappedEdgeCount = isDirected ? edges.size() : edges.size() / 2;
    
    // Write metadata
    outFile << "# nodes: " << remappedNodeCount 
           << " edges: " << remappedEdgeCount 
           << " directed: " << (isDirected ? 1 : 0) << endl;
    
    // Write edges with remapped IDs
    set<pair<int, int>> uniqueEdges; // To handle potential duplicates
    
    for (const auto& edge : edges) {
        int u = nodeIdMap[edge.first];
        int v = nodeIdMap[edge.second];
        
        // For undirected graphs, store edges in canonical order
        if (!isDirected && u > v) {
            swap(u, v);
        }
        
        // Add to set of unique edges (handles duplicates)
        uniqueEdges.insert({u, v});
    }
    
    // Write unique edges to file
    for (const auto& edge : uniqueEdges) {
        outFile << edge.first << " " << edge.second << endl;
    }
    
    outFile.close();
    
    // Create mapping file with original to remapped ID mapping
    string mappingFile = inputFile.substr(0, inputFile.find_last_of('.')) + "-mapping.txt";
    ofstream mapFile(mappingFile);
    if (!mapFile.is_open()) {
        cerr << "Error: Cannot create mapping file " << mappingFile << endl;
        return;
    }
    
    // Write header
    mapFile << "# Original to remapped node ID mapping" << endl;
    mapFile << "# remapped_id, original_id" << endl;
    
    // Create a reverse map for easier sorting by remapped ID
    vector<pair<int, int>> sortedMapping;
    for (const auto& pair : nodeIdMap) {
        sortedMapping.push_back({pair.second, pair.first});  // remapped_id, original_id
    }
    
    // Sort by remapped ID for easier lookup
    sort(sortedMapping.begin(), sortedMapping.end());
    
    // Write mapping entries
    for (const auto& pair : sortedMapping) {
        mapFile << pair.first << "," << pair.second << endl;
    }
    
    mapFile.close();
    
    cout << "Processed " << inputFile << ":" << endl;
    cout << "  - Original: " << (originalNodeCount > 0 ? originalNodeCount : nodeIdMap.size()) 
         << " nodes, " << (originalEdgeCount > 0 ? originalEdgeCount : edges.size()) << " edges" << endl;
    cout << "  - Remapped: " << remappedNodeCount << " nodes, " << uniqueEdges.size() << " edges" << endl;
    cout << "  - " << (isDirected ? "Directed" : "Undirected") << " graph" << endl;
    cout << "  - Output saved to " << outputFile << endl;
    cout << "  - Node ID mapping saved to " << mappingFile << endl;
}

int main(int argc, char* argv[]) {
    vector<string> inputFiles;
    
    if (argc > 1) {
        // Process specified files
        for (int i = 1; i < argc; i++) {
            inputFiles.push_back(argv[i]);
        }
    } else {
        // Default files
        inputFiles = {"As-733.txt", "As-Caida.txt", "Ca-HepTh.txt"};
    }
    
    for (const string& inputFile : inputFiles) {
        string outputFile = inputFile.substr(0, inputFile.find_last_of('.')) + "-clean.txt";
        preprocessDataset(inputFile, outputFile);
    }
    
    cout << "\nAll datasets processed successfully." << endl;
    cout << "You can now run Algorithm1 with the cleaned datasets." << endl;
    
    return 0;
}