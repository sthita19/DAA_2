#include <bits/stdc++.h>
using namespace std;

int h;
vector<vector<int>> g;

void enum_cliques(int v, vector<int> &path, int pos, vector<vector<int>> &cliques)
{
    if (path.size() == h - 1)
    {
        cliques.push_back(path);
        return;
    }
    for (int i = pos; i < g[v].size(); ++i)
    {
        int u = g[v][i];
        bool ok = true;
        for (int x : path)
        {
            if (!binary_search(g[u].begin(), g[u].end(), x))
            {
                ok = false;
                break;
            }
        }
        if (ok)
        {
            path.push_back(u);
            enum_cliques(v, path, i + 1, cliques);
            path.pop_back();
        }
    }
}

vector<int> core_num()
{
    int n = g.size();
    vector<int> deg(n, 0), core(n, 0);
    vector<bool> removed(n, false);

    for (int v = 0; v < n; ++v)
    {
        vector<vector<int>> cliques;
        vector<int> path;
        enum_cliques(v, path, 0, cliques);
        deg[v] = cliques.size();
    }

    vector<list<int>> bins(*max_element(deg.begin(), deg.end()) + 1);
    vector<list<int>::iterator> ptrs(n);
    for (int v = 0; v < n; ++v)
    {
        bins[deg[v]].push_front(v);
        ptrs[v] = bins[deg[v]].begin();
    }

    int k = 0;
    for (int cur = 0; cur < bins.size(); ++cur)
    {
        while (!bins[cur].empty())
        {
            int v = bins[cur].front();
            bins[cur].pop_front();

            core[v] = cur;
            removed[v] = true;

            vector<vector<int>> cliques;
            vector<int> path;
            enum_cliques(v, path, 0, cliques);

            for (auto &clique : cliques)
            {
                for (int u : clique)
                {
                    if (!removed[u] && deg[u] > cur)
                    {
                        auto &lst = bins[deg[u]];
                        lst.erase(ptrs[u]);
                        deg[u]--;
                        bins[deg[u]].push_front(u);
                        ptrs[u] = bins[deg[u]].begin();
                    }
                }
            }
        }
    }
    return core;
}

int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);

    int n, m;
    cin >> n >> m >> h;
    g.assign(n, {});

    for (int i = 0; i < m; ++i)
    {
        int u, v;
        cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }

    for (auto &vec : g)
    {
        sort(vec.begin(), vec.end());
        vec.erase(unique(vec.begin(), vec.end()), vec.end());
    }

    auto core = core_num();
    for (int x : core)
        cout << x << " ";

    return 0;
}