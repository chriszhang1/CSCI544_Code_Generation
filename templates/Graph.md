# Graph-Algorithm Problem-Solving Prompt (Python 3 — General Graphs)

Use this template whenever you tackle a coding problem that involves **graphs** (directed or undirected, weighted or unweighted). It mirrors the structure of the two-pointer prompt you provided, but focuses on graph-specific concerns.

## 1  Analyze the Problem Requirements and Constraints
- **Input / Output**  
  - How is the graph given? Adjacency list, edge list, matrix, implicit description?  
  - Directed vs. undirected?  Weighted vs. unweighted?  Negative weights possible?  
  - Vertex and edge count limits and value ranges.
- **Goal Type**  
  - Traversal / reachability, shortest path, connectivity, cycles, components, spanning tree, topological order, flow, etc.
- **Edge-Case Checklist**  
  - Empty graph (V = 0 or E = 0)  
  - Single-vertex graph  
  - Disconnected components  
  - Self-loops / parallel edges  
  - Cycles vs. acyclic input  
  - Fully connected dense graphs vs. sparse graphs  
  - Extreme values (max V, E) that stress memory or O(V·E) solutions
- **Complexity Targets**  
  - For most traversal/shortest-path problems aim for **O(V + E)**;  
    for weighted shortest paths with a priority queue, **O(E log V)**;  
    for dense graphs consider **O(V²)** algorithms if within limits.

## 2  Formulate a Clear Strategy for Algorithm & Data-Structure Selection
1. **Graph Representation**  
   - Choose adjacency list (sparse) or adjacency matrix (dense / constant-time edge lookup) or edge list (for Kruskal / sorting-based).  
   - Explain memory impact: adjacency list ≈ O(V + E), matrix ≈ O(V²).

2. **Core Algorithm Choice**  
   - Traversal: **BFS** (level order / shortest path in unweighted), **DFS** (cycle detection, topological sort).  
   - Shortest path: **Dijkstra**, **0-1 BFS**, **Bellman-Ford**, **SPFA**, **Floyd-Warshall** (all-pairs on ≤ 500 vertices).  
   - Connectivity: **Union-Find (DSU)**, **DFS/BFS** component labeling.  
   - MST: **Kruskal**, **Prim**.  
   - Topological ordering: **Kahn BFS** or DFS post-order.  
   - Flow / matching: **Edmonds-Karp**, **Dinic**, **Hopcroft–Karp**.

3. **Algorithm Mechanics (Pointer Analogue)**  
   - **Initialization**  
     - Visited set / distance array / parent map.  
     - Queue, stack, heap, DSU parent array, etc.  
   - **State Update Rules**  
     - When exploring an edge (u, v), update relaxation condition, push to data structure if improvement.  
     - Maintain invariants (e.g., distances finalized in Dijkstra when popped).  
   - **Termination Conditions**  
     - Traversal finishes when queue/stack/heap empty or target reached.  
     - Early exit optimizations (e.g., stop Dijkstra once destination distance fixed).  
   - **Edge-Case Handling**  
     - Guard against pushing the same node twice without benefit (distance check).  
     - Skip edges that violate constraints (negative weight in Dijkstra, etc.).

## 3  Time & Space Complexity Analysis
- Derive big-O for both **time** and **space** using V and E notation.  
- Justify why this meets—or is optimal for—the problem’s constraints.  
- Mention logarithmic factors from priority queues or sorting.

## 4  Testing and Validation
1. **Sample Case**  
   - Verify your code on the official example.

2. **Additional Test Suites**  
   - **Trivial graphs:** empty graph, single vertex, self-loop only.  
   - **Edge cases:** disconnected components, fully connected graph, tree structure, graph with parallel edges.  
   - **Stress tests:** maximum V and E sizes permitted, random heavy cases.  
   - **Pathological cases:** negative cycles (if relevant), multiple shortest paths with equal weight, graph where answer is at extreme end.

3. **Step-by-Step Walkthrough**  
   - Choose one non-trivial test.  
   - Show queue/stack/heap/DSU contents and visited/dist arrays at each key step.  
   - Illustrate how the algorithm converges to the correct answer.

## 5  Code Review and Refinement
- **Boundary Safety**  
  - Never access adjacency lists without vertex-existence checks.  
  - Ensure indices remain in [0, V − 1].
- **Correctness Verification**  
  - Distances must be monotonic non-decreasing when popped from Dijkstra’s heap.  
  - After DFS/BFS, validate every requirement (e.g., topological sort length equals V).
- **Performance Profiling**  
  - Confirm memory usage fits limits (avoid O(V²) on 2 × 10⁵ vertices).  
  - Replace recursion with iterative stack if recursion depth can exceed Python’s limit (≈ 10⁴).
- **Iterative Refinement**  
  - Address any failing edge case by tracing state transitions and fixing the responsible condition or update rule.
