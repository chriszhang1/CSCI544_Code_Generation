# Depth First Search (DFS) Problem Solving Prompt

## Task Overview
As an experienced software developer, your task is to design and implement a solution in Python 3 for the given coding problem using the **Depth First Search (DFS)** technique. The following steps should be followed to ensure a comprehensive and correct solution:

### 1. **Analyze the Problem Requirements and Constraints**
   - Thoroughly review the problem's input, output, and constraints.
   - Identify any specific edge cases, such as minimal or extreme values, empty inputs, or other relevant corner cases.
   - Understand the problem's complexity requirements and ensure the solution operates efficiently (typically O(V + E) for DFS on graphs or O(n) for tree-based problems).

### 2. **Formulate a Clear Strategy for DFS Implementation**
   - **Tree or Graph Representation**: Understand how the problem can be represented as a graph or tree, if applicable. If it's a tree-based problem, focus on nodes and edges. If it's a graph, consider whether the graph is directed, undirected, weighted, or unweighted.
   
   - **DFS Traversal**: 
     - **Recursive Approach**: In the recursive DFS approach, you’ll use the call stack to explore nodes or vertices deeply before backtracking.
     - **Iterative Approach**: If recursion depth might be a concern or to avoid stack overflow, an iterative DFS approach using an explicit stack should be considered.
     
   - **Initial DFS Call**: Specify the starting point of the DFS (typically the root for trees or any arbitrary node for graphs).
   
   - **DFS Movement Logic**:
     - When traversing the structure (graph or tree), DFS will explore as deep as possible along each branch before backtracking.
     - Keep track of visited nodes to avoid revisiting them (important for graphs).
     
   - **Termination Conditions**: Clearly define when DFS should stop:
     - If searching for a specific target node, the search stops when the target is found.
     - For traversal problems, DFS stops when all nodes have been visited or when a certain condition is met.

### 3. **Time Complexity Consideration**
   - **DFS on Graph**: For a graph with \(V\) vertices and \(E\) edges, the time complexity of DFS is \(O(V + E)\) since every vertex and edge is processed once.
   - **DFS on Tree**: In tree-based problems, the time complexity is \(O(n)\), where \(n\) is the number of nodes in the tree.
   - Provide an analysis of the algorithm’s time complexity and explain why it adheres to the optimal expected performance.

### 4. **Testing and Validation**
   - **Sample Case**: Start by testing your solution on the provided sample case to verify the basic correctness of the approach.
   
   - **Additional Test Cases**:
     - **Edge Cases**: Test with edge cases that could challenge the DFS approach, such as:
       - **Empty Input**: An empty graph or tree.
       - **Single Node**: A tree with a single node or a graph with only one vertex.
       - **Large Inputs**: Graphs or trees with a large number of vertices or edges.
       - **Cycle in Graph**: A graph that contains cycles (ensure DFS does not get stuck in an infinite loop).
       - **Disconnected Graph**: A graph with disconnected components (ensure all components are explored if required).
   
   - **Step-by-Step Walkthrough**: For at least one test case, provide a step-by-step walkthrough of the DFS traversal. This should demonstrate the correctness of your approach, showing how the algorithm visits nodes and backtracks.

### 5. **Code Review and Refinement**
   - **Boundary Conditions**: Ensure that DFS never violates the boundaries, such as going out of bounds or revisiting already visited nodes (particularly in graph problems).
   
   - **Correctness**: After implementing the solution, carefully review the code to ensure that DFS correctly explores all the nodes, and handles edge cases as expected.
   
   - **Edge Case Handling**: Make sure the solution handles situations like an empty tree, disconnected graphs, or graphs with cycles effectively. If any component fails, refine the solution iteratively.

### 6. **Refining Recursive/Iterative DFS**
   - **Recursion Depth**: Ensure that recursion depth is appropriate, especially for large trees or graphs. If needed, consider switching to an iterative DFS to avoid issues like stack overflow.
   - **Stack Overflow Prevention**: If recursion depth is a concern (such as for very deep trees or large graphs), use an explicit stack for the DFS traversal instead of recursion.
