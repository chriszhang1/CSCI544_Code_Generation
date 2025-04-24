# Tree Approach Problem-Solving Prompt

## Task Overview  
As an experienced software developer, your task is to design and implement an efficient solution in **Python 3** for a coding problem whose **core data structure is a tree** (binary, N-ary, trie, segment tree, etc.). Follow this structured checklist to ensure correctness, clarity, and performance.

### 1. Analyze the Problem Requirements and Constraints
- **Tree Type**  
  - Is the tree binary, k-ary, rooted/unrooted, balanced, BST, heap, trie, segment tree, prefix tree, etc.?
- **Input / Output**  
  - How is the tree given—edge list, parent array, child pointers, level-order list, adjacency map?  
  - What must you return—value, node reference, list, boolean, rebuilt tree?
- **Constraints**  
  - Number of nodes `n`  
  - Key/value range  
  - Height limits  
  - Memory or recursion-depth limits (Python’s default recursion cap ≈ 1000)
- **Edge Cases**  
  - Empty tree / single node  
  - Degenerate (linked-list-like) trees  
  - Duplicate keys or values  
  - Unbalanced vs balanced trees  
  - Very deep recursion (→ iterative or tail-rec optimization)

### 2. Formulate the Tree Strategy
1. **Representation**  
   - Decide whether to convert input into a convenient node class, adjacency list, or keep as given.  
   - For derived structures (e.g., segment tree), plan build and storage layout.

2. **Traversal Choice**  
   - **DFS (pre/in/post-order)** for local aggregation, path problems, recursion ease.  
   - **BFS / level-order** for shortest path, width operations, parallel processing.  
   - **Iterative vs recursive**: use explicit stack/queue if recursion depth can exceed limits.

3. **Algorithm Pattern**  
   - **Divide & Conquer**: combine left/right subtree results.  
   - **Dynamic Programming on Trees**: memoize or return multiple values per call.  
   - **Lowest Common Ancestor, Heavy-Light, Euler Tour** when queries are numerous.  
   - **Morris traversal / threaded binary tree** to achieve `O(1)` space if required.

4. **State per Node**  
   - Define exactly what each recursive call or iterative visit must return (e.g., height, diameter, max path sum, balanced flag).

5. **Modification vs Query**  
   - For mutable operations (insert/delete), ensure structural invariants (BST order, heap property) are restored efficiently.

### 3. Time and Space Complexity Considerations
- **Traversal**: `O(n)` time, `O(h)` space (recursion stack) or `O(n)` if using explicit auxiliary structures.  
- **Balanced Operations** (AVL/Red-Black insert/delete): `O(log n)` time each, `O(1)` extra space.  
- **Segment / Fenwick Trees**: Build: `O(n)`; point/range query or update: `O(log n)`.  
- **Trie Operations**: Insert/lookup `O(L)` where `L` = key length.

Always reason:  
- **Worst-case height `h`** (skewed tree → `h ≈ n`) vs balanced (`h ≈ log n`).  
- Recursion depth vs Python’s limit; convert to iterative if `h` may exceed 900–1000.  
- Memory used by auxiliary arrays, maps, or parent pointers you create.

### 4. Testing and Validation
1. **Sample Test Case(s)** – validate against provided examples.  
2. **Additional Tests**  
   - **Edge structure**: empty, single node, fully skewed left/right.  
   - **Balanced vs unbalanced** cases.  
   - **Duplicate values / equal keys** if allowed.  
   - **Max-depth stress**: `n ≈ 10⁵` nodes forming a line.  
3. **Step-by-Step Walkthrough**  
   - Pick one non-trivial tree; show pre/in/post-order traversal or dynamic return values at each step.  
   - Visualize how you update global / non-local variables (e.g., `maxPath`) during recursion.

### 5. Code Review and Refinement
- **Safety**: guard against `None` before accessing `.left` / `.right`.  
- **Recursion vs Iteration**: switch if hitting recursion-depth errors.  
- **Avoid Redundancy**: compute attributes once per node; reuse returned tuples.  
- **Naming & Clarity**: `node`, `leftHeight`, `rightHeight`, etc.  
- **Inline Comments**: explain the **invariant** each recursive call maintains.  
- **Complexity Check**: verify each node is visited ≤ `O(1)` times.  
- **Refactor**: extract helper functions for readability; unit-test them individually.
