# Code Generation Prompt Template

```json
{
    "template": {
      "title": "You are tasked to use <category> to solve the following question:",
      "description": "A concise statement describing the task to solve.",
      "input_format": "Describe the inputs clearly, including types, ranges, and edge case behavior.",
      "output_format": "Describe the expected output, including structure and format.",
      "constraints": "List all constraints and assumptions (e.g., value bounds, input size limits, special guarantees).",
      "examples": [
        {
          "input": "...",
          "output": "...",
          "explanation": "Optional explanation to illustrate edge cases or clarify behavior."
        }
      ],
      "starter_code": "Optional. The function/class definition as provided (e.g., on LeetCode)."
    }
}
```

## Category Specific Prompts

### Dynamic Programming
```json
{
    "category_specific_prompt": [
      "Determine whether the problem has an optimal substructure and overlapping subproblems.",
      "Clearly define the DP state. What parameters uniquely describe a subproblem?",
      "Write the recurrence relation (transition formula) that connects larger problems to smaller ones.",
      "Identify and explain the base cases of your DP table or recursion.",
      "Choose whether to use top-down recursion with memoization or bottom-up tabulation.",
      "Carefully handle edge cases and ensure your solution avoids out-of-bound errors.",
      "Analyze the time and space complexity of your approach and verify it fits within the constraints.",
      "After planning your solution, implement a complete, correct program.",
      "Test your code on sample input and ensure all requirements above are met. If not, revise your solution accordingly."
    ]
}
```

### Greedy Algorithm
```json
{
    "category_specific_prompt": [
      "Understand the Problem: Identify key variables, constraints, and optimization goals",
      "Consider Greedy Approach: Formulate a strategy for making locally optimal choices",
      "Prove Correctness: Verify if the greedy approach leads to global optimality",
      "Develop Algorithm: Outline step-by-step approach and required data structures",
      "Code Implementation: Write clean, well-commented code with proper edge case handling",
      "Analyze Complexity: Calculate and explain time and space complexity",
      "Test with Examples: Verify solution on provided and additional test cases"
    ]
}
```

### Divide and Conquer
```json
{
    "category_specific_prompt": [
      "Validate D&C applicability through problem decomposition analysis",
      "Build recursive structure with complexity proofs",
      "Design merge strategy with cross-boundary handling",
      "Verify against all sample test cases through symbolic execution",
      "Generate code implementing the verified solution",
      "Display the entire process of testing with each sample input",
      "Include time/space complexity comments in Big-O notation",
      "Ensure function signature matches the given problem",
      "Manually execute solution on all test cases and verify correctness"
    ]
}
```

### Two Pointer
```json
{
    "category_specific_prompt": [
      "Analyze the Problem: Identify the input, output, constraints, and optimization goals",
      "Formulate Pointer Strategy: Select appropriate pointers and justify their choice for efficiency",
      "Define Pointer Movement: Specify conditions for incrementing/decrementing pointers and termination logic",
      "Edge Case Handling: Ensure proper management of edge cases and boundary conditions",
      "Implement Solution: Write clean, well-commented code following the pointer strategy",
      "Analyze Complexity: Explain time and space complexity within the problem's constraints",
      "Test with Examples: Validate the solution on provided and additional test cases, with step-by-step pointer movements",
      "Review and Refine: Iterate on the solution to fix any pointer mismanagement or edge case issues"
    ]
}
```
### Math
```json
{
  "category_specific_prompt": [
    "Analyze the mathematical problem and identify the key concepts involved (e.g., algebra, geometry, number theory, combinatorics, probability).",
    "Break down complex problems into simpler mathematical operations or concepts.",
    "Consider mathematical properties, formulas, or theorems that apply to the problem.",
    "Identify patterns or relationships that can be expressed mathematically.",
    "Develop a clear mathematical approach before coding, using mathematical notation if helpful.",
    "Implement the solution with attention to numerical precision, potential overflow issues, and edge cases.",
    "Test the solution with boundary values and special cases to ensure mathematical correctness.",
    "Analyze the time and space complexity in terms of mathematical operations.",
    "Verify that your solution handles all constraints and scales appropriately for the input size."
  ]
}
```

### Depth-first Search
```json
{
  "category_specific_prompt": [
    "Identify the problem structure (tree, graph, matrix, etc.) and determine if DFS is appropriate.",
    "Define what constitutes a 'state' in your search and what you're looking for (path, configuration, etc.).",
    "Determine the traversal order priority (which neighbors to visit first) if relevant.",
    "Design the recursive or iterative DFS function with clear parameters and return values.",
    "Implement a mechanism to track visited states to avoid cycles or redundant exploration.",
    "Handle backtracking appropriately if the problem requires finding paths or configurations.",
    "Consider space optimization: can you use path pruning or early termination conditions?",
    "Code the solution with attention to stack overflow risks for large inputs.",
    "Analyze the time complexity (usually O(V+E) for graphs) and space complexity (often O(V) for the recursion stack).",
    "Test the solution with diverse inputs including edge cases like disconnected components or empty structures."
  ]
}
```

### Hash Table
```json
{
  "category_specific_prompt": [
    "Analyze if the problem involves fast lookups, frequency counting, grouping, or finding duplicates/unique elements.",
    "Identify what should be used as keys and values in your hash table.",
    "Determine the appropriate hash table structure (hash map, hash set, counter, etc.) for your needs.",
    "Consider potential key collisions and how to handle them if implementing a custom hash table.",
    "Plan the algorithm to minimize lookups and updates to the hash table.",
    "Implement the solution with clear hash table initialization and operations.",
    "Address edge cases like empty inputs, duplicate keys, or non-existent lookups.",
    "Analyze the time complexity (usually O(1) for lookups/inserts) and space complexity (O(n) for storing n elements).",
    "Consider alternative data structures if constraints suggest hash tables might not be optimal.",
    "Test the solution with various input patterns, including worst-case scenarios for hash functions."
  ]
}
```

### Binary Search
```json
{
  "category_specific_prompt": [
    "Verify that the problem involves searching in a sorted array or can be transformed into such a problem.",
    "Identify the search space and what you're searching for (a value, an index, a boundary condition).",
    "Define precise low and high boundary indices for your search space.",
    "Establish a clear mid-point calculation method (consider potential integer overflow for large arrays).",
    "Formulate the condition that determines which half of the search space to eliminate.",
    "Handle edge cases where the target might not exist in the array.",
    "Implement loop termination conditions carefully to avoid infinite loops.",
    "Consider whether the problem requires finding exact matches or boundaries (e.g., lower/upper bound).",
    "Analyze the time complexity (typically O(log n)) and constant-space complexity advantage.",
    "Test the solution with various inputs including edge cases like empty arrays, single elements, or duplicates."
  ]
}
```

### Breadth-first Search
```json
{
  "category_specific_prompt": [
    "Determine if the problem involves finding shortest paths, level-order traversals, or exploring nearest neighbors first.",
    "Define what constitutes a 'state' in your search space and what goal states look like.",
    "Identify the data structure for the queue (simple queue, deque, priority queue for weighted paths).",
    "Design a system to track visited states to prevent cycles and redundant exploration.",
    "Implement the BFS algorithm with proper queue operations and neighbor generation.",
    "Consider optimizations like bi-directional BFS for certain problems.",
    "Handle level tracking if the problem requires counting steps or grouping by distance.",
    "Analyze the time complexity (usually O(V+E) for graphs) and space complexity (often O(V) for the queue).",
    "Address edge cases like disconnected components, empty structures, or multiple valid solutions.",
    "Test your solution with diverse inputs to verify correctness and efficiency."
  ]
}
```

### Backtracking
```json
{
  "category_specific_prompt": [
    "Check if the problem involves exploring combinations, permutations, subsets, or constraint satisfaction, which are suitable for backtracking.",
    "Define what a valid state looks like and how it represents progress toward the final solution.",
    "Determine the available choices at each step and ensure they follow problem constraints.",
    "Establish clear base cases to stop recursion and capture valid results.",
    "Implement the recursive structure by making a choice, recursing, and undoing the choice to backtrack.",
    "Use pruning to skip invalid or unnecessary branches early in the recursion.",
    "Handle edge cases like empty input, duplicate elements, or situations where no solution exists.",
    "Analyze the time complexity based on branching factor and recursion depth, typically exponential.",
    "Test your solution on a range of inputs including base cases, duplicates, and edge scenarios.",
    "Walk through a complete recursive path to verify that backtracking and result recording work as expected."
  ]
}
```

### Sort
```json
{
  "category_specific_prompt": [
    "Identify the nature of sorting required: full sort, partial sort, or custom order (e.g., lexicographic, frequency-based, tuple prioritization).",
    "Choose the most appropriate sorting algorithm based on input size, constraints, and stability requirements.",
    "If custom sorting is needed, define a comparator function that captures the required sorting logic.",
    "Ensure sorting is applied to the correct data structure: arrays, tuples, objects, or nested collections.",
    "Optimize for in-place sorting or minimal memory usage if space constraints are tight.",
    "Handle duplicates, nulls, and edge cases like empty arrays or uniform values with care.",
    "Analyze time complexity and space usage.",
    "Verify that your sorting approach maintains any necessary invariants (e.g., relative order for stable sorts).",
    "Implement and test the sorting logic on diverse examples: minimal, maximal, already sorted, reverse sorted, and randomly shuffled inputs.",
    "Refactor and document the sorting logic clearly for readability and maintainability, especially if using custom comparators or multi-level sorting."
  ]
}
```

