# Backtracking Approach Problem Solving Prompt

## Task Overview
As an experienced software developer, your task is to design and implement a solution in Python 3 for the given coding problem using backtracking as the primary technique. Follow this structured approach to ensure correctness, efficiency, and clear reasoning:

### 1. **Analyze the Problem Requirements and Constraints**
- Understand the type of problem being solved: permutations, combinations, subsets, path finding, constraint satisfaction, etc.
- Identify the input and expected output format.
- Note constraints such as:
  - Maximum input size (to determine feasibility of recursive exploration)
  - Valid solution criteria (what makes a partial or complete result valid)
  - Uniqueness requirements (e.g., avoid duplicate outputs)
- Identify edge cases:
  - Empty input
  - Repeated values or characters
  - No valid solution exists

### 2. **Define the Backtracking State and Structure**
- Clearly define what constitutes a "state" (e.g., a path, prefix, configuration).
- Determine the choices available at each step (e.g., which elements can be added to the current state).
- Decide how to build the state incrementally (recursive structure).
- Define base cases:
  - When a valid solution is found (and should be added to the result list)
  - When recursion should terminate (invalid or complete state)
- Ensure backtracking logic undoes decisions (e.g., popping from a list or unmarking visited elements).

### 3. **Time and Space Complexity Consideration**
- Estimate time complexity based on the branching factor and recursion depth:
- Account for space complexity due to recursion stack and result storage.
- Optimize by pruning unnecessary branches early (e.g., stop recursion if constraints are violated).

### 4. **Testing and Validation**
- Sample Test Case: Begin with the sample case(s) from the problem statement.
- Additional Test Cases:
  - Edge cases: empty input, no valid path, input with duplicates
  - Minimal and maximal input sizes
  - Recursive depth extremes (to test stack behavior)
- Step-by-Step Walkthrough:
  - Choose one example and walk through recursive calls and backtracking decisions step-by-step.

### 5. **Code Review and Refinement**
- Verify that all valid outputs are generated (no duplicates unless allowed).
- Confirm that the backtracking correctly undoes previous choices.
- Ensure that pruning conditions are sound and improve performance.
- Maintain clean and readable code:
  - Clear function names and parameters
  - Well-placed comments explaining recursion, base cases, and pruning
- Iterate to resolve issues and improve performance where necessary.
