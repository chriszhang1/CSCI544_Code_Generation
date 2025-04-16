# Sorting Approach Problem Solving Prompt

## Task Overview
As an experienced software developer, your task is to design and implement a solution in Python 3 for the given coding problem using sorting as the core technique. The following structured approach will help ensure correctness, efficiency, and completeness:

### 1. **Analyze the Problem Requirements and Constraints**
- Carefully read the problem statement to understand the required input/output format.
- Determine whether the problem requires full sorting, partial sorting (e.g., top-k), or sorting with a custom comparator.
- Understand the data type (e.g., integers, strings, tuples, custom objects) and the desired sorting order.
- Pay attention to constraints such as:
  - Size of the input (may affect time complexity decisions)
  - Range of values (may enable linear-time sorting)
  - Stability of sorting (whether relative order matters)
- Identify edge cases:
  - Empty list or array
  - Single-element input
  - Already sorted or reverse sorted data
  - All values the same vs. all values unique

### 2. **Formulate a Clear Strategy for Sorting**
- Algorithm Selection: Choose an efficient sorting algorithm based on the problem's size and constraints:
- Custom Sort Logic:Define a lambda or key function for multi-level or non-standard sorting (e.g., sort by frequency then value).
- In-Place vs. Out-of-Place: Determine whether sorting should modify the original data or return a new structure.

### 3. **Time and Space Complexity Consideration**
- Evaluate the time complexity:
  - O(n log n) for comparison-based sorting
  - O(n) for specialized counting-based sorts
- Justify space usage:
  - Is extra space allowed or must the sort be in-place?
  - Does the language-provided sort meet the constraints?
- Ensure that your solution fits within the problem's limits

### 4. **Testing and Validation**
- Sample Test Case: Verify correctness against the sample input provided in the problem.
- Additional Test Cases:
  - Random input with mixed values
  - Edge cases:
    - Empty array
    - Single-element array
    - Already sorted / reverse sorted
    - Large input size (performance test)
    - Inputs requiring custom sorting logic (e.g., sort by frequency or absolute value)
- Step-by-Step Walkthrough:
  - Choose at least one complex case and explain the sorting transformation step-by-step.

### 5. **Code Review and Refinement**
- Validate correctness by re-checking sorting conditions (ascending/descending, stability, key correctness).
- Ensure that all edge cases are handled gracefully.
- Keep code readable: Avoid unnecessary re-sorting or list copying
- Iterate as needed if logic fails on any test case and refine the solution until the implementation is robust and reliable.
