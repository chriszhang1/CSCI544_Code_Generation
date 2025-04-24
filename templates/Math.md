# Math Coding Problem Solving Prompt

## Task Overview
As an experienced software developer, your task is to design and implement a solution in Python 3 for the given mathematical coding problem. The following steps should be followed to ensure a comprehensive and correct solution:

### 1. **Analyze the Problem Requirements and Constraints**
   - Thoroughly review the problem's input, output, and constraints.
   - Identify any specific edge cases, such as minimal or extreme values, empty inputs, or other relevant corner cases.
   - Understand the problem's complexity requirements and ensure the solution operates efficiently (typically O(n), O(log n), or other suitable complexities for mathematical operations).

### 2. **Formulate a Clear Strategy for Solving the Problem**
   - **Problem Breakdown**: Clearly define what mathematical concepts are involved (e.g., number theory, algebra, geometry, probability, etc.).
   - **Mathematical Formula or Approach**: Identify any formulas or approaches that can directly help in solving the problem. For example, if it's a problem involving prime numbers, the Sieve of Eratosthenes or trial division might be useful. If it's an optimization problem, consider greedy methods, dynamic programming, or backtracking.
   
   - **Efficient Approach**: Discuss the optimal approach to solve the problem. Consider any mathematical tricks or optimizations that can reduce unnecessary computation.
     - For example, for a problem involving large numbers, think of using modular arithmetic to avoid overflow.
     - If the problem involves large datasets, consider sorting or binary search for faster results.
   
   - **Edge Case Handling**: Ensure that edge cases are handled correctly, such as:
     - Smallest or largest possible values
     - Divisibility issues (e.g., division by zero)
     - Negative values (if applicable)
     - Boundary conditions (e.g., range limits)

### 3. **Time Complexity Consideration**
   - For problems involving numbers or large datasets, always aim for an efficient solution, usually:
     - **O(n)** or **O(log n)** for algorithms that involve direct calculations, searching, or sorting.
     - **O(n^2)** or **O(n log n)** if nested iterations are involved, but try to optimize if possible.
     - For optimization problems, **dynamic programming** solutions can often reduce time complexity.
   
   - Provide an analysis of the algorithm’s time complexity and explain why it adheres to the expected performance.

### 4. **Testing and Validation**
   - **Sample Case**: Start by testing your solution on the provided sample case to verify the correctness of your approach.
   
   - **Additional Test Cases**:
     - **Edge Cases**: Test with edge cases that could challenge the algorithm, such as:
       - **Zero** or **Negative numbers**: These can have special conditions, such as not allowing division by zero.
       - **Large Numbers**: Use values at the upper bounds of input limits.
       - **Boundary cases**: Test the minimum and maximum values.
   
   - **Step-by-Step Walkthrough**: For at least one test case, provide a step-by-step walkthrough of the logic used to solve the problem. This can include intermediate values, iterations, or formula applications.

### 5. **Code Review and Refinement**
   - **Boundary Conditions**: Make sure the algorithm handles edge cases like empty inputs, large values, or unusual number properties (e.g., prime numbers, even/odd properties, etc.).
   
   - **Optimization**: After implementing the solution, review the code for potential optimization opportunities. This could include improving space complexity, avoiding redundant calculations, or reducing computational complexity.
   
   - **Validation**: Ensure all test cases pass, including edge cases. Confirm that the code handles all scenarios correctly and efficiently.
