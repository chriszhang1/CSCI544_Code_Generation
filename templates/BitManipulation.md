# Bit Manipulation Approach Problem Solving Prompt

## Task Overview
As an experienced software developer, your task is to design and implement a solution in Python 3 for a coding problem using bit manipulation as the primary technique. Follow this structured approach to ensure correctness, optimality, and clarity:


### 1. **Analyze the Problem Requirements and Constraints**
- Determine if the problem involves binary representations, bitwise comparisons, toggling bits, subsets, or parity checks.
- Identify the input and expected output format, and whether the data types are constrained to integers or binary strings.
- Note key constraints:
  - Input size and value bounds (to ensure bit-level operations are efficient)
  - Whether the solution must be performed in constant space or time
  - Whether there are requirements related to bit positions or number of set bits
- Identify edge cases:
  - Input of 0 or 1
  - All bits set or cleared
  - Maximum or minimum possible values

### 2. **Formulate a Bit Manipulation Strategy**
- Determine which bitwise operations apply: AND (&), OR (|), XOR (^), NOT (~), shifts (<<, >>), or masking.
- Consider how to represent and manipulate specific bits (e.g., checking, setting, flipping, or clearing a bit).
- If the problem involves subsets, powers of two, or binary digits, explore using bitmasks.
- Use tricks or patterns (e.g., `x & (x - 1)` to clear the lowest set bit) when relevant.
- Define a clean, repeatable process that transforms the input to the required output using bitwise logic.

### 3. **Time and Space Complexity Consideration**
- Analyze the number of operations relative to the number of bits (typically 32 or 64).
- Determine if your approach is constant time O(1) or logarithmic in value size (O(log n)).
- Ensure that your space usage is minimal, often just integers and counters.
- Justify use of additional space if using lookup tables, masks, or preprocessing.

### 4. **Testing and Validation**
- **Sample Test Case**: Begin with the problem's provided test case to ensure basic correctness.
- **Additional Test Cases**:
  - Edge values: 0, 1, INT_MAX, INT_MIN
  - Inputs with only one set bit, or all bits set
  - Problems involving repeated toggling or counting set bits
- **Step-by-Step Walkthrough**:
  - For a key example, walk through the bitwise operations and show how the state changes at each step.

### 5. **Code Review and Refinement**
- Confirm correct use of all bitwise operators and precedence.
- Review for unnecessary operations or overly complex masking.
- Ensure that edge cases involving sign bits, overflow, or negative values are handled properly.
- Clean up the code:
  - Use meaningful names for masks and bit fields
  - Comment any non-obvious tricks or optimizations
- Test iteratively and profile performance if working with high-volume or large-bit inputs.
