# Binary Search Problem Solving Prompt

## Task Overview
As an experienced software developer, your task is to design and implement a solution in Python 3 for the given coding problem using **Binary Search**. The following steps should be followed to ensure a comprehensive and correct solution:

### 1. **Analyze the Problem Requirements and Constraints**
   - Thoroughly review the problem's input, output, and constraints.
   - Identify any specific edge cases, such as minimal or extreme values, empty inputs, or other relevant corner cases.
   - Understand the problem's complexity requirements and ensure the solution operates efficiently (typically O(log n) for binary search).

### 2. **Formulate a Clear Strategy for Binary Search**
   - **Binary Search Applicability**: Binary search is typically used for problems involving sorted arrays, where you need to efficiently search for an element or compute an optimal answer (e.g., finding the target element, searching for a boundary condition, or solving optimization problems).
   
   - **Binary Search Framework**:
     - **Input**: A sorted array or range (it can be either increasing or decreasing).
     - **Search Space**: The search space is usually halved with each iteration.
     - **Termination Conditions**: The search continues as long as the search space is valid (i.e., `left <= right`). The loop terminates when the element is found, or when the search space is exhausted.
   
   - **Middle Element**:
     - In each step, calculate the middle element using `mid = left + (right - left) // 2` to avoid overflow.
   
   - **Pointer Movement Logic**: 
     - If the middle element equals the target, return the result.
     - If the target is smaller than the middle element, move the `right` pointer to `mid - 1` (search left half).
     - If the target is larger than the middle element, move the `left` pointer to `mid + 1` (search right half).
   
   - **Edge Case Handling**: Handle cases where the search space is empty, the array is already sorted, or when the element doesn't exist in the array.

### 3. **Time Complexity Consideration**
   - **Time Complexity**: The time complexity of binary search is (O(log n)), where `n` is the number of elements in the array. This is because with each step, the search space is halved.
   
   - **Space Complexity**: The space complexity of binary search is (O(1)) if done iteratively, as it only requires a constant amount of space for the pointers. If implemented recursively, it may require (O(log n)) space for the recursion stack.

   - **Worst-Case Scenario**: In the worst case, binary search may have to examine all the levels of the search space, but it still operates in logarithmic time.

### 4. **Testing and Validation**
   - **Sample Case**: Start by testing your solution on the provided sample case to verify the correctness of your approach.
   
   - **Additional Test Cases**:
     - **Edge Cases**: Test with edge cases that could challenge the algorithm, such as:
       - **Empty Input**: An empty array or range.
       - **Single Element**: An array with only one element.
       - **Large Numbers**: Use values at the upper bounds of input limits.
       - **No Solution**: Test cases where the element isn't present in the array.
   
   - **Step-by-Step Walkthrough**: For at least one test case, provide a step-by-step walkthrough of the binary search. This can include intermediate values of `left`, `right`, and `mid`, and how the pointers change during the process.

### 5. **Code Review and Refinement**
   - **Boundary Conditions**: Ensure that the algorithm handles edge cases like empty arrays, arrays with one element, and cases where the element isn't found.
   
   - **Optimization**: After implementing the solution, review the code for potential optimization opportunities, like ensuring the code uses the correct integer division for calculating the middle index.
   
   - **Validation**: Ensure all test cases pass, including edge cases. Confirm that the code handles all scenarios correctly and efficiently.
