# Heap Approach Problem Solving Prompt

## Task Overview
As an experienced software developer, your task is to design and implement a solution in Python 3 for a coding problemusing heap as the primary technique. Follow this structured approach to ensure correctness, performance, and clean design:

### 1. **Analyze the Problem Requirements and Constraints**
- Identify whether the problem requires dynamic access to minimum, maximum, or top-k elements.
- Understand the expected input and output format and how frequently elements will be added or removed.
- Note important constraints:
  - Input size and update frequency (affects push/pop overhead)
  - Type of priority (natural order, inverse, or custom priority)
  - Whether all elements must be stored or only a subset (e.g., top-k tracking)
- Identify edge cases:
  - Empty input
  - Repeated values
  - Heap updates during iteration or conditional removals

### 2. **Formulate the Heap Strategy**
- Choose between a min-heap or max-heap depending on whether the smallest or largest element needs priority.
- Use Python’s heapq for a min-heap; for max-heap behavior, push negative values or use tuples with inverted keys.
- Consider using tuples for complex sorting or custom prioritization.
- Maintain heap size if the problem requires only partial tracking (e.g., top-k elements).
- Avoid unnecessary heapify operations by maintaining heap invariants during incremental updates.

### 3. **Time and Space Complexity Consideration**
- Understand that each push or pop operation has O(log n) complexity.
- Evaluate the total number of heap operations in the context of the problem.
- Consider space usage when storing multiple fields in the heap (tuples, objects).
- Optimize by bounding the heap size when possible (e.g., discarding low-priority items).

### 4. **Testing and Validation**
- **Sample Test Case**: Validate the heap logic using the example(s) provided in the problem.
- **Additional Test Cases**:
  - Inputs with ties or duplicates in priority
  - Inputs requiring frequent insertions and removals
  - Edge cases with no input or very large inputs
- **Step-by-Step Walkthrough**:
  - Walk through the heap structure step-by-step for at least one case, showing each push/pop and final ordering.

### 5. **Code Review and Refinement**
- Ensure the heap maintains the correct order and priority under all conditions.
- Avoid pushing unnecessary data or performing redundant comparisons.
- Confirm correctness when using negated values or complex tuples for custom behavior.
- Keep your code readable and maintainable:
  - Use descriptive variable names for heap elements and structure
  - Include comments to clarify heap setup and transformation logic
- Refactor as needed for efficiency and test edge cases iteratively.
