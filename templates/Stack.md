# Stack Approach Problem Solving Prompt

## Task Overview
As an experienced software developer, your task is to design and implement a solution in Python 3 for a coding problem using stack as the primary data structure. Follow this structured approach to ensure correctness, clarity, and efficiency:

### 1. **Analyze the Problem Requirements and Constraints**
- Identify if the problem benefits from LIFO behavior, such as nested structure parsing, recent element tracking, or matching pairs.
- Understand the expected input and output format.
- Note problem constraints that may impact stack depth or memory use:
  - Input size and range
  - Requirement for matching structure (e.g., parentheses, tags)
  - Monotonic behavior or comparative relationships
- Identify edge cases:
  - Empty input
  - Unmatched or incomplete structures
  - Duplicate or repeating patterns

### 2. **Formulate the Stack Strategy**
- Decide what type of stack is appropriate: basic stack, monotonic (increasing/decreasing), or custom-object stack.
- Define what each stack entry represents and when it should be pushed or popped.
- Determine the conditions that trigger stack operations (e.g., compare top of stack with current element).
- Ensure the stack correctly models the problem’s logical structure or constraints.
- For monotonic stacks, define direction (increasing or decreasing) and behavior on duplicates or equal values.

### 3. **Time and Space Complexity Consideration**
- Analyze how many times each element is pushed or popped:
  - For monotonic stacks, elements are typically processed in amortized O(1) time.
- Evaluate total space usage of the stack in the worst case.
- Ensure that all stack operations remain within acceptable time and space limits for large inputs.

### 4. **Testing and Validation**
- **Sample Test Case**: Start by validating your implementation on provided example(s).
- **Additional Test Cases**:
  - Balanced and unbalanced structures
  - Deep nesting or large stack depth
  - Inputs triggering maximum or minimum stack behavior
- **Step-by-Step Walkthrough**:
  - Choose a sample and walk through the stack operations visually, showing each push/pop and stack state.

### 5. **Code Review and Refinement**
- Confirm that the stack never overflows or operates out of bounds.
- Check for redundant stack operations that can be optimized.
- Verify correctness of all conditions related to pushing, popping, and result updates.
- Maintain readable, clean code:
  - Use descriptive variable names for stack elements
  - Add comments where stack behavior may not be immediately obvious
- Test iteratively and refactor to improve clarity or performance.
