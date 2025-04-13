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

## Template Fields Explanation

- **title**: The main question prompt, with a placeholder for the category
- **description**: A clear statement of what needs to be solved
- **input_format**: Detailed specification of input parameters and their characteristics
- **output_format**: Clear description of expected output format and structure
- **constraints**: Important limitations and assumptions for the solution
- **examples**: One or more example cases with inputs, outputs, and optional explanations
- **starter_code**: Initial code structure provided (if any)

## Specialized Templates

### Dynamic Programming Template

```json
{
    "template": {
      "title": "You are tasked to solve this problem using dynamic programming:",
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
      "starter_code": "Optional. The function/class definition as provided (e.g., on LeetCode).",
      "dp_requirements": [
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
}
```

### DP Requirements Explanation

The DP template includes all the general template fields plus specific requirements for dynamic programming solutions:

1. **Optimal Substructure Analysis**: Verify if the problem can be broken down into smaller subproblems
2. **State Definition**: Clearly specify what parameters define a subproblem
3. **Recurrence Relation**: Define how to compute larger problems from smaller ones
4. **Base Cases**: Identify the smallest subproblems that can be solved directly
5. **Implementation Strategy**: Choose between top-down or bottom-up approach
6. **Edge Case Handling**: Ensure all boundary conditions are properly handled
7. **Complexity Analysis**: Verify the solution meets time and space constraints
8. **Implementation**: Write complete, correct code
9. **Testing**: Verify the solution works on all test cases
