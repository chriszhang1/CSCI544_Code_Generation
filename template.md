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
