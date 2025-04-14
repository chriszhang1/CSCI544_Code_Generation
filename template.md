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
      "Begin by thoroughly analyzing the problem requirements and constraints. Clearly define the problem’s goal and identify the variables involved.",
      "Formulate a clear strategy for pointer selection, justifying the choice of pointers, and explaining how they efficiently solve the problem. Ensure the approach is optimal.",
      "Define the precise movement logic for the pointers. Detail the conditions under which each pointer is incremented or decremented, including termination conditions and exit criteria.",
      "Address edge case handling, ensuring that all potential pitfalls (such as empty input, minimal input, or extreme values) are properly managed and do not cause out-of-bound errors.",
      "Implement the solution with clean, well-commented code that follows best practices. Make sure the pointer movement logic is correctly applied and easy to follow.",
      "Analyze the time and space complexity of your approach. Ensure it meets the problem's constraints and is optimal for large inputs.",
      "Test the solution using provided sample test cases, as well as additional test cases that cover normal scenarios, edge cases, and extreme values. For at least one test case, provide a step-by-step walkthrough.",
      "After implementation, review the code to ensure correctness. Make sure that the pointers never violate boundaries, all test cases pass, and the logic aligns perfectly with the approach.",
      "If any component fails (e.g., incorrect edge-case handling or pointer mismanagement), refine the solution iteratively until it is robust and reliable."
    ]
}
```
