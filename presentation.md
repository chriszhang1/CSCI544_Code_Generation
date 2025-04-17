# Enhancing Code Generation Accuracy Through Classification and Tailored Prompt Engineering

## Introduction
Good afternoon everyone,

We are group 7, presenting our project on enhancing code generation accuracy through classification and tailored prompt engineering.

## Motivation
General prompts often fail to address specific algorithmic demands. By tailoring prompts to problem types, we can guide models toward more accurate solutions.

## Problem Definition
We tackled three challenges:
1. **Problem Classification** – categorizing based on algorithmic characteristics
2. **Prompt Tailoring** – creating specialized prompts
3. **Evaluating Model Output** – automated performance assessment

## Solution Overview
Our approach:
1. Input LeetCode problems
2. Classify for algorithmic tags
3. Generate tailored prompts
4. Submit to GPT-4o Mini
5. Evaluate on LeetCode

## Automatic Problem Classification
We fine-tuned the DeepSeek‑R1‑Distill‑Llama‑8B model to identify algorithmic categories from problem descriptions.

## Current Tags
We classify problems into categories like:
- Dynamic Programming
- Math
- Tree
- Depth-first Search
- Greedy
- Hash Table
- Binary Search

## Tag-based Prompt Generation
Our specialized prompts highlight constraints, formats, pitfalls, and encourage structured reasoning.

### Template Creation Process
We developed a systematic approach for template creation. Each template covers problem analysis, patterns, implementation, edge cases, and complexity.

For each category, we focus on unique aspects: state transitions in DP, traversal methods in Trees, movement strategies in Two Pointers, and state management in Backtracking.

We integrate LeetCode patterns, optimization techniques, and common mistakes while maintaining consistency through testing and refinement.

## Automatic Prompt Evaluation
Our evaluation process:
1. Fetch problems
2. Classify
3. Generate raw and tailored prompts
4. Submit to GPT-4o Mini
5. Evaluate on LeetCode

## Results
Our evaluation showed:
- 97.44% accuracy in Tree problems
- Over 80% in Two Pointers, String, and Hash Table
- Significant improvement over raw prompts

## Conclusion
Our systematic approach - from classification to template creation to evaluation - provides a robust framework for improving AI code generation accuracy.

Thank you, and we welcome your questions. 