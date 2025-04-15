**This problem must be solved using a _segment tree_. Please follow these steps:**

1. **Verify suitability**  
   Confirm that the required updates and queries (point or range) need `O(log N)` time and that the input sizes \(N\) and \(Q\) are large enough to justify a segment‑tree solution.

2. **Define the segment‑tree node**  
   Specify the aggregate value stored in each node (e.g., sum/min/max) and any lazy‑propagation tag needed for range updates.

3. **Describe the build process**  
   Explain how to construct the tree (recursive or iterative) in `O(N)` time.

4. **Write the update operations**  
   Provide clear algorithms for point updates and (if required) range updates, including how to apply and push lazy tags.

5. **Write the query operations**  
   Show how to answer point or range queries by combining child aggregates while respecting lazy tags.

6. **Handle edge cases**  
   Guard against out‑of‑range indices, empty segments, and degenerate inputs such as `N = 1`.

7. **Analyze complexity**  
   Prove that each update and query runs in `O(log N)` time and that the tree uses `O(N)` space (≈ `4 × N` nodes).

8. **Implement the full program**  
   Produce clean, well‑commented code (e.g., C++ 17) that follows the algorithms above.

9. **Test thoroughly**  
   Run all sample cases, add edge cases (full‑range updates, single‑element queries, alternating operations), and verify correctness.  
   If any test fails, refine your design and code until all requirements are satisfied.