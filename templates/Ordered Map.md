9. **This problem must be solved using an _ordered map_ (balanced‑BST–based container such as `std::map`, `std::set`, or an order‑statistic tree). Please follow these steps:**
   
   1. **Verify suitability**  
      Confirm that the operations required (ordered insertion, deletion, predecessor/successor lookup, rank, k-th element, range count, etc.) demand `O(log N)` performance and rely on key ordering rather than mere hashing.
   
   2. **Choose an ordered‑map variant**  
      - Standard balanced BST (`std::map` / `std::set`) for insert / erase / lower_bound.  
      - Order‑statistic tree (Fenwick‑tree‑indexed set or PBDS `tree`) if rank or k‑th queries are needed.  
      - Augmented node fields (subtree size, extra aggregates) when custom order statistics are required.
   
   3. **Define key and value types**  
      Specify what constitutes the key (integer, pair, string) and what auxiliary data (frequency, aggregate value) is stored.
   
   4. **Describe core operations**  
      Provide algorithms for:  
      - **Insertion / deletion:** `O(log N)` via BST rotations.  
      - **Ordered queries:** predecessor, successor, `lower_bound`, `upper_bound`.  
      - **Rank / k‑th element:** if using an order‑statistic tree, explain how subtree sizes support these queries.
   
   5. **Handle duplicates**  
      Decide whether to allow duplicate keys.  
      - Use a multiset or store a frequency counter in each node.  
      - Clarify how duplicates affect rank and deletion logic.
   
   6. **Edge‑case handling**  
      Guard against empty structure, queries outside key range, and overflow of iterator arithmetic.
   
   7. **Analyze complexity**  
      - **Insert / erase / search / rank / k‑th:** `O(log N)` each.  
      - **Space:** `O(N)` nodes plus overhead for parent/child pointers.
   
   8. **Implement the full program**  
      Write clean, well‑commented code (e.g., C++ 17). Prefer RAII and iterators; avoid undefined behavior when erasing while iterating.
   
   9. **Test thoroughly**  
      - Verify all sample cases.  
      - Add edge cases: empty map, duplicate keys, min/max key queries, consecutive insert‑erase cycles.  
      - Stress‑test with `10⁵+` mixed operations.  
      If any test fails, debug and refine until all requirements are met.