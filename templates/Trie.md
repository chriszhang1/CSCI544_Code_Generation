Solve the problem with a Trie (prefix tree). Before you code, think about the algorithm using the checklist below.

1. Understand the Task
- Identify the alphabet, max word length/count, and required operations (insert, search, prefix‑count, delete, etc.).
- Clarify all edge cases (empty string, duplicates, non‑existent keys).

2. Why a Trie?
- Shared‑prefix storage ⇒ **O(total chars)** per operation.
- Show a quick memory/time estimate to prove it fits the constraints.

3. Design
```
TrieNode {
  children[Σ]   // array or hashmap by alphabet size
  bool isEnd    // word ends here
  int  subCnt   // words in subtree (if needed)
}
```

- Add extra fields (frequency, failure links) only if the task requires them.

4. Algorithms

5. Implementation Tips
   	•	Use iterative loops to avoid recursion depth limits.
   	•	Handle empty strings and index bounds safely.
   	•	Keep null checks to avoid segmentation faults.

6. Complexities
   	•	Time: O(total chars) overall.
   	•	Space: O(total chars) nodes (lower if using a compressed Trie).

7. Test & Verify
   	1.	Trace all sample cases step‑by‑step.
   	2.	Add edge cases: empty word, longest word, high‑volume prefix queries.
   	3.	Confirm every test passes; if not, refine and retest.