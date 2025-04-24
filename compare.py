import json
from typing import Dict, Any, Optional

def load_json_file(filename: str) -> Dict[str, Any]:
    """Load a JSON file and return its contents as a dictionary."""
    with open(filename, 'r') as f:
        return json.load(f)

def safe_compare(a: Optional[int], b: Optional[int]) -> bool:
    """Safely compare two values that might be None."""
    if a is None and b is None:
        return False
    if a is None:
        return False
    if b is None:
        return True
    return b > a

def compare_results(raw_results: Dict[str, Any], better_results: Dict[str, Any]) -> None:
    """Compare results between raw and better JSON files and print only worsened cases."""
    print("Analyzing worsened cases between raw and better prompts...")
    print("-" * 80)
    
    # Track statistics
    total_questions = 0
    worsened_questions = 0
    
    # Compare each question
    for question_number in sorted(raw_results.keys()):
        if question_number not in better_results:
            print(f"Question {question_number} not found in better results")
            continue
            
        raw = raw_results[question_number]
        better = better_results[question_number]
        total_questions += 1
        
        # Skip questions that were successful in both
        if raw['status'] == 'Success' and better['status'] == 'Success':
            continue
            
        # Compare accuracy
        raw_accuracy = raw.get('accuracy', 0)
        better_accuracy = better.get('accuracy', 0)
        
        # Compare cases passed
        raw_cases = raw.get('cases_passed')
        better_cases = better.get('cases_passed')
        raw_total = raw.get('total_cases')
        better_total = better.get('total_cases')
        
        # Determine if there was worsening
        worsened = False
        if raw['status'] == 'Success' and better['status'] != 'Success':
            worsened = True
        elif better_accuracy < raw_accuracy:
            worsened = True
        elif safe_compare(better_cases, raw_cases) and raw_total == better_total:
            worsened = True
            
        if worsened:
            worsened_questions += 1
            
            # Print details for worsened questions
            print(f"\nQuestion {question_number}: {raw['question_name']}")
            print(f"Raw Prompt:")
            print(f"  Status: {raw['status']}")
            print(f"  Accuracy: {raw_accuracy}%")
            if raw_cases is not None and raw_total is not None:
                print(f"  Cases Passed: {raw_cases}/{raw_total}")
            if raw['status'] == 'Success':
                print(f"  Runtime: {raw.get('runtime', {}).get('time_ms', 'N/A')}ms")
                print(f"  Memory: {raw.get('memory', {}).get('usage_mb', 'N/A')}MB")
                
            print(f"\nBetter Prompt:")
            print(f"  Status: {better['status']}")
            print(f"  Accuracy: {better_accuracy}%")
            if better_cases is not None and better_total is not None:
                print(f"  Cases Passed: {better_cases}/{better_total}")
            if better['status'] == 'Success':
                print(f"  Runtime: {better.get('runtime', {}).get('time_ms', 'N/A')}ms")
                print(f"  Memory: {better.get('memory', {}).get('usage_mb', 'N/A')}MB")
                
            print(f"\nWorsened: Yes")
            print("-" * 80)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total Questions Analyzed: {total_questions}")
    print(f"Questions Worsened: {worsened_questions}")
    print(f"Worsening Rate: {(worsened_questions/total_questions)*100:.2f}%")

def main():
    # Load the JSON files
    raw_results = load_json_file('leetcode_results_raw.json')
    better_results = load_json_file('leetcode_results_better.json')
    
    # Compare the results
    compare_results(raw_results, better_results)

if __name__ == "__main__":
    main() 