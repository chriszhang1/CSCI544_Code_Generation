import json
import logging
from typing import Dict, Any

def load_results(filename: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading results from {filename}: {e}")
        return {}

def compute_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute statistics from results"""
    total_questions = len(results)
    total_accuracy = 0
    valid_accuracy_count = 0
    
    # Count successes and failures
    success_count = 0
    failure_count = 0
    
    # Track runtime and memory stats for successful solutions
    runtime_stats = []
    memory_stats = []
    
    for result in results.values():
        # Count successes and failures
        if result['status'] == 'Success':
            success_count += 1
            if result.get('runtime'):
                runtime_stats.append(result['runtime']['time_ms'])
            if result.get('memory'):
                memory_stats.append(result['memory']['usage_mb'])
        else:
            failure_count += 1
            
        # Add to accuracy calculation
        if result.get('accuracy') is not None:
            total_accuracy += result['accuracy']
            valid_accuracy_count += 1
    
    # Calculate statistics
    average_accuracy = round(total_accuracy / valid_accuracy_count, 2) if valid_accuracy_count > 0 else 0
    success_rate = round((success_count / total_questions) * 100, 2) if total_questions > 0 else 0
    
    # Calculate runtime and memory stats if available
    avg_runtime = round(sum(runtime_stats) / len(runtime_stats), 2) if runtime_stats else None
    avg_memory = round(sum(memory_stats) / len(memory_stats), 2) if memory_stats else None
    
    return {
        "total_questions": total_questions,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": success_rate,
        "average_accuracy": average_accuracy,
        "average_runtime_ms": avg_runtime,
        "average_memory_mb": avg_memory
    }

def save_statistics(stats: Dict[str, Any], filename: str = "statistics.json") -> None:
    """Save statistics to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving statistics to {filename}: {e}")

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )
    
    # Load results
    results = load_results("leetcode_results_raw.json")
    if not results:
        logging.error("No results found")
        return
    
    # Compute statistics
    stats = compute_statistics(results)
    
    # Print statistics
    print("\nLeetCode Results Statistics:")
    print(f"Total Questions: {stats['total_questions']}")
    print(f"Successful Solutions: {stats['success_count']}")
    print(f"Failed Solutions: {stats['failure_count']}")
    print(f"Success Rate: {stats['success_rate']}%")
    print(f"Average Accuracy: {stats['average_accuracy']}%")
    if stats['average_runtime_ms']:
        print(f"Average Runtime: {stats['average_runtime_ms']} ms")
    if stats['average_memory_mb']:
        print(f"Average Memory: {stats['average_memory_mb']} MB")
    
    # Save statistics
    save_statistics(stats)

if __name__ == "__main__":
    main() 