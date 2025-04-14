import subprocess
import re
import os
import glob
from dotenv import load_dotenv
from openai import OpenAI
import sys
import json
import datetime

# Load environment variables from .env file first
load_dotenv()

# Check if API key is loaded
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("Error: OPENAI_API_KEY not found in environment variables")
    print("Make sure you have created a .env file with your API key")
    sys.exit(1)

# Then create OpenAI client with the loaded API key
client = OpenAI(api_key=api_key)

def get_leetcode_question(question_number):
    """Get the leetcode question content by running leetcode pick command"""
    command = f"leetcode pick {question_number}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        return None
        
    return result.stdout

def get_template_file_path(question_number):
    """Get the path to the template file"""
    template_path = os.path.expanduser(f"~/.leetcode/code/{question_number}.*")
    template_files = glob.glob(template_path)
    
    if not template_files:
        print(f"No template file found for question {question_number}")
        return None
        
    return template_files[0]

def get_function_signature(question_number):
    """Get the function signature by running leetcode edit command and reading the template"""
    # Run leetcode edit command
    command = f"leetcode edit {question_number}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running edit command: {result.stderr}")
        return None
    
    # Get template file path
    template_file = get_template_file_path(question_number)
    if not template_file:
        return None
    
    # Read the template file
    with open(template_file, 'r') as f:
        template_content = f.read()
    
    return template_content

def clean_code_output(code):
    """Clean the code output from GPT to ensure proper format"""
    # Remove markdown code blocks if present
    code = re.sub(r'```python\n', '', code)
    code = re.sub(r'```\n?', '', code)
    
    # Remove any leading/trailing whitespace
    code = code.strip()
    
    # Ensure proper imports
    if 'List' in code and 'from typing import List' not in code:
        code = 'from typing import List\n' + code
    
    return code

def get_solution_from_gpt(question, function_signature):
    """Get solution from GPT API"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are a Python coding assistant for LeetCode problems. 
Your task is to provide ONLY the solution code, exactly matching the required format.
- Do NOT include markdown formatting or code blocks
- Do NOT include any explanations or comments
- Include ONLY the necessary imports and the Solution class
- Match the function signature EXACTLY as provided
- Ensure proper indentation
- The code should be ready to run as-is"""},
                {"role": "user", "content": f"""
Generate a Python solution for this LeetCode question.
Use this exact function signature and format:

{function_signature}

Question:
{question}

Provide ONLY the solution code."""}
            ],
            temperature=0.7  # Slightly reduce randomness for more consistent output
        )
        solution = response.choices[0].message.content.strip()
        return clean_code_output(solution)
    except Exception as e:
        print(f"Error calling GPT API: {e}")
        return None

def save_solution(question_number, solution):
    """Save the solution to our solutions directory and update the leetcode template""" 
    # Update the leetcode template file
    template_file = get_template_file_path(question_number)
    if template_file:
        with open(template_file, 'w') as f:
            f.write(solution)
        print(f"Updated template file: {template_file}")
    else:
        print("Warning: Could not update template file")

def process_test_results(stdout, stderr):
    """Process the test results from leetcode exec output"""
    result = {
        'status': 'Unknown',
        'accuracy': 0,
        'runtime': None,
        'memory': None,
        'cases_passed': None,
        'total_cases': None
    }
    
    # Check for success case
    if 'Success' in stdout:
        result['status'] = 'Success'
        result['accuracy'] = 100
        
        # Extract runtime and memory info
        runtime_match = re.search(r'Runtime: (\d+) ms, faster than (\d+)%', stdout)
        if runtime_match:
            result['runtime'] = {
                'time_ms': int(runtime_match.group(1)),
                'percentile': int(runtime_match.group(2))
            }
            
        memory_match = re.search(r'Memory Usage: ([\d.]+) MB, less than (\d+)%', stdout)
        if memory_match:
            result['memory'] = {
                'usage_mb': float(memory_match.group(1)),
                'percentile': int(memory_match.group(2))
            }
    
    # Check for failure case
    else:
        result['status'] = 'Failed'
        cases_match = re.search(r'Cases passed:\s*(\d+)\nTotal cases:\s*(\d+)', stdout)
        if cases_match:
            cases_passed = int(cases_match.group(1))
            total_cases = int(cases_match.group(2))
            result['cases_passed'] = cases_passed
            result['total_cases'] = total_cases
            result['accuracy'] = round((cases_passed / total_cases) * 100, 2)
    
    return result

def get_question_name(question_number):
    """Extract question name from the template file name"""
    template_file = get_template_file_path(question_number)
    if template_file:
        # Extract name from pattern: number.name.py
        match = re.search(rf'{question_number}\.(.+)\.py', os.path.basename(template_file))
        if match:
            return match.group(1)
    return None

def save_test_results(question_number, results):
    """Save test results to a JSON file"""
    results_file = 'leetcode_results.json'
    
    # Load existing results if file exists
    all_results = {}
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    
    # Add new result
    question_name = get_question_name(question_number)
    all_results[question_number] = {
        'question_number': question_number,
        'question_name': question_name,
        'timestamp': datetime.datetime.now().isoformat(),
        **results
    }
    
    # Save updated results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return results_file

def run_command_method1(command):
    # Shell=True allows shell commands, but be careful with user input for security
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Process the test results
    test_results = process_test_results(result.stdout, result.stderr)
    
    # Filter out INFO messages from stderr
    real_errors = []
    if result.stderr:
        for line in result.stderr.splitlines():
            if line.strip() and not line.strip().startswith('[INFO'):
                real_errors.append(line)
    
    return {
        'stdout': result.stdout,
        'stderr': '\n'.join(real_errors) if real_errors else '',
        'return_code': result.returncode,
        'test_results': test_results
    }

# Example usage
if __name__ == "__main__":
    # Get question number from command line argument or use default
    question_number = sys.argv[1] if len(sys.argv) > 1 else "1"
    
    # Step 1: Get the leetcode question
    print(f"Getting question {question_number}...")
    question = get_leetcode_question(question_number)
    if question is None:
        print("Failed to get leetcode question")
        exit(1)
    
    # Step 2: Get the function signature
    print("Getting function signature...")
    function_signature = get_function_signature(question_number)
    if function_signature is None:
        print("Failed to get function signature")
        exit(1)
    
    print("Retrieved question and function signature")
    
    # Step 3: Get solution from GPT
    print("Getting solution from GPT...")
    solution = get_solution_from_gpt(question, function_signature)
    if solution is None:
        print("Failed to get solution from GPT")
        exit(1)
    
    # Step 4: Save the solution and update template
    save_solution(question_number, solution)
    
    # Step 5: Test the solution
    print("\nTesting solution:")
    command = f"leetcode exec {question_number}"
    result = run_command_method1(command)
    
    # Save and display results
    results_file = save_test_results(question_number, result['test_results'])
    
    # Print test results
    test_results = result['test_results']
    if test_results['status'] == 'Success':
        print(f"\nSuccess! (100% accuracy)")
        if test_results['runtime']:
            print(f"Runtime: {test_results['runtime']['time_ms']}ms "
                  f"(faster than {test_results['runtime']['percentile']}%)")
        if test_results['memory']:
            print(f"Memory: {test_results['memory']['usage_mb']}MB "
                  f"(less than {test_results['memory']['percentile']}%)")
    else:
        print(f"\nFailed")
        print(f"Cases passed: {test_results['cases_passed']}/{test_results['total_cases']}")
        print(f"Accuracy: {test_results['accuracy']}%")
    
    print(f"\nResults saved to {results_file}")
    
    if result['stderr']:  # Now only contains real errors
        print("\nErrors encountered:")
        print(result['stderr'])
