import subprocess
import re

# Method 1: Using subprocess.run (recommended for Python 3.5+)
def run_command_method1(command):
    # Shell=True allows shell commands, but be careful with user input for security
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Extract cases passed using regex
    cases_passed = None
    if result.stdout:
        match = re.search(r'Cases passed:\s*(\d+)', result.stdout)
        if match:
            cases_passed = int(match.group(1))
    
    return {
        'stdout': result.stdout,
        'stderr': result.stderr,
        'return_code': result.returncode,
        'cases_passed': cases_passed
    }

# Method 2: Using subprocess.Popen
def run_command_method2(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    print("Method 2 Output:")
    print("stdout:", stdout)
    print("stderr:", stderr)
    print("return code:", process.returncode)
    print("\n")

# Example usage
if __name__ == "__main__":
    # Example command - list files
    command = "leetcode exec 1"
    print("Running command:", command)
    result = run_command_method1(command)
    
    print("\nExtracted information:")
    print(f"Cases passed: {result['cases_passed']}")
    # print(f"Full stdout:\n{result['stdout']}")
    # print(f"Full stderr:\n{result['stderr']}")
    # print(f"Return code: {result['return_code']}")
    #run_command_method2(command)

    # Example command that might produce an error
    # command = "ls nonexistent_directory"
    # print("Running command:", command)
    # run_command_method1(command)
    # run_command_method2(command) 