import subprocess
import re
import os
import glob
import argparse
from dotenv import load_dotenv
from openai import OpenAI
import sys
import json
import datetime
import logging

# Import the TagPredictor from separate module
from tag_predictor import TagPredictor

class LeetCodeAPI:
    @staticmethod
    def get_question(question_number):
        """Get leetcode question content"""
        command = f"leetcode pick {question_number}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(f"Error getting question: {result.stderr}")
            return None
        return result.stdout

    @staticmethod
    def get_template_file_path(question_number):
        """Get path to template file"""
        template_path = os.path.expanduser(f"~/.leetcode/code/{question_number}.*")
        template_files = glob.glob(template_path)

        if not template_files:
            logging.warning(f"No template file found for question {question_number}")
            return None
        return template_files[0]

    @staticmethod
    def get_function_signature(question_number):
        """Get function signature from template"""
        # remove existing template file
        template_file = LeetCodeAPI.get_template_file_path(question_number)
        if template_file and os.path.exists(template_file):
            try:
                os.remove(template_file)
                logging.info(f"Removed existing solution file: {template_file}")
            except Exception as e:
                logging.warning(f"Failed to remove existing file {template_file}: {e}")
        
        # get new template
        command = f"leetcode edit {question_number}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(f"Error getting function signature: {result.stderr}")
            return None

        template_file = LeetCodeAPI.get_template_file_path(question_number)
        if not template_file:
            return None

        with open(template_file, 'r') as f:
            return f.read()

    @staticmethod
    def test_solution(question_number):
        """Test the solution using leetcode exec"""
        command = f"leetcode exec {question_number}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Filter out INFO messages from stderr
        real_errors = [line for line in result.stderr.splitlines() 
                      if line.strip() and not line.strip().startswith('[INFO')]

        return {
            'stdout': result.stdout,
            'stderr': '\n'.join(real_errors) if real_errors else '',
            'return_code': result.returncode,
        }

    @staticmethod
    def get_question_name(question_number):
        """Extract question name from template file name"""
        template_file = LeetCodeAPI.get_template_file_path(question_number)
        if template_file:
            match = re.search(rf'{question_number}\.(.+)\.py', os.path.basename(template_file))
            if match:
                return match.group(1)
        return None

class PromptBuilder:
    """Builds custom prompts using predicted tags"""
    
    def __init__(self, all_tags_file="alltp.json"):
        self.prompt_data = self._load_prompt_data(all_tags_file)
    
    def _load_prompt_data(self, file_path):
        """Load prompt data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data
        except Exception as e:
            logging.error(f"Error loading prompt data from {file_path}: {e}")
            return {"main_prompt": [], "prompts": {}}
    
    def build_system_prompt(self):
        """Build system prompt from main prompt"""
        if not self.prompt_data or not self.prompt_data.get("main_prompt"):
            return "You are a Python coding assistant for LeetCode problems."
        
        return "\n".join(self.prompt_data.get("main_prompt", []))
    
    def build_tag_specific_content(self, tags):
        """Build content for specific tags"""
        if not tags or not self.prompt_data or not self.prompt_data.get("prompts"):
            return ""
        
        tag_contents = []
        
        for tag in tags:
            if tag in self.prompt_data["prompts"]:
                tag_content = "\n".join(self.prompt_data["prompts"][tag])
                tag_contents.append(f"## {tag.replace('_', ' ').title()} Protocol:\n{tag_content}")
        
        if not tag_contents:
            return ""
            
        return "\n\n".join([
            "# Tag-Specific Protocols",
            "Apply these specific protocols based on the problem characteristics:",
            "\n\n".join(tag_contents)
        ])

class GPTService:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.raw_prompt = self._load_raw_prompt()

        self.prompt_builder = PromptBuilder()
    
    def _load_raw_prompt(self):
        """Load raw prompt from file"""
        try:
            with open("raw_prompt.txt", "r") as f:
                return f.read().strip()

        except FileNotFoundError:
    # ===================== default prompt ================================================
            return """You are a Python coding assistant for LeetCode problems. 
Your task is to provide ONLY the solution code, exactly matching the required format.
- Do NOT include markdown formatting or code blocks
- Do NOT include any explanations or comments
- Include ONLY the necessary imports and the Solution class
- Match the function signature EXACTLY as provided
- Ensure proper indentation
- The code should be ready to run as-is"""
    # ================================

    def get_solution(self, question, function_signature, prompt_type="raw", tags=None):
        """Get solution from GPT API using specified prompt"""
        try:
            # Use different approaches based on prompt type
            if prompt_type == "raw":
                system_content = self.raw_prompt
                user_content = self._build_base_user_prompt(function_signature, question)
            elif prompt_type == "tags":
                # Use tag-based prompting
                system_content = self.prompt_builder.build_system_prompt()
                tag_content = self.prompt_builder.build_tag_specific_content(tags)
                user_content = self._build_tag_user_prompt(function_signature, question, tag_content)
            else:
                # Default to raw prompt
                system_content = self.raw_prompt
                user_content = self._build_base_user_prompt(function_signature, question)
            
            # Call the API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7
            )
            solution = response.choices[0].message.content.strip()
            return self._clean_code_output(solution)
        except Exception as e:
            logging.error(f"Error calling GPT API: {e}")
            return None
    
    def _build_base_user_prompt(self, function_signature, question):
        """Build basic user prompt"""
        return f"""
Generate a Python solution for this LeetCode question.
Use this exact function signature and format:

{function_signature}

Question:
{question}

Provide ONLY the solution code."""
    
    def _build_tag_user_prompt(self, function_signature, question, tag_content):
        """Build tag-enhanced user prompt"""
        base_prompt = self._build_base_user_prompt(function_signature, question)
        
        if tag_content:
            return f"""
{tag_content}

{base_prompt}
"""
        else:
            return base_prompt
    
    @staticmethod
    def _clean_code_output(code):
        """Clean GPT code output for proper format"""
        code = re.sub(r'```python\n', '', code)
        code = re.sub(r'```\n?', '', code)
        code = code.strip()
        
        if 'List' in code and 'from typing import List' not in code:
            code = 'from typing import List\n' + code
        
        return code

class ResultProcessor:
    @staticmethod
    def process_test_results(stdout, stderr):
        """Process leetcode test results"""
        result = {
            'status': 'Unknown',
            'accuracy': 0,
            'runtime': None,
            'memory': None,
            'cases_passed': None,
            'total_cases': None
        }
        
        if 'Success' in stdout:
            result['status'] = 'Success'
            result['accuracy'] = 100
            
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
    
    @staticmethod
    def save_test_results(question_number, results, predicted_tags=None):
        """Save test results to JSON file"""
        results_file = 'leetcode_results.json'
        
        all_results = {}
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        all_results = json.loads(content)
            except json.JSONDecodeError:
                logging.warning("Could not read existing results file. Creating new one.")
        
        question_name = LeetCodeAPI.get_question_name(question_number)
        result_entry = {
            'question_number': question_number,
            'question_name': question_name,
            'timestamp': datetime.datetime.now().isoformat(),
            **results
        }
        
        # Add predicted tags if available
        if predicted_tags:
            result_entry['predicted_tags'] = list(predicted_tags)
        
        all_results[str(question_number)] = result_entry
        
        try:
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
        except Exception as e:
            logging.error(f"Could not save results to file: {e}")
            return None
        
        return results_file

class FileHandler:
    @staticmethod
    def save_solution(question_number, solution):
        """Save solution to template file"""
        template_file = LeetCodeAPI.get_template_file_path(question_number)
        if template_file:
            with open(template_file, 'w') as f:
                f.write(solution)
            logging.info(f"Updated template file: {template_file}")
            return True
        logging.warning("Could not update template file")
        return False

class LeetCodeSolver:
    def __init__(self, prompt_type="raw", prediction_method="gpt"):
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logging.error("OPENAI_API_KEY not found in environment variables")
            sys.exit(1)
        
        self.gpt_service = GPTService(api_key)
        self.prompt_type = prompt_type
        self.predicted_tags = None
        self.prediction_method = prediction_method
        self.tag_predictor = TagPredictor(api_key=api_key, prediction_method=prediction_method)

    def solve_question(self, question_number):
        """Solve a single LeetCode question"""
        logging.info(f"Processing question {question_number}...")
        
        # Get question and function signature
        question = LeetCodeAPI.get_question(question_number)
        if not question:
            logging.error("Failed to get question")
            return False
        
        function_signature = LeetCodeAPI.get_function_signature(question_number)
        if not function_signature:
            logging.error("Failed to get function signature")
            return False
        
        # Predict tags if using tag-based prompt
        if self.prompt_type == "tags":
            logging.info(f"Predicting problem tags using {self.prediction_method}...")
            self.predicted_tags = self.tag_predictor.predict_tags(question)
            if self.predicted_tags:
                logging.info(f"Predicted tags: {', '.join(self.predicted_tags)}")
        
        # Get solution from GPT
        logging.info(f"Getting solution using {self.prompt_type} prompt...")
        solution = self.gpt_service.get_solution(
            question, 
            function_signature, 
            self.prompt_type,
            self.predicted_tags
        )
        if not solution:
            logging.error("Failed to get solution from GPT")
            return False
        
        # Save solution
        FileHandler.save_solution(question_number, solution)
        
        # Test solution
        logging.info("Testing solution...")
        test_output = LeetCodeAPI.test_solution(question_number)
        test_results = ResultProcessor.process_test_results(test_output['stdout'], test_output['stderr'])
        
        # Save and display results
        results_file = ResultProcessor.save_test_results(
            question_number, 
            test_results,
            self.predicted_tags
        )
        self._display_results(test_results, test_output['stderr'], results_file)
        
        return True
    
    def _display_results(self, test_results, stderr, results_file):
        """Display test results in a clean format"""
        if test_results['status'] == 'Success':
            logging.info(f"✅ Success! (100% accuracy)")
            if test_results['runtime']:
                logging.info(f"Runtime: {test_results['runtime']['time_ms']}ms "
                      f"(faster than {test_results['runtime']['percentile']}%)")
            if test_results['memory']:
                logging.info(f"Memory: {test_results['memory']['usage_mb']}MB "
                      f"(less than {test_results['memory']['percentile']}%)")
        else:
            logging.info(f"❌ Failed")
            logging.info(f"Cases passed: {test_results['cases_passed']}/{test_results['total_cases']}")
            logging.info(f"Accuracy: {test_results['accuracy']}%")
        
        logging.info(f"Results saved to {results_file}")
        
        if stderr:
            logging.error("Errors encountered:")
            logging.error(stderr)
        
        # Display predicted tags if available
        if self.predicted_tags:
            logging.info(f"Tags: {', '.join(self.predicted_tags)}")

def main():
    # Configure argument parser
    parser = argparse.ArgumentParser(description="Solve LeetCode problems using GPT")
    parser.add_argument("start", type=int, help="Starting problem number")
    parser.add_argument("end", type=int, nargs="?", help="Ending problem number (optional)")
    parser.add_argument("--prompt", "-p", type=str, default="raw", choices=["raw", "tags"], 
                        help="Type of prompt to use (raw or tags)")
    parser.add_argument("--prediction", type=str, default="gpt", choices=["gpt", "bert"],
                        help="Tag prediction method to use")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Set logging level")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s"
    )

    start_question = args.start
    end_question = args.end if args.end else start_question

    solver = LeetCodeSolver(prompt_type=args.prompt, prediction_method=args.prediction)

    for i in range(start_question, end_question + 1):
        logging.info(f"====== Solving Question {i} ======")
        solver.solve_question(i)
        logging.info("")  # Empty line for separation

if __name__ == "__main__":
    main()
