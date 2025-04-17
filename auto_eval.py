import os
import sys
import argparse
import logging
from deepseek import ProblemClassifier
from run_command import LeetCodeAPI, GPTService, ResultProcessor, FileHandler

class TemplateHandler:
    def __init__(self, templates_dir="templates"):
        self.templates_dir = templates_dir
        self.templates = self._load_templates()

    def _load_templates(self):
        """Load all template files from the templates directory"""
        templates = {}
        for filename in os.listdir(self.templates_dir):
            if filename.endswith('.md'):
                category = filename[:-3]  # Remove .md extension
                with open(os.path.join(self.templates_dir, filename), 'r') as f:
                    templates[category] = f.read()
        return templates

    def get_template_content(self, categories):
        """Get template content for given categories"""
        template_content = []
        for category in categories:
            if category in self.templates:
                template_content.append(f"## {category} Template:\n{self.templates[category]}")
        return "\n\n".join(template_content)

class EnhancedLeetCodeSolver:
    def __init__(self):
        self.classifier = ProblemClassifier()
        self.template_handler = TemplateHandler()
        self.leetcode_api = LeetCodeAPI()
        self.gpt_service = GPTService(os.getenv('OPENAI_API_KEY'))

    def solve_question(self, question_number):
        """Solve a single LeetCode question with enhanced template-based prompting"""
        print(f"Processing question {question_number}...")
        
        # Get question content
        question = self.leetcode_api.get_question(question_number)
        if not question:
            print("Failed to get question")
            return False
        
        # Get function signature
        function_signature = self.leetcode_api.get_function_signature(question_number)
        if not function_signature:
            print("Failed to get function signature")
            return False

        # Classify the problem
        print("Classifying problem...")
        categories = self.classifier.classify(question)
        print(f"Predicted categories: {categories[0]}, {categories[1]}")

        # Get relevant templates
        template_content = self.template_handler.get_template_content(categories)
        
        # Build enhanced prompt
        system_prompt = """You are a Python coding assistant for LeetCode problems. 
Consider solving the problem using one or both of the following approaches: {categories}

{template_content}

Your task is to provide ONLY the solution code, exactly matching the required format.
- Do NOT include markdown formatting or code blocks
- Do NOT include any explanations or comments
- Include ONLY the necessary imports and the Solution class
- Match the function signature EXACTLY as provided
- Ensure proper indentation
- The code should be ready to run as-is"""

        system_prompt = system_prompt.format(
            categories=", ".join(categories),
            template_content=template_content
        )

        # Get solution from GPT
        print("Getting solution from GPT...")
        solution = self.gpt_service.get_solution(
            question,
            function_signature,
            system_prompt=system_prompt
        )
        
        if not solution:
            print("Failed to get solution from GPT")
            return False

        # Save solution
        FileHandler.save_solution(question_number, solution)
        
        # Test solution
        print("Testing solution...")
        test_output = self.leetcode_api.test_solution(question_number)
        test_results = ResultProcessor.process_test_results(test_output['stdout'], test_output['stderr'])
        
        # Save results
        results_file = ResultProcessor.save_test_results(
            question_number,
            test_results,
            categories,
            prompt_type="better"
        )
        
        # Display results
        self._display_results(test_results, test_output['stderr'], results_file)
        
        return True

    def _display_results(self, test_results, stderr, results_file):
        """Display test results"""
        if test_results['status'] == 'Success':
            print(f"✅ Success! (100% accuracy)")
            if test_results['runtime']:
                print(f"Runtime: {test_results['runtime']['time_ms']}ms "
                      f"(faster than {test_results['runtime']['percentile']}%)")
            if test_results['memory']:
                print(f"Memory: {test_results['memory']['usage_mb']}MB "
                      f"(less than {test_results['memory']['percentile']}%)")
        else:
            print(f"❌ Failed")
            print(f"Cases passed: {test_results['cases_passed']}/{test_results['total_cases']}")
            print(f"Accuracy: {test_results['accuracy']}%")
        
        print(f"Results saved to {results_file}")
        
        if stderr:
            print("Errors encountered:")
            print(stderr)

def main():
    # Configure argument parser
    parser = argparse.ArgumentParser(description="Solve LeetCode problems using GPT with enhanced template-based prompting")
    parser.add_argument("start", type=int, help="Starting problem number")
    parser.add_argument("end", type=int, nargs="?", help="Ending problem number (optional)")
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

    solver = EnhancedLeetCodeSolver()

    for i in range(start_question, end_question + 1):
        logging.info(f"====== Solving Question {i} ======")
        solver.solve_question(i)
        logging.info("")  # Empty line for separation

if __name__ == "__main__":
    main()
