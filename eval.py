import os
import sys
import json
import argparse
import logging
from dotenv import load_dotenv
from deepseek import ProblemClassifier
from run_command import GPTService, ResultProcessor, FileHandler

# Load environment variables from .env file
load_dotenv()

def test_gpt_api():
    """Test GPT API connection with a simple prompt"""
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logging.error("OPENAI_API_KEY not found in environment variables")
            return False
            
        logging.info("Testing GPT API connection...")
        gpt_service = GPTService(api_key)
        
        # Simple test prompt
        test_prompt = "What is 2+2?"
        test_signature = "def add(a: int, b: int) -> int:"
        
        logging.info("\n=== Test Prompt ===")
        logging.info(f"Question: {test_prompt}")
        logging.info(f"Function Signature: {test_signature}")
        logging.info("==================\n")
        
        response = gpt_service.get_solution(test_prompt, test_signature)
        if response:
            logging.info("✅ GPT API connection successful")
            return True
        else:
            logging.error("❌ GPT API connection failed - no response received")
            return False
            
    except Exception as e:
        logging.error(f"❌ GPT API connection failed: {str(e)}")
        return False

class QuestionLoader:
    def __init__(self, questions_file="questions.json"):
        self.questions_file = questions_file
        self.questions = self._load_questions()

    def _load_questions(self):
        """Load questions from JSON file"""
        try:
            with open(self.questions_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading questions from {self.questions_file}: {e}")
            return {}

    def get_question(self, question_number):
        """Get question details from loaded questions"""
        return self.questions.get(str(question_number))

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
                with open(os.path.join(self.templates_dir, filename), 'r', encoding='utf-8') as f:
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
        self.question_loader = QuestionLoader()
        self.gpt_service = GPTService(os.getenv('OPENAI_API_KEY'))

    def solve_question(self, question_number):
        """Solve a single LeetCode question with enhanced template-based prompting"""
        print(f"Processing question {question_number}...")
        
        # Get question content from JSON
        question_data = self.question_loader.get_question(question_number)
        if not question_data:
            print(f"Question {question_number} not found in questions.json")
            return False

        # Build question content
        question = f"{question_data['question_title']}\n\n{question_data['question_description']}"
        function_signature = question_data['function_signature']

        # Classify the problem
        print("Classifying problem...")
        categories = self.classifier.classify(question)
        print(f"Predicted categories: {categories[0]}, {categories[1]}")

        # Get relevant templates
        template_content = self.template_handler.get_template_content(categories)
        
        # Build enhanced prompt
        prompt = f"""Consider solving the problem using one or both of the following approaches: {", ".join(categories)}

{template_content}

Question:
{question}

Function Signature:
{function_signature}

Provide ONLY the solution code."""
        
        # Print the prompt before sending
        print("\n=== Generated Prompt ===")
        print(prompt)
        print("======================\n")
        
        # Get solution from GPT
        print("Getting solution from GPT...")
        solution = self.gpt_service.get_solution(
            question,
            function_signature
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
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    # Test GPT API first
    if not test_gpt_api():
        logging.error("Cannot proceed without working GPT API connection")
        return

    solver = EnhancedLeetCodeSolver()
    question_loader = QuestionLoader()
    
    # Get all question numbers from the loaded questions
    question_numbers = sorted([int(num) for num in question_loader.questions.keys()])
    
    if not question_numbers:
        logging.error("No questions found in questions.json")
        return
        
    logging.info(f"Found {len(question_numbers)} questions to process")
    
    for question_number in question_numbers:
        logging.info(f"====== Solving Question {question_number} ======")
        solver.solve_question(question_number)
        logging.info("")  # Empty line for separation

if __name__ == "__main__":
    main()
