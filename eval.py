import os
import sys
import json
import argparse
import logging
from dotenv import load_dotenv
from run_command import GPTService, ResultProcessor, FileHandler, LeetCodeAPI

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

class ClassificationLoader:
    def __init__(self, classification_file="classification_results.json"):
        self.classification_file = classification_file
        self.classifications = self._load_classifications()
        # List of valid categories (from LeetCode's standard categories)
        self.valid_categories = {
            "Dynamic Programming", "Math", "Tree", "Depth-first Search", "Greedy",
            "Hash Table", "Binary Search", "Breadth-first Search", "Sort",
            "Two Pointers", "Backtracking", "Stack", "Graph", "Bit Manipulation",
            "Heap", "Linked List", "Recursion", "Union Find", "Sliding Window",
            "Trie", "Divide and Conquer", "Segment Tree", "Ordered Map", "Queue",
            "String", "Array", "Binary Tree", "Matrix", "Design"
        }

    def _load_classifications(self):
        """Load classifications from JSON file"""
        try:
            with open(self.classification_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading classifications from {self.classification_file}: {e}")
            return {}

    def _is_valid_category(self, category):
        """Check if a category is valid"""
        if not category or not isinstance(category, str):
            return False
        # Check if it's a standard LeetCode category
        return category in self.valid_categories

    def get_categories(self, question_number):
        """Get categories for a question, filtering out invalid ones"""
        raw_categories = self.classifications.get(str(question_number), {}).get("categories", [])
        
        # Filter out invalid categories
        valid_categories = [cat for cat in raw_categories if self._is_valid_category(cat)]
        
        # Return valid categories if any exist
        if valid_categories:
            # If we only have one valid category, use it
            if len(valid_categories) == 1:
                return [valid_categories[0]]
            # If we have two or more valid categories, use the first two
            return valid_categories[:2]
        
        # If no valid categories found, return empty list
        return []

class TemplateHandler:
    def __init__(self, templates_file="alltp.json"):
        self.templates_file = templates_file
        self.templates = self._load_templates()

    def _load_templates(self):
        """Load all templates from the JSON file"""
        try:
            with open(self.templates_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('prompts', {})
        except Exception as e:
            logging.error(f"Error loading templates from {self.templates_file}: {e}")
            return {}

    def _normalize_category(self, category):
        """Convert category name to template key format"""
        # Convert to lowercase and replace spaces with underscores
        return category.lower().replace(" ", "_").replace("-", "_")

    def get_template_content(self, categories):
        """Get template content for given categories"""
        template_content = []
        for category in categories:
            # Normalize the category name to match template keys
            template_key = self._normalize_category(category)
            if template_key in self.templates:
                template = self.templates[template_key]
                # Format the template content
                template_content.append(f"## {category}:\n" + "\n".join(template))
        return "\n\n".join(template_content)

class EnhancedLeetCodeSolver:
    def __init__(self, prompt_type="raw"):
        self.gpt_service = GPTService(os.getenv('OPENAI_API_KEY'))
        self.prompt_type = prompt_type
        self.result_processor = ResultProcessor()
        self.template_handler = TemplateHandler()
        self.question_loader = QuestionLoader()
        self.classification_loader = ClassificationLoader()
        self.main_prompt, self.prompt_ending = self._load_prompts()

    def _load_prompts(self):
        """Load the main prompt and prompt ending from alltp.json"""
        try:
            with open("alltp.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                main_prompt = "\n".join(data.get('main_prompt', []))
                prompt_ending = "\n".join(data.get('prompt_ending', []))
                return main_prompt, prompt_ending
        except Exception as e:
            logging.error(f"Error loading prompts from alltp.json: {e}")
            return "", ""

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

        # Get pre-classified categories
        categories = self.classification_loader.get_categories(question_number)
        if categories:
            print(f"Using pre-classified categories: {', '.join(categories)}")
        else:
            print("No valid categories detected for this question")

        # Get relevant templates
        template_content = self.template_handler.get_template_content(categories)
        
        # Build enhanced prompt
        prompt_parts = [self.main_prompt]
        
        # Only add category line if we have valid categories
        if categories:
            prompt_parts.append(f"Consider solving the problem using one or both of the following approaches: {', '.join(categories)}")
        
        if template_content:
            prompt_parts.append(template_content)
            
        prompt_parts.extend([
            f"Question:\n{question}",
            f"Function Signature:\n{function_signature}",
            self.prompt_ending,
            "Provide ONLY the solution code."
        ])
        
        prompt = "\n\n".join(prompt_parts)
        
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
        
        # Test solution using leetcode-cli
        print("Testing solution...")
        test_output = LeetCodeAPI.test_solution(question_number)
        test_results = ResultProcessor.process_test_results(test_output['stdout'], test_output['stderr'])
        
        # Save results
        results_file = ResultProcessor.save_test_results(
            question_number,
            test_results,
            "better"
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

    def print_prompt(self, question_number):
        """Print the prompt for a given question number without solving it"""
        print(f"Generating prompt for question {question_number}...")
        
        # Get question content from JSON
        question_data = self.question_loader.get_question(question_number)
        if not question_data:
            print(f"Question {question_number} not found in questions.json")
            return False

        # Build question content
        question = f"{question_data['question_title']}\n\n{question_data['question_description']}"
        function_signature = question_data['function_signature']

        # Get pre-classified categories
        categories = self.classification_loader.get_categories(question_number)
        if categories:
            print(f"Using pre-classified categories: {', '.join(categories)}")
        else:
            print("No valid categories detected for this question")

        # Get relevant templates
        template_content = self.template_handler.get_template_content(categories)
        
        # Build and print enhanced prompt
        prompt_parts = [self.main_prompt]
        
        # Only add category line if we have valid categories
        if categories:
            prompt_parts.append(f"Consider solving the problem using one or both of the following approaches: {', '.join(categories)}")
        
        if template_content:
            prompt_parts.append(template_content)
            
        prompt_parts.extend([
            f"Question:\n{question}",
            f"Function Signature:\n{function_signature}",
            self.prompt_ending,
            "Provide ONLY the solution code."
        ])
        
        prompt = "\n\n".join(prompt_parts)
        
        print("\n=== Generated Prompt ===")
        print(prompt)
        print("======================\n")
        
        return True

    def eval_question(self, question_number: int) -> bool:
        """Evaluate a single question number.
        
        Args:
            question_number: The question number to evaluate
            
        Returns:
            bool: True if evaluation was successful, False otherwise
        """
        print(f"Evaluating question {question_number}...")
        
        # Get question content from JSON
        question_data = self.question_loader.get_question(question_number)
        if not question_data:
            print(f"Question {question_number} not found in questions.json")
            return False

        # Build question content
        question = f"{question_data['question_title']}\n\n{question_data['question_description']}"
        function_signature = question_data['function_signature']

        # Get pre-classified categories
        categories = self.classification_loader.get_categories(question_number)
        if categories:
            print(f"Using pre-classified categories: {', '.join(categories)}")
        else:
            print("No valid categories detected for this question")

        # Get relevant templates
        template_content = self.template_handler.get_template_content(categories)
        
        # Build enhanced prompt
        prompt_parts = [self.main_prompt]
        
        # Only add category line if we have valid categories
        if categories:
            prompt_parts.append(f"Consider solving the problem using one or both of the following approaches: {', '.join(categories)}")
        
        if template_content:
            prompt_parts.append(template_content)
            
        prompt_parts.extend([
            f"Question:\n{question}",
            f"Function Signature:\n{function_signature}",
            self.prompt_ending,
            "Provide ONLY the solution code."
        ])
        
        prompt = "\n\n".join(prompt_parts)
        
        # Get solution from GPT
        print("Getting solution from GPT...")
        solution = self.gpt_service.get_solution(prompt, function_signature)
        
        if not solution:
            print("Failed to get solution from GPT")
            return False

        # Save solution
        FileHandler.save_solution(question_number, solution)
        
        # Test solution using leetcode-cli
        print("Testing solution...")
        test_output = LeetCodeAPI.test_solution(question_number)
        test_results = ResultProcessor.process_test_results(test_output['stdout'], test_output['stderr'])
        
        # Save results
        results_file = ResultProcessor.save_test_results(
            question_number,
            test_results,
            "better"
        )
        
        # Display results
        self._display_results(test_results, test_output['stderr'], results_file)
        
        return True

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
    # solver = EnhancedLeetCodeSolver()
    # solver.print_prompt(3196)
    # solver.eval_question(3196)
