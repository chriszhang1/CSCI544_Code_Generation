import os
import json
import logging
from deepseek import ProblemClassifier

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

def save_classification_results(results, output_file="classification_results.json"):
    """Save classification results to JSON file"""
    try:
        # Load existing results if file exists
        existing_results = {}
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Existing {output_file} is corrupted, will be overwritten")
        
        # Update with new results
        existing_results.update(results)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_results, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved classification results to {output_file}")
    except Exception as e:
        logging.error(f"Error saving classification results: {str(e)}")

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    # Initialize components
    classifier = ProblemClassifier()
    question_loader = QuestionLoader()
    
    # Get all question numbers from the loaded questions
    question_numbers = sorted([int(num) for num in question_loader.questions.keys()])
    
    if not question_numbers:
        logging.error("No questions found in questions.json")
        return
        
    logging.info(f"Found {len(question_numbers)} questions to classify")
    
    # Dictionary to store classification results
    classification_results = {}
    
    # Process each question
    for question_number in question_numbers:
        logging.info(f"Classifying question {question_number}...")
        
        # Get question content from JSON
        question_data = question_loader.get_question(question_number)
        if not question_data:
            logging.warning(f"Question {question_number} not found in questions.json")
            continue

        # Build question content
        question = f"{question_data['question_title']}\n\n{question_data['question_description']}"

        # Classify the problem
        categories = classifier.classify(question)
        logging.info(f"Question {question_number} categories: {categories[0]}, {categories[1]}")
        
        # Store results
        classification_results[str(question_number)] = {
            "categories": categories
        }
        
        # Save results after each question (in case of interruption)
        save_classification_results(classification_results)
        
        logging.info("")  # Empty line for separation

if __name__ == "__main__":
    main()
