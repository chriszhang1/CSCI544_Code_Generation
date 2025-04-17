import subprocess
import os
import json
import logging
from typing import Optional, Dict, Any

class LeetCodeFetcher:
    def __init__(self):
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def fetch_question(self, question_number: int) -> Optional[str]:
        """
        Fetch a single LeetCode question using leetcode-cli
        
        Args:
            question_number: The LeetCode question number
            
        Returns:
            The question content as a string, or None if failed
        """
        try:
            logging.info(f"Fetching question {question_number}...")
            command = f"leetcode pick {question_number}"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                logging.error(f"Error fetching question {question_number}: {result.stderr}")
                return None

            return result.stdout
        except Exception as e:
            logging.error(f"Exception while fetching question {question_number}: {str(e)}")
            return None

    def fetch_questions(self, start: int, end: int) -> Dict[int, str]:
        """
        Fetch multiple LeetCode questions
        
        Args:
            start: Starting question number
            end: Ending question number
            
        Returns:
            Dictionary mapping question numbers to their content
        """
        questions = {}
        for i in range(start, end + 1):
            content = self.fetch_question(i)
            if content:
                questions[i] = content
                logging.info(f"Successfully fetched question {i}")
            else:
                logging.warning(f"Failed to fetch question {i}")
        return questions

    def save_questions(self, questions: Dict[int, str], output_file: str = "questions.json") -> None:
        """
        Save fetched questions to a JSON file
        
        Args:
            questions: Dictionary of question numbers to content
            output_file: Path to the output JSON file
        """
        try:
            # Load existing questions if file exists
            existing_questions = {}
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        existing_questions = json.load(f)
                except json.JSONDecodeError:
                    logging.warning(f"Existing {output_file} is corrupted, will be overwritten")
            
            # Update with new questions
            existing_questions.update(questions)
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_questions, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved questions to {output_file}")
        except Exception as e:
            logging.error(f"Error saving questions: {str(e)}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch LeetCode questions")
    parser.add_argument("start", type=int, help="Starting question number")
    parser.add_argument("end", type=int, nargs="?", help="Ending question number (optional)")
    parser.add_argument("--output", type=str, default="questions.json",
                        help="Output JSON file path")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Set logging level")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    fetcher = LeetCodeFetcher()
    
    # If end is not provided, only fetch the start question
    end = args.end if args.end else args.start
    
    # Fetch questions
    questions = fetcher.fetch_questions(args.start, end)
    
    # Save questions
    fetcher.save_questions(questions, args.output)

if __name__ == "__main__":
    main()
