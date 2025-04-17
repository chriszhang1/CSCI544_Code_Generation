import subprocess
import os
import json
import logging
import re
import glob
from typing import Optional, Dict, Any
import time

class LeetCodeFetcher:
    def __init__(self):
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _get_template_file_path(self, question_number: int) -> Optional[str]:
        """
        Get path to template file
        
        Args:
            question_number: The LeetCode question number
            
        Returns:
            Path to the template file, or None if not found
        """
        # First try to find the exact file pattern: number.title.py
        template_path = os.path.expanduser(f"~/.leetcode/code/{question_number}.*.py")
        template_files = glob.glob(template_path)
        
        if not template_files:
            return None
        return template_files[0]

    def _get_function_signature(self, question_number: int) -> Optional[str]:
        """
        Get the function signature from template file
        
        Args:
            question_number: The LeetCode question number
            
        Returns:
            The function signature as a string, or None if failed
        """
        try:
            # Remove existing template file if it exists
            template_file = self._get_template_file_path(question_number)
            if template_file and os.path.exists(template_file):
                try:
                    os.remove(template_file)
                    logging.info(f"Removed existing solution file: {template_file}")
                except Exception as e:
                    logging.warning(f"Failed to remove existing file {template_file}: {e}")
            
            # Get new template
            command = f"leetcode edit {question_number}"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                logging.error(f"Error getting function signature: {result.stderr}")
                return None
                
            # Read the template file
            template_file = self._get_template_file_path(question_number)
            if not template_file:
                # Try to find the template file again after a short delay
                time.sleep(0.5)  # Give the leetcode-cli some time to create the file
                template_file = self._get_template_file_path(question_number)
                if not template_file:
                    logging.error(f"Could not find template file for question {question_number}")
                    return None
                
            with open(template_file, 'r') as f:
                return f.read()
                
        except Exception as e:
            logging.error(f"Exception while getting function signature for question {question_number}: {str(e)}")
            return None

    def _parse_question_content(self, content: str) -> Optional[Dict[str, str]]:
        """
        Parse the question content into title and description
        
        Args:
            content: The raw question content
            
        Returns:
            Dictionary with title and description, or None if invalid
        """
        if not content:
            return None
            
        lines = content.strip().split('\n')
        if not lines:
            return None
            
        # Find the title line (first line with [number])
        title_line = None
        for line in lines:
            if re.match(r'^\[\d+\]', line):
                title_line = line
                break
                
        if not title_line:
            return None
            
        # Extract title and number
        title_match = re.match(r'^\[(\d+)\]\s*(.*)', title_line)
        if not title_match:
            return None
            
        question_number = title_match.group(1)
        question_title = title_match.group(2).strip()
        
        # Get description (everything after title line)
        description_lines = []
        found_title = False
        
        for line in lines:
            if found_title:
                description_lines.append(line)
            elif line == title_line:
                found_title = True
                
        description = '\n'.join(description_lines).strip()
        if not description:
            return None
            
        return {
            "question_number": question_number,
            "question_title": question_title,
            "question_description": description
        }

    def fetch_question(self, question_number: int) -> Optional[Dict[str, str]]:
        """
        Fetch a single LeetCode question using leetcode-cli
        
        Args:
            question_number: The LeetCode question number
            
        Returns:
            Dictionary containing question details, or None if failed
        """
        try:
            logging.info(f"Fetching question {question_number}...")
            command = f"leetcode pick {question_number}"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                logging.error(f"Error fetching question {question_number}: {result.stderr}")
                return None

            content = result.stdout.strip()
            question_details = self._parse_question_content(content)
            
            if not question_details:
                logging.warning(f"Question {question_number} has invalid or empty content")
                return None
                
            # Get function signature
            function_signature = self._get_function_signature(question_number)
            if function_signature:
                question_details["function_signature"] = function_signature
            else:
                logging.warning(f"Could not get function signature for question {question_number}")
                return None

            return question_details
        except Exception as e:
            logging.error(f"Exception while fetching question {question_number}: {str(e)}")
            return None

    def fetch_questions(self, start: int, end: int) -> Dict[int, Dict[str, str]]:
        """
        Fetch multiple LeetCode questions
        
        Args:
            start: Starting question number
            end: Ending question number
            
        Returns:
            Dictionary mapping question numbers to their details
        """
        questions = {}
        for i in range(start, end + 1):
            details = self.fetch_question(i)
            if details:
                questions[i] = details
                logging.info(f"Successfully fetched question {i}")
            else:
                logging.warning(f"Failed to fetch question {i}")
        return questions

    def save_questions(self, questions: Dict[int, Dict[str, str]], output_file: str = "questions.json") -> None:
        """
        Save fetched questions to a JSON file
        
        Args:
            questions: Dictionary of question numbers to details
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
