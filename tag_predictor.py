import json
import logging
from openai import OpenAI

class TagPredictor:
    """
    Predicts algorithm tags for LeetCode problems.
    Supports different prediction methods including GPT API and local BERT models.
    """
    
    def __init__(self, api_key=None, all_tags_file="alltp.json", prediction_method="gpt"):
        """
        Initialize the tag predictor
        
        Args:
            api_key: OpenAI API key for GPT-based prediction
            all_tags_file: JSON file containing available tags
            prediction_method: Method to use for prediction ("gpt" or "bert")
        """
        self.api_key = api_key
        self.prediction_method = prediction_method
        
        if prediction_method == "gpt" and not api_key:
            logging.warning("No API key provided for GPT-based tag prediction")
        
        if prediction_method == "gpt":
            self.client = OpenAI(api_key=api_key) if api_key else None
        elif prediction_method == "bert":
            # Placeholder for future BERT model initialization
            self.bert_model = None
            logging.info("BERT-based prediction method is not yet implemented")
        
        self.all_tags = self._load_tags(all_tags_file)
    
    def _load_tags(self, file_path):
        """Load available tags from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                available_tags = list(data["prompts"].keys())
                logging.info(f"Loaded {len(available_tags)} tags from {file_path}")
                return available_tags
        except Exception as e:
            logging.error(f"Error loading tags from {file_path}: {e}")
            return []
    
    def predict_tags(self, question_text):
        """
        Predict tags for a LeetCode problem using the selected method
        
        Args:
            question_text: The problem description
            
        Returns:
            set: Set of predicted tags
        """
        if self.prediction_method == "gpt":
            return self._predict_tags_gpt(question_text)
        elif self.prediction_method == "bert":
            return self._predict_tags_bert(question_text)
        else:
            logging.error(f"Unknown prediction method: {self.prediction_method}")
            return set()
    
    def _predict_tags_gpt(self, question_text):
        """Predict tags using GPT API"""
        if not self.client:
            logging.error("OpenAI client not initialized")
            return set()
            
        if not self.all_tags:
            logging.warning("No tags available for prediction")
            return set()
        
        prompt = f"""
        Analyze this LeetCode problem and identify all applicable algorithm tags from the following list:
        {', '.join(self.all_tags)}
        
        Problem:
        {question_text}
        
        Return ONLY a Python set literal containing the relevant tags, like: 
        {{"tag1", "tag2"}}
        
        Your response must:
        1. Only include tags from the provided list
        2. Be a valid Python set literal that can be evaluated
        3. Contain no explanation or additional text
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a tag classifier for LeetCode problems."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            tags_text = response.choices[0].message.content.strip()
            
            try:
                # Clean up response for evaluation
                tags_text = tags_text.replace("```python", "").replace("```", "").strip()
                predicted_tags = eval(tags_text)
                
                # Validate tags
                valid_tags = {tag for tag in predicted_tags if tag in self.all_tags}
                
                if len(valid_tags) < len(predicted_tags):
                    logging.warning(f"Removed {len(predicted_tags) - len(valid_tags)} invalid tags")
                
                logging.info(f"Predicted tags: {valid_tags}")
                return valid_tags
                
            except Exception as e:
                logging.error(f"Error parsing predicted tags: {e}")
                logging.debug(f"Raw response: {tags_text}")
                return set()
                
        except Exception as e:
            logging.error(f"Error in tag prediction API call: {e}")
            return set()
    
    def _predict_tags_bert(self, question_text):
        """Predict tags using local BERT model (placeholder for future implementation)"""
        logging.warning("BERT prediction not yet implemented")
        return set()