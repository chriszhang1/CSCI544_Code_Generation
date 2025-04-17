import os
# Force disable TensorFlow before importing any other libraries
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Limit to first GPU
os.environ["USE_TF"] = "0"
os.environ["FORCE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging

# Import PyTorch first to ensure it initializes properly
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Explicitly set transformers to use PyTorch
transformers.utils.is_torch_available = lambda: True
transformers.utils.is_tf_available = lambda: False

class ProblemClassifier:
    def __init__(self, base_model="unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit", 
                 adapter_path="./leetcode-classifier-model"):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load and configure the model with proper memory management"""
        print(f"Using device: {self.device}")
        
        # Configure tokenizer (PyTorch-specific)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            use_fast=True,
        )
        
        if self.device == "cuda":
            # Get available GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = total_memory * 0.8  # Use 80% of available memory
            
            # Configure 4-bit quantization with PyTorch settings
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
            # Directly load the model with quantization config
            # Avoid using init_empty_weights which causes the meta tensor error
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=quantization_config,
                device_map="auto",  # Let PyTorch handle memory allocation
                torch_dtype=torch.float16
            )
            
            # Load adapter
            self.model = PeftModel.from_pretrained(
                self.model, 
                self.adapter_path
            )
            
            print(f"Model loaded successfully on GPU with {int(available_memory / 1024**3)}GB memory allocation!")
        else:
            print("Warning: Running on CPU. Performance will be significantly slower.")
            # CPU configuration
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

    def classify(self, problem_description):
        """Classify a problem into categories"""
        # Create prompt
        prompt = f"""Given the following programming problem, classify it into one of these categories:
Dynamic Programming, Math, Tree, Depth-first Search, Greedy, Hash Table, Binary Search, Breadth-first Search, Sort, Two Pointers, Backtracking, Stack, Graph, Bit Manipulation, Heap, Linked List, Recursion, Union Find, Sliding Window, Trie, Divide and Conquer, Segment Tree, Ordered Map, Queue

Problem description:
{problem_description}

The problem type is:"""

        # Tokenize and generate with PyTorch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )

        # Decode and extract the classification
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_text = generated_text.split("The problem type is:")[-1].strip()

        # List of possible categories
        categories = [
            "Dynamic Programming", "Math", "Tree", "Depth-first Search", "Greedy", 
            "Hash Table", "Binary Search", "Breadth-first Search", "Sort", 
            "Two Pointers", "Backtracking", "Stack", "Graph", "Bit Manipulation", 
            "Heap", "Linked List", "Recursion", "Union Find", "Sliding Window", 
            "Trie", "Divide and Conquer", "Segment Tree", "Ordered Map", "Queue"
        ]

        # Find all matching categories
        matches = []
        for category in categories:
            if category.lower() in predicted_text.lower():
                matches.append(category)

        # Return top 2 matches if available, otherwise return what we have
        if len(matches) >= 2:
            return matches[:2]
        elif len(matches) == 1:
            return [matches[0], "No second match found"]
        else:
            return [predicted_text, "No match found"]

def main():
    classifier = ProblemClassifier()
    
    while True:
        print("\n" + "="*50)
        problem = input("\nEnter the LeetCode problem description (press Enter to quit): ").strip()
        if not problem:
            print("\nExiting...")
            break
        results = classifier.classify(problem)
        print(f"\nTop matches:\n1. {results[0]}\n2. {results[1]}")

if __name__ == "__main__":
    main()