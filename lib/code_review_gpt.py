# code_review.py
from typing import List, Dict
import logging
import yaml
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig, 
    Trainer, 
    TrainingArguments
)
import torch

class CodeReviewer:
    def __init__(self, model_path: str):
        try:
            config = AutoConfig.from_pretrained(model_path)
            
            if config.max_position_embeddings < config.vocab_size:
                config.max_position_embeddings = config.vocab_size + 1
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float32
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            # Load standards
            with open(f"{model_path}/standards.yaml", 'r') as f:
                self.standards = yaml.safe_load(f).get('ansible_standards', [])
            
            if not self.standards:
                logging.warning("No standards found in standards.yaml.")
                
            logging.info(f"Model loaded successfully. Using device: {self.device}")
        except Exception as e:
            logging.error(f"Error initializing CodeReviewer: {e}")
            raise

    def review_code(self, code: str) -> List[Dict]:
        try:
            issues = []
            for standard in self.standards:
                prompt = f"""
                Review the following Ansible code according to this standard:
                {standard['name']}
                {standard['description']}

                Good example:
                {standard['example']['good']}

                Bad example:
                {standard['example']['bad']}

                Ansible code to review:
                {code}

                Provide a review focusing only on this standard. If there are issues, specify the line number, describe the issue, and suggest a fix. If there are no issues related to this standard, say "No issues found for this standard."

                Review:
                """

                logging.debug(f"Prompt:\n{prompt}")
                
                inputs = self.tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=1500,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        bos_token_id=self.tokenizer.bos_token_id
                    )
                
                review_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                logging.debug(f"Model output:\n{review_text}")

                try:
                    if "No issues found for this standard." not in review_text:
                        parsed_issues = self._parse_review(review_text, standard['name'])
                        if parsed_issues:
                            issues.extend(parsed_issues)
                except Exception as e:
                    logging.warning(f"Error parsing review for standard {standard['name']}: {e}")
                    continue

            return issues
        except Exception as e:
            logging.error(f"Error during code review: {e}")
            raise

    def _parse_review(self, review_text: str, standard_name: str) -> List[Dict]:
        issues = []
        try:
            lines = review_text.split('\n')
            current_issue = {}

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.lower().startswith("line"):
                    if current_issue and all(k in current_issue for k in ['standard', 'line_number', 'description']):
                        issues.append(current_issue)
                    current_issue = {
                        "standard": standard_name,
                        "line_number": self._extract_line_number(line),
                        "description": "",
                        "suggestion": ""
                    }
                elif line.lower().startswith("issue:"):
                    current_issue["description"] = line[6:].strip()
                elif line.lower().startswith("suggestion:"):
                    current_issue["suggestion"] = line[11:].strip()

            if current_issue and all(k in current_issue for k in ['standard', 'line_number', 'description']):
                issues.append(current_issue)
            
        except Exception as e:
            logging.warning(f"Error parsing review text: {e}")
            
        return issues

    def _extract_line_number(self, line: str) -> int:
        try:
            num = ''.join(filter(str.isdigit, line.split()[1]))
            return int(num)
        except (IndexError, ValueError):
            logging.warning(f"Could not extract line number from: {line}")
            return 0
