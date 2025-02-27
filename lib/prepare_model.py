import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_standards(file_path: str) -> List[Dict]:
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data['ansible_standards']
    except (yaml.YAMLError, KeyError, IOError) as e:
        logging.error(f"Error loading standards file: {e}")
        raise

def prepare_model(standards_file: str, model_name: str, output_dir: str):
    try:
        standards = load_standards(standards_file)

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Explicitly set padding token
        special_tokens_dict = {'pad_token': '[PAD]'}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

        # Ensure the pad_token_id is set in the model config
        model.config.pad_token_id = tokenizer.pad_token_id

        logging.info(f"Padding token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")

        # Save the model and tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Save standards as a part of the model artifacts
        with open(f"{output_dir}/standards.yaml", 'w') as f:
            yaml.dump({'ansible_standards': standards}, f)

        logging.info(f"Model prepared and saved to {output_dir}")
    except Exception as e:
        logging.error(f"Error during model preparation: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare the Ansible code review model")
    parser.add_argument("--standards_file", default="standards.yaml", help="Path to the standards YAML file")
    parser.add_argument("--model_name", default="facebook/incoder-1B", help="Name of the base model to use")
    parser.add_argument("--output_dir", default="./prepared_model", help="Directory to save the prepared model")
    
    args = parser.parse_args()
    
    prepare_model(args.standards_file, args.model_name, args.output_dir)