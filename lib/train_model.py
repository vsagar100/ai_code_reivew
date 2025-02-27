import torch
import logging
from transformers import Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from lib.code_review import CodeReviewer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrainerModule:
    def __init__(self, reviewer: CodeReviewer, standards_file: str):
        self.reviewer = reviewer
        self.standards_file = standards_file

    def prepare_dataset(self) -> Dataset:
        import yaml
        try:
            with open(self.standards_file, 'r') as file:
                standards = yaml.safe_load(file)['ansible_standards']

            examples = []
            responses = []
            for standard in standards:
                prompt = (
                    f"Standard: {standard['name']}\n"
                    f"Description: {standard['description']}\n"
                    f"Good example:\n{standard['example']['good']}\n"
                    f"Bad example:\n{standard['example']['bad']}\n"
                    f"Please review and improve the bad example based on the given standard."
                )
                response = f"The code should follow the guidelines provided in the standard: {standard['name']}."
                examples.append(prompt)
                responses.append(response)

            return Dataset.from_dict({"prompt": examples, "response": responses})

        except Exception as e:
            logging.error(f"Error preparing dataset: {e}")
            raise

    def train_model(self, output_dir: str):
        dataset = self.prepare_dataset()
        tokenizer = self.reviewer.tokenizer

        def tokenize_function(examples):
            full_input = [p + " " + r for p, r in zip(examples['prompt'], examples['response'])]
            tokens = tokenizer(full_input, padding='max_length', truncation=True, max_length=256, return_tensors="pt")
            tokens['labels'] = tokens['input_ids'].clone()
            return tokens

        # Tokenize dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "response"])

        # Set training arguments to optimize memory usage
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,  # Reduce batch size to save memory
            gradient_accumulation_steps=16,  # Increase accumulation steps to reduce GPU load
            fp16=False,  # Switch to full precision to avoid GPU compatibility issues with fp16
            num_train_epochs=3,
            logging_dir="./logs",
            save_steps=10_000,
            logging_steps=500,
            evaluation_strategy="steps",
            save_total_limit=2,
            report_to=[],  # Disable wandb
            dataloader_num_workers=1,  # Reduce number of data loading threads
            optim="adamw_torch"
        )

        # Set `use_cache` to False for gradient checkpointing compatibility
        self.reviewer.model.config.use_cache = False

        # Initialize Trainer
        trainer = Trainer(
            model=self.reviewer.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_dataset
        )

        try:
            # Clear CUDA cache before training
            torch.cuda.empty_cache()

            trainer.train()
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)

            logging.info(f"Model trained and saved to {output_dir}")

        except torch.cuda.OutOfMemoryError:
            logging.error("Out of memory error occurred during training. Try reducing batch size or other memory settings.")
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise
