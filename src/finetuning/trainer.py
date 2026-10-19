"""Training logic for finetuning."""

import logging
import time

import wandb
from datasets import Dataset
from trl import DataCollatorForCompletionOnlyLM
from trl import SFTConfig, SFTTrainer

from config_finetuning import FinetuningConfig
from utils_finetuning import get_response_template

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handle model training with SFT."""
    
    def __init__(self, config: FinetuningConfig):
        self.config = config
    
    def train(
        self,
        model,
        tokenizer,
        train_dataset: Dataset,
        eval_dataset: Dataset
    ):
        """
        Execute training loop.
        
        Args:
            model: Model to train (with LoRA applied)
            tokenizer: Tokenizer
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
        """
        # Setup training arguments
        training_args = SFTConfig(
            output_dir=str(self.config.get_output_path()),
            num_train_epochs=self.config.model_config.num_epochs,
            per_device_train_batch_size=self.config.model_config.train_batch_size,
            per_device_eval_batch_size=self.config.model_config.eval_batch_size,
            gradient_accumulation_steps=self.config.model_config.gradient_accumulation_steps,
            gradient_checkpointing=self.config.model_config.gradient_checkpointing,
            max_seq_length=self.config.model_config.max_length,
            dataset_text_field="text",

            # Evaluation and saving
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1, # Keep last checkpoint
            save_only_model=True,
            logging_steps=10,

            # Optimization
            learning_rate=self.config.model_config.learning_rate,
            weight_decay=self.config.model_config.weight_decay,
            warmup_steps=self.config.model_config.warmup_steps,
            optim=self.config.model_config.optimizer,
            lr_scheduler_type=self.config.model_config.lr_scheduler_type,

            # Precision
            bf16=self.config.model_config.bf16,

            # Other
            seed=self.config.seed_config.seed,
            report_to="wandb",
            run_name=self.config.wandb_run_name,
        )

        # Data collator
        response_template = get_response_template(self.config.model_config.model_name)
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer
        )

        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            data_collator=data_collator
        )
        
        # Train
        logger.info("\nStarting finetuning...\n")
        
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        
        # Log training time
        elapsed = end_time - start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info(
            f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s"
        )
        
        # Save final model
        final_path = self.config.get_output_path() / "final_model"
        trainer.save_model(str(final_path))
        logger.info(f"Final model saved to {final_path}")
        
        # Finish WandB
        wandb.finish()