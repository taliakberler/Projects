"""
TruthSeeker Real or Fake News Tweet Detection

Talia Berler
CSC616
Professor Stephen Dennis

LLM.py combines the StackedTransformerLLM class architecture
with the training pipeline from base_model.py. The model stacks a language model on top
of a transformer base model for improved classification of fake news tweets.
"""

# ----------------------------
# Set Global Vars, Import Libraries
# ----------------------------

# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import regex
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from huggingface_hub import HfFolder
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import logging
import torch
import torch.nn as nn

# Set global path var
PATH = "/scratch/scratch/projects/bert_news/"
INPUT = "new_dataset.csv"
OUTPUT = "output/"

# Set HF token
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    HfFolder.save_token(hf_token)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Save eval results for all models
all_model_results = {}

MODEL_MAP = {
    "bert": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    "bertweet": "vinai/bertweet-base",  # BERTweet requires normalization
    "roberta": "roberta-base"
}

# Choose LLM model here
LLM_MODEL = "deepseek-ai/deepseek-llm-7b-base"  # Update with correct model ID

# Create a timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Path to output folder
experiment_base_dir = PATH+OUTPUT

# Verify the base path exists
if not os.path.exists(experiment_base_dir):
    logger.info(f"Warning: Base directory {experiment_base_dir} does not exist!")
    logger.info("Creating the directory structure...")
    os.makedirs(experiment_base_dir, exist_ok=True)

# Create results directory for this run
experiment_dir = os.path.join(experiment_base_dir, f"experiment_{timestamp}")
logger.info(f"Creating experiment directory: {experiment_dir}") 
os.makedirs(experiment_dir, exist_ok=True)
logger.info(f"Experiment directory created at: {experiment_dir}")


# ----------------------------
# Define Stacked Transformer LLM Class
# ----------------------------

class StackedTransformerLLM(nn.Module):
    def __init__(self, base_model_name, llm_model_name, num_labels=2):
        super(StackedTransformerLLM, self).__init__()

        # Load pre-trained base model (e.g., BERT)
        self.base_model = AutoModel.from_pretrained(base_model_name)

        # Load small LLM for classification
        self.llm_model = AutoModelForSequenceClassification.from_pretrained(
            llm_model_name,
            num_labels=num_labels  # final output classes
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # Process different input parameters based on model type
        base_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Add token_type_ids if provided (needed for BERT but not for all models)
        if token_type_ids is not None:
            base_inputs['token_type_ids'] = token_type_ids
            
        # 1. Pass through base transformer
        base_outputs = self.base_model(**base_inputs)
        hidden_states = base_outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_dim)

        # 2. Pool the [CLS] token (first token)
        cls_embedding = hidden_states[:, 0, :]  # shape: (batch_size, hidden_dim)

        # 3. Expand to simulate sequence input for LLM
        cls_embedding_expanded = cls_embedding.unsqueeze(1)  # shape: (batch_size, 1, hidden_dim)

        # 4. Pass through LLM model (using inputs_embeds instead of input_ids)
        outputs = self.llm_model(inputs_embeds=cls_embedding_expanded, labels=labels)

        return outputs


# ----------------------------
# 1. Transformer Model Set Up
# 2. Evaluation and Visualization of Metrics
# ----------------------------

def normalize_tweet(text):
    """
    Normalize tweet text following BERTweet requirements:
    - Convert user mentions to @USER
    - Convert URLs to HTTPURL
    """

    # Replace user mentions (simple regex for demonstration)

    text = regex.sub(r'@\w+', '@USER', text)

    # Replace URLs
    text = regex.sub(r'https?://\S+|www\.\S+', 'HTTPURL', text)

    return text

def tokenize_data(dataset, tokenizer, model_name=None):
    logger.info(f"Tokenizing dataset with {tokenizer.__class__.__name__}")

    # Special handling for different model types
    tokenizer_kwargs = {
        'truncation': True,
        'padding': 'max_length'
    }
    
    # Set max_length based on model type
    if model_name and 'bertweet' in model_name.lower():
        tokenizer_kwargs['max_length'] = 128  # BERTweet for tweets
    else:
        tokenizer_kwargs['max_length'] = 300  # Other models
        
    # Determine whether to include token_type_ids
    # BERT and DistilBERT need token_type_ids
    if 'bert' in model_name.lower() and 'roberta' not in model_name.lower() and 'bertweet' not in model_name.lower():
        tokenizer_kwargs['return_token_type_ids'] = True
    
    # Special handling for BERTweet
    if model_name and 'bertweet' in model_name.lower():
        logger.info("Applying BERTweet-specific normalization")

        # Function to normalize and tokenize for BERTweet
        def tokenize_bertweet(examples):
            # First normalize tweets
            normalized_texts = [normalize_tweet(text) for text in examples['text']]
            # Use the tokenizer with our kwargs
            return tokenizer(normalized_texts, **tokenizer_kwargs)

        tokenized_data = dataset.map(tokenize_bertweet, batched=True)
    else:
        # Regular tokenization for other models
        tokenized_data = dataset.map(
            lambda x: tokenizer(x['text'], **tokenizer_kwargs),
            batched=True
        )

    # Check the structure after tokenization
    logger.info("Tokenized dataset structure:")
    logger.info(str(tokenized_data[0]))  # Check the first item for the expected fields
    return tokenized_data

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)

    # Create confusion matrix for later visualization
    cm = confusion_matrix(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist()  # Convert to list for JSON serialization
    }

def plot_training_metrics(trainer, model_dir, model_name):
    """Plot training and validation metrics"""
    try:
        # Extract metrics from trainer logs
        logs = trainer.state.log_history

        # Filter log entries for training and validation
        train_logs = [log for log in logs if 'loss' in log and 'eval_loss' not in log]
        eval_logs = [log for log in logs if 'eval_loss' in log]

        if train_logs and eval_logs:
            # Plot training loss
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            train_steps = [log.get('step', i) for i, log in enumerate(train_logs)]
            train_loss = [log['loss'] for log in train_logs]
            plt.plot(train_steps, train_loss)
            plt.title('Training Loss')
            plt.xlabel('Step')
            plt.ylabel('Loss')

            # Plot evaluation metrics
            plt.subplot(1, 2, 2)
            eval_steps = [log.get('step', i) for i, log in enumerate(eval_logs)]
            eval_f1 = [log.get('eval_f1', 0) for log in eval_logs]
            eval_acc = [log.get('eval_accuracy', 0) for log in eval_logs]

            plt.plot(eval_steps, eval_f1, label='F1')
            plt.plot(eval_steps, eval_acc, label='Accuracy')
            plt.title('Evaluation Metrics')
            plt.xlabel('Step')
            plt.ylabel('Score')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, f'{model_name}_training_metrics.png'))
            logger.info(f"Training metrics plot saved to {model_dir}/{model_name}_training_metrics.png")
        else:
            logger.warning("Not enough logs to generate training metrics plot")
    except Exception as e:
        logger.error(f"Error plotting training metrics: {str(e)}")

def plot_confusion_matrix(cm, model_dir, model_name, phase="test"):
    """Plot and save confusion matrix"""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix ({phase})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(model_dir, f'{model_name}_confusion_matrix_{phase}.png'))
        logger.info(f"Confusion matrix plot saved to {model_dir}/{model_name}_confusion_matrix_{phase}.png")
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")

def plot_roc(y_pred_prob, y_true, model_dir, model_name):
    """Plot and save ROC curve for model"""
    try:
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)  

        # Plot ROC
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(model_dir, f'{model_name}_roc.png'))

        # Plot TPR and FPR vs. Threshold
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, tpr, label='True Positive Rate (Sensitivity)')
        plt.plot(thresholds, 1 - fpr, label='True Negative Rate (Specificity)')
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.title('TPR and TNR vs Threshold')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(model_dir, f'{model_name}_TPR_vs_FPR.png'))
    except Exception as e:
        logger.error(f"Error plotting ROC curve: {str(e)}")


class MetricCallback(EarlyStoppingCallback):
    """Custom callback to log and save metrics during training"""
    def __init__(self, model_name, save_dir, patience=3, threshold=0.0):
        super().__init__(early_stopping_patience=patience, early_stopping_threshold=threshold)
        self.model_name = model_name
        self.save_dir = save_dir
        self.metrics_history = {
            'train': [],
            'eval': [],
            'test': []
        }

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            # Save metrics to history
            self.metrics_history['eval'].append({
                'step': state.global_step,
                'epoch': state.epoch,
                'metrics': metrics
            })

            # Save metrics to file
            metrics_file = os.path.join(self.save_dir, 'eval_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)

            # Log metrics
            phase = "Evaluation"
            logger.info(f"{phase} metrics at step {state.global_step}, epoch {state.epoch:.2f}:")
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    logger.info(f"  {k}: {v:.4f}")

        # Call parent class for early stopping logic
        super().on_evaluate(args, state, control, metrics, **kwargs)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Save training logs
            if 'loss' in logs and 'eval_loss' not in logs:
                self.metrics_history['train'].append({
                    'step': state.global_step,
                    'epoch': state.epoch,
                    'loss': logs.get('loss')
                })

            # Save metrics to file with each log
            metrics_file = os.path.join(self.save_dir, 'training_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)


# Function to train model with stacked LLM architecture
def train_model(model_name, dataset):
    logger.info(f"\n--- Training {model_name.upper()} with LLM Classification Head ---\n")

    # Initialize results dictionary for this model
    all_model_results[model_name] = {'config': {}, 'train_results': {}, 'val_results': {}, 'test_results': {}}

    # Create model-specific directory
    model_dir = os.path.join(experiment_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Load and split dataset
    df = dataset
    logger.info(f"Dataset head:\n{df.head()}")
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    # Then, split the train set into train and validation sets
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)  # 10% of train for validation

    # Log the shapes of the datasets
    logger.info(f"Train dataframe shape: {train_df.shape}")
    logger.info(f"Validation dataframe shape: {val_df.shape}")
    logger.info(f"Test dataframe shape: {test_df.shape}")

    # Save dataset splits for reproducibility
    train_df.to_csv(os.path.join(model_dir, "train_data.csv"), index=False)
    val_df.to_csv(os.path.join(model_dir, "val_data.csv"), index=False)
    test_df.to_csv(os.path.join(model_dir, "test_data.csv"), index=False)

    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    # Log the structure of the HuggingFace dataset
    logger.info(f"Train dataset columns: {train_dataset.column_names}")
    logger.info(f"Validation dataset columns: {val_dataset.column_names}")
    logger.info(f"Test dataset columns: {test_dataset.column_names}")

    # Tokenizer & model
    logger.info(f"Loading tokenizer for base model: {MODEL_MAP[model_name]}")

    # Special configuration for BERTweet
    tokenizer_kwargs = {"use_fast": True}
    if 'bertweet' in model_name.lower():
        tokenizer_kwargs["normalization"] = True

    # Save tokenizer for later use
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[model_name], **tokenizer_kwargs)
    tokenizer.save_pretrained(os.path.join(model_dir, "tokenizer"))
    logger.info(f"Tokenizer saved to {model_dir}/tokenizer")

    # INITIALIZE STACKED MODEL: Base Transformer + LLM head
    logger.info(f"Initializing stacked model with base {MODEL_MAP[model_name]} and LLM head {LLM_MODEL}")
    model = StackedTransformerLLM(
        base_model_name=MODEL_MAP[model_name],
        llm_model_name=LLM_MODEL,
        num_labels=2
    )

    # Tokenization
    train_dataset = tokenize_data(train_dataset, tokenizer, model_name)
    val_dataset = tokenize_data(val_dataset, tokenizer, model_name)
    test_dataset = tokenize_data(test_dataset, tokenizer, model_name)

    # Collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(model_dir, "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,  # Slightly lower learning rate for the LLM stack
        per_device_train_batch_size=8,   # Reduced batch size due to larger model
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,   # Increased for larger model
        num_train_epochs=5,              # Reduced epochs for larger model
        weight_decay=0.01,
        logging_dir=os.path.join(model_dir, "logs"),
        report_to="none",  # Disable logging to W&B by default
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",  # set metric for early stopping
        greater_is_better=True,  # set to True for accuracy or F1
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        run_name=f"{model_name}_llm_run",  # custom run name
        # Additional arguments for better logging
        logging_first_step=True,
        save_total_limit=3,  # only keep the 3 best checkpoints
        fp16=True,  # Use mixed precision training to reduce memory usage
        push_to_hub=True,
        hub_model_id=f"wildgeese25/{model_name.replace('_','-')}-fake-news-detector-LLM-stacked",
        hub_strategy="end"
    )

    # Save training configuration
    training_config = {
        "model_name": model_name,
        "base_model_path": MODEL_MAP[model_name],
        "llm_model_path": LLM_MODEL,
        "dataset_path": PATH+INPUT,
        "training_arguments": training_args.to_dict(),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "timestamp": timestamp
    }
    all_model_results[model_name]['config'] = training_config

    # Initialize custom callback
    metric_callback = MetricCallback(model_name, model_dir, patience=3, threshold=0.0)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,  # Updated to use tokenizer directly
        callbacks=[metric_callback],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Start training
    logger.info("Starting training...")
    train_output = trainer.train()

    # Log training summary
    logger.info(f"Training completed in {train_output.metrics['train_runtime']:.2f} seconds")
    logger.info(f"Training loss: {train_output.metrics['train_loss']:.4f}")

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    all_model_results[model_name]['val_results'] = eval_results

    logger.info(f"\nEval results for {model_name} (Validation):")
    for k, v in eval_results.items():
        if isinstance(v, (int, float)):
            logger.info(f"{k}: {v:.4f}")

    # Plot confusion matrix for validation
    if 'eval_confusion_matrix' in eval_results:
        plot_confusion_matrix(
            np.array(eval_results['eval_confusion_matrix']),
            model_dir,
            model_name,
            phase="validation"
        )

    # Now, perform testing on the test dataset after training
    logger.info(f"\n--- Testing {model_name.upper()} with LLM head ---\n")
    test_results = trainer.evaluate(test_dataset)  # Evaluate on the test dataset
    all_model_results[model_name]['test_results'] = test_results

    logger.info(f"\nTest results for {model_name}:")
    for k, v in test_results.items():
        if isinstance(v, (int, float)):
            logger.info(f"{k}: {v:.4f}")

    # Plot confusion matrix for test
    if 'eval_confusion_matrix' in test_results:
        plot_confusion_matrix(
            np.array(test_results['eval_confusion_matrix']),
            model_dir,
            model_name,
            phase="test"
        )

    # Generate detailed classification report
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    # Generate ROC curve and save to directory
    try:
        # For ROC curve we need probabilities
        probabilities = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)
        plot_roc(probabilities[:, 1].numpy(), labels, model_dir, model_name=model_name)
    except Exception as e:
        logger.error(f"Error generating ROC curve: {e}")
        # Fallback to using raw predictions
        plot_roc(predictions.predictions[:, 1], labels, model_dir, model_name=model_name)

    # Generate classification report
    class_report = classification_report(labels, preds, output_dict=True)

    # Save classification report
    with open(os.path.join(model_dir, 'classification_report.json'), 'w') as f:
        json.dump(class_report, f, indent=2)
    logger.info(f"Classification report saved to {model_dir}/classification_report.json")

    # Generate and save training metrics plots
    plot_training_metrics(trainer, model_dir, model_name)

    # Save model and tokenizer
    logger.info(f"Saving model to {model_dir}/final_model")
    trainer.save_model(os.path.join(model_dir, "final_model"))

    # Save all results to a single JSON file
    with open(os.path.join(model_dir, 'all_results.json'), 'w') as f:
        # Convert any numpy or torch values to standard Python types
        serializable_results = json.loads(
            json.dumps(all_model_results[model_name], default=lambda o: float(o) if isinstance(o, (np.number, torch.Tensor)) else str(o))
        )
        json.dump(serializable_results, f, indent=2)

    logger.info(f"All results saved to {model_dir}/all_results.json")

    return all_model_results


def main():
    # ----------------------------
    # Load & Explore the Data
    # ----------------------------
    global all_model_results
   
    # Load dataset
    df = pd.read_csv(PATH+INPUT) 

    # Get overview of data
    logger.info(f"Saving preliminary information on input dataset: {PATH+INPUT}")
    logger.info(f"Data has shape: {df.shape}")
    logger.info(f"The target balance is: {df['label'].value_counts()}")
    logger.info("Dataframe head: ")
    logger.info(f"{df.head()}")

    # ----------------------------
    # Run model training and evaluation
    # ----------------------------

    # Loop through all models in MODEL_MAP
    # You can modify this to just run for specific base models
    for model in list(MODEL_MAP.keys()):
        # Train only the selected model
        logger.info(f"Training model: {model}")
        train_model(model, df)

    # Create a summary of results for all models
    summary = {
        "experiment_timestamp": timestamp,
        "model_summaries": {}
    }

    # Find best models
    if len(all_model_results) > 0:
        best_accuracy_model = max(all_model_results.items(), 
                                key=lambda x: x[1]['test_results'].get('eval_accuracy', 0))[0]
        best_f1_model = max(all_model_results.items(), 
                        key=lambda x: x[1]['test_results'].get('eval_f1', 0))[0]
        
        summary["best_model_by_accuracy"] = best_accuracy_model
        summary["best_model_by_f1"] = best_f1_model
        
        # Add metrics for each model
        for model_name, results in all_model_results.items():
            test_metrics = results['test_results']
            summary["model_summaries"][model_name] = {
                "accuracy": test_metrics.get('eval_accuracy', 0),
                "f1": test_metrics.get('eval_f1', 0),
                "precision": test_metrics.get('eval_precision', 0),
                "recall": test_metrics.get('eval_recall', 0)
            }
        
        # Save summary
        with open(os.path.join(experiment_dir, "experiment_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Experiment summary saved to {experiment_dir}/experiment_summary.json")
        logger.info(f"Best model by accuracy: {summary['best_model_by_accuracy']}")
        logger.info(f"Best model by F1: {summary['best_model_by_f1']}")

if __name__ == "__main__":
    main()
