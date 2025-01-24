# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0
import os

from sentence_transformers.training_args import MultiDatasetBatchSamplers

os.environ["HF_DATASETS_CACHE"] = "/mnt/nfs/nomic_data_hf/"
os.environ["HF_HOME"] = "/mnt/nfs/nomic_data_hf/"
os.environ["WANDB_PROJECT"] = "ModernColBERT"

import argparse

from datasets import DatasetDict, load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils


def load_train_datasets():
    """Load all available splits from nomic-embed-unsupervised-data, with caching"""
    cache_dir = "/mnt/nfs/nomic_data_hf/cached_datasets"
    os.makedirs(cache_dir, exist_ok=True)
    train_dataset = DatasetDict()
    try:
        train_dataset = DatasetDict.load_from_disk(cache_dir)
        print("Loaded cached datasets.")
        return train_dataset
    except FileNotFoundError:
        print("No cached datasets found. Loading datasets...")
        splits = [
            "reddit_title_body",
            "amazon_reviews",
            "paq",
            "s2orc_citation_titles",
            "s2orc_title_abstract",
            "s2orc_abstract_citation",
            "s2orc_abstract_body",
            "wikianswers",
            "wikipedia",
            "gooaq",
            "codesearch",
            "yahoo_title_answer",
            "agnews",
            "amazonqa",
            "yahoo_qa",
            "yahoo_title_question",
            "ccnews",
            "npr",
            "eli5",
            "cnn",
            "stackexchange_duplicate_questions",
            "stackexchange_title_body",
            "stackexchange_body_body",
            "sentence_compression",
            "wikihow",
            "altlex",
            "quora",
            "simplewiki",
            "squad",
        ]

        for split in splits:
            print(f"Loading {split} dataset...")
            # data_files = {split: f"data/{split}-*"}
            dataset = load_dataset(
                "nomic-ai/nomic-embed-unsupervised-data", split=split
            )
            train_dataset[split] = dataset
            print(f"Loaded {split} dataset with {len(dataset)} examples.")
        train_dataset.save_to_disk(cache_dir)
        return train_dataset


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Load datasets
    train_dataset = load_train_datasets()
    print("Training dataset:", train_dataset)

    # Define training parameters
    num_train_epochs = 1
    lr = 3e-6
    batch_size = 256
    # model_name = "answerdotai/ModernBERT-base"
    model_name = "nomic-ai/modernbert-embed-base"
    model_shortname = model_name.split("/")[-1]

    # Set run name and output directory
    run_name = f"{model_shortname}-contrastive-nomic-unsupervised-{lr}-perdataset"
    output_dir = f"output/{model_shortname}/{run_name}"

    # Initialize model
    model = models.ColBERT(model_name_or_path=model_name, document_length=180)

    # Setup evaluation and loss
    dev_evaluator = evaluation.NanoBEIREvaluator()
    train_loss = losses.Contrastive(model=model)

    # Configure training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        eval_strategy="steps",
        eval_steps=5000,
        save_steps=5000,
        logging_steps=50,
        fp16=False,
        bf16=True,
        run_name=run_name,
        learning_rate=lr,
    )

    # Initialize and run trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(model.tokenize),
    )

    trainer.train()
    model.save_pretrained(f"{output_dir}/final")


if __name__ == "__main__":
    main()
