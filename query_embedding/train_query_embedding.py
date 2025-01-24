import os
import pandas as pd
import random
from torch.utils.data import DataLoader
from sentence_transformers import (
    InputExample,
    losses,
    SentenceTransformer,
    evaluation
)

def load_dataset(filepath, limit=None):
    """
    Load and optionally limit the dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.
        limit (int, optional): Maximum number of rows to load. Defaults to None.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(filepath)
    if limit:
        df = df.iloc[:limit]
    print(f"Loaded {len(df)} rows from {filepath}")
    return df

def split_dataset(df, train_size, test_size):
    """
    Split the dataset into training and testing sets.

    Args:
        df (pd.DataFrame): The complete dataset.
        train_size (int): Number of training samples.
        test_size (int): Number of testing samples.

    Returns:
        tuple: Training and testing DataFrames.
    """
    train_df = df[:train_size].reset_index(drop=True)
    test_df = df[train_size:train_size + test_size].reset_index(drop=True)
    return train_df, test_df

def create_input_examples(df, query_col, question_col, negative_col=None, easy=True):
    """
    Create positive and negative InputExamples for training/testing.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        query_col (str): Column name for queries.
        question_col (str): Column name for questions.
        negative_col (str). Column name for column with a negative example for the given query.

    Returns:
        list: List of InputExample instances.
    """
    examples = []
    for idx, row in df.iterrows():
        # Positive pair
        examples.append(InputExample(
            texts=[str(row[query_col]), str(row[question_col])],
            label=1.0
        ))
        if negative_col:
            # hard negative pair
            examples.append(InputExample(
                texts=[str(row[query_col]), str(row[negative_col])],
                label=0.0
            ))
        if easy: # todo maybe change
            if random.random() > 0.7:
                # easy Negative pair
                negative_idx = (idx + 1) % len(df)
                examples.append(InputExample(
                    texts=[str(row[query_col]), str(df.loc[negative_idx, question_col])],
                    label=0.0
                ))
    return examples

def configure_environment(wandb_mode="offline"):
    """
    Set environment variables for WandB.

    Args:
        wandb_mode (str, optional): Mode for WandB. Defaults to "offline".
    """
    os.environ["WANDB_MODE"] = wandb_mode
    if wandb_mode == "offline":
        os.environ["WANDB_DISABLED"] = "true"


def train_model(model, train_dataloader, train_loss, evaluator, output_path, epochs=1):
    """
    Train the SentenceTransformer model.

    Args:
        model (SentenceTransformer): The model to train.
        train_dataloader (DataLoader): DataLoader for training data.
        train_loss (losses.Loss): Loss function.
        evaluator (evaluation.BinaryClassificationEvaluator): Evaluator for validation.
        output_path (str): Path to save the trained model.
        epochs (int, optional): Number of training epochs. Defaults to 1.
    """
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        evaluation_steps=len(train_dataloader) // 8,
        output_path=output_path,
        save_best_model=True
    )

def main():
    # Settings
    DO_TRAIN = True
    QUERY_COL = 'instantiated_query'
    QUESTION_COL = 'question'
    DATA_PATH = '../data/v2/hard_negatives_gpt4_row_0_to_1000_and_4000_to_4900.csv'
    #MODEL_NAME = 'all-MiniLM-L6-v2'
    MODEL_NAME = "../embedding_models/ModernBERT_untrained"
    OUTPUT_PATH = '../MODEL'
    TRAIN_SIZE = 1600
    TEST_SIZE = 190
    EPOCHS=3

    # Configure environment
    configure_environment()

    # Load and prepare dataset
    df = load_dataset(DATA_PATH)
    train_df, test_df = split_dataset(df, TRAIN_SIZE, TEST_SIZE)

    # Create training and testing examples
    train_examples = create_input_examples(train_df, QUERY_COL, QUESTION_COL, negative_col='negative_significant',
                                           easy=False)
    test_examples = create_input_examples(test_df, QUERY_COL, QUESTION_COL, negative_col='negative_significant',
                                          easy=False)

    test_examples_easy = create_input_examples(test_df, QUERY_COL, QUESTION_COL, easy=True)
    test_examples_hard = create_input_examples(test_df, QUERY_COL, QUESTION_COL, negative_col='negative_significant', easy=False)


    # Load pre-trained model
    model = SentenceTransformer(MODEL_NAME)


    # Setup DataLoader and loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    # Setup evaluators
    test_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
        test_examples, name='test-eval'
    )
    test_evaluator_binary = evaluation.BinaryClassificationEvaluator.from_input_examples(
        test_examples, name='test-binary-eval'
    )

    train_evaluator_binary = evaluation.BinaryClassificationEvaluator.from_input_examples(
        train_examples, name='train-binary-eval'
    )

    test_evaluator_easy = evaluation.BinaryClassificationEvaluator.from_input_examples(
        test_examples_easy, name='test-binary-easy'
    )
    test_evaluator_hard = evaluation.BinaryClassificationEvaluator.from_input_examples(
        test_examples_hard, name='test-binary-hard'
    )

    # Train the model
    if DO_TRAIN:
        train_model(
            model=model,
            train_dataloader=train_dataloader,
            train_loss=train_loss,
            evaluator=test_evaluator_binary,
            output_path='../TRAINING_RESULTS',
            epochs=EPOCHS
        )

    # Save the trained model
    model.save(OUTPUT_PATH)
    print('Model saved to', OUTPUT_PATH)

    # Final evaluation
    print('Final evaluation after training:')
    results_embed = model.evaluate(test_evaluator)
    print("Embedding Similarity Evaluator:", results_embed)
    results_binary = model.evaluate(test_evaluator_binary)
    print("Binary Classification Evaluator:", results_binary)
    results_binary = model.evaluate(train_evaluator_binary)
    print("Binary Classification Evaluator TRAIN:", results_binary)

    results_binary = model.evaluate(test_evaluator_easy)
    print("Binary Classification Evaluator Test Easy:", results_binary)
    results_binary = model.evaluate(test_evaluator_hard)
    print("Binary Classification Evaluator Test Hard:", results_binary)


if __name__ == "__main__":
    main()
