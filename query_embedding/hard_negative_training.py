import os
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import (
    InputExample,
    losses,
    SentenceTransformer,
    evaluation
)
from sentence_transformers.util import mine_hard_negatives
from datasets import Dataset
import logging

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

def create_input_examples(df, query_col, question_col):
    """
    Create positive InputExamples for training/testing.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        query_col (str): Column name for queries.
        question_col (str): Column name for questions.

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

def mine_hard_negatives_dataset(train_examples, model, num_negatives=1, **kwargs):
    """
    Mine hard negatives using the provided model and parameters.

    Args:
        train_examples (list): List of InputExample instances (anchor, positive).
        model (SentenceTransformer): The SentenceTransformer model.
        num_negatives (int, optional): Number of hard negatives per example. Defaults to 1.
        **kwargs: Additional keyword arguments for mine_hard_negatives.

    Returns:
        list: List of InputExample instances with hard negatives.
    """
    # Convert InputExamples to a dictionary
    data = {
        'query': [ex.texts[0] for ex in train_examples],
        'answer': [ex.texts[1] for ex in train_examples]
    }

    # Create HuggingFace Dataset
    hf_dataset = Dataset.from_dict(data)

    # Mine hard negatives
    mined_dataset = mine_hard_negatives(
        dataset=hf_dataset,
        model=model,
        num_negatives=num_negatives,
        **kwargs
    )

    # Convert mined_dataset back to InputExamples with negatives
    final_examples = []
    for data in mined_dataset:
        anchor = data['query']
        positive = data['answer']
        negative = data['negative']
        # Create one InputExample per negative
        final_examples.append(InputExample(
            texts=[anchor, positive, negative],
            label=1  # Label can be used for TripletLoss if needed
        ))
    return final_examples

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
        output_path=output_path
    )

def main():
    # Settings
    DO_TRAIN = True
    QUERY_COL = 'sparql_wikidata_translated'
    QUESTION_COL = 'question'
    DATA_PATH = '../data/v2/lc_quad_preprocessed.csv'
    MODEL_NAME = 'all-MiniLM-L6-v2'
    OUTPUT_PATH = '../hard_neg_model_instantiated'
    TRAIN_SIZE = 1000  # Increased training size for better mining
    TEST_SIZE = 200
    NUM_NEGATIVES = 1  # Number of hard negatives per example

    # Configure environment
    configure_environment()

    # Load and prepare dataset
    df = load_dataset(DATA_PATH, limit=70000)
    train_df, test_df = split_dataset(df, TRAIN_SIZE, TEST_SIZE)

    # Create training and testing examples
    train_examples = create_input_examples(train_df, QUERY_COL, QUESTION_COL)
    test_examples = create_input_examples(test_df, QUERY_COL, QUESTION_COL)

    # Load pre-trained model
    model = SentenceTransformer(MODEL_NAME)
    #model = SentenceTransformer('../all-MiniLM-L6-v2-instantiated-wikidata-question')


    # Mine hard negatives
    print("Mining hard negatives...")
    hard_negatives = mine_hard_negatives_dataset(
        train_examples,
        model=model,
        num_negatives=NUM_NEGATIVES,
        range_min=10,
        range_max=50,
        max_score=0.8,
        margin=0.1,
        sampling_strategy="random",
        batch_size=128,
        use_faiss=True,
    )
    print(f"Generated {len(hard_negatives)} triplet examples with hard negatives.")

    if not hard_negatives:
        logging.error("No hard negatives were mined. Please adjust the mining parameters.")
        return


    train_dataloader = DataLoader(hard_negatives, shuffle=True, batch_size=16)

    train_loss = losses.TripletLoss(model=model)

    # Setup evaluators
    test_evaluator_binary = evaluation.BinaryClassificationEvaluator.from_input_examples(
        test_examples, name='test-binary-eval', batch_size=16
    )

    # Train the model
    if DO_TRAIN:
        train_model(
            model=model,
            train_dataloader=train_dataloader,
            train_loss=train_loss,
            evaluator=test_evaluator_binary,
            output_path=OUTPUT_PATH,
            epochs=1  # Increased epochs for better training
        )

    # Save the trained model
    model.save(OUTPUT_PATH)
    print('Model saved to', OUTPUT_PATH)

    # Final evaluation
    print('Final evaluation after training:')
    results_embed = model.evaluate(evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
        test_examples, name='test-eval', batch_size=16))
    print("Embedding Similarity Evaluator:", results_embed)
    results_binary = model.evaluate(test_evaluator_binary)
    print("Binary Classification Evaluator:", results_binary)

    # Optionally, push the mined dataset to HuggingFace Hub
    # Ensure you have the necessary credentials and permissions
    # mined_dataset.push_to_hub("your-username/natural-questions-hard-negatives", private=True)

if __name__ == "__main__":
    main()
