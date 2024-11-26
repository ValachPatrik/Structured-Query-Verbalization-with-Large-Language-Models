from sentence_transformers import InputExample, losses, SentenceTransformer, evaluation
from torch.utils.data import DataLoader
import pandas as pd

import os

#Settings:

DO_TRAIN = True

# Set the column names for sparql_wikidata and paraphrased_question
query_row_name = 'sparql_wikidata'
question_row_name = 'question'

os.environ["WANDB_MODE"] = "offline"



# Load the dataset
df = pd.read_csv('lc_quad_preprocessed.csv')

print(len(df))

df = df.iloc[:70000]


# Split the dataset into train and test
train_df = df[:6000]
test_df = df[6000:6300]

# Reset index to ensure it starts from 0 for both train and test
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# Reset index to ensure it starts from 0 for both train and test
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# Create training examples
train_examples = []
for idx, row in train_df.iterrows():
    # Using sparql_wikidata and paraphrased_question for positive pairs
    train_examples.append(
        InputExample(texts=[str(row[query_row_name]), str(row['question'])], label=1.0))

    # Creating negative pairs by pairing each paraphrased_question with sparql_wikidata of a different row
    negative_idx = (idx + 1) % len(train_df)  # Ensure the index is within bounds
    train_examples.append(
        InputExample(texts=[str(row[query_row_name]), str(train_df.loc[negative_idx, 'question'])],
                     label=0.0))

# Create test examples
test_examples = []
for idx, row in test_df.iterrows():
    test_examples.append(InputExample(texts=[str(row[query_row_name]), str(row['question'])], label=1.0))
    negative_idx = (idx + 1) % len(test_df)  # Ensure the index is within bounds
    test_examples.append(
        InputExample(texts=[str(row[query_row_name]), str(test_df.loc[negative_idx, 'question'])],
                     label=0.0))

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define a DataLoader to load training examples in batches
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Use a contrastive loss function for training
train_loss = losses.CosineSimilarityLoss(model)

# Configure the training
num_epochs = 1
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up

# Evaluator for validation during training
test_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(test_examples, name='test-eval')
test_evaluator_binary = evaluation.BinaryClassificationEvaluator.from_input_examples(test_examples, name='test-binary-eval')

def run_multiple_evaluators(model, evaluators):
    for evaluator in evaluators:
        evaluator(model, output_path='./custom_sentence_embedding_model')


# Train the model
if DO_TRAIN:
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        evaluator=test_evaluator_binary,
        evaluation_steps=len(train_dataloader) // 8,  # Evaluate X times per epoch
        output_path='./custom_sentence_embedding_model'
    )

# Save the model
model.save('./custom_sentence_embedding_model')

print('Final evaluation after training:')
# Final evaluation of the model
R = model.evaluate(test_evaluator)
print(R)
R = model.evaluate(test_evaluator_binary)
print(R)

