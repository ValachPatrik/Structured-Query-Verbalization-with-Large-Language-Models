import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

'''
Given a trained sentence embedding model, this code creates histograms showing bert scores for true and false combinations
'''

QUERY_ROW_NAME = 'sparql_wikidata'
QUESTION_ROW_NAME = 'question'


# Load the dataset
df = pd.read_csv('data/v2/lc_quad_preprocessed.csv')

df = df.iloc[:10000]
# Split the dataset into test
test_df = df[6000:6300]

# Reset index to ensure it starts from 0
test_df.reset_index(drop=True, inplace=True)

# Load the trained model
model = SentenceTransformer('./custom_sentence_embedding_model')
#model = SentenceTransformer('bert-base-uncased')


# Initialize lists to store cosine similarity scores
true_pair_similarities = []
false_pair_similarities = []

# Calculate cosine similarity for true and false pairs
for idx, row in test_df.iterrows():
    # Encode true pair (sparql_wikidata and paraphrased_question)
    query = str(row[QUERY_ROW_NAME])
    paraphrased = str(row['question'])

    query_embedding = model.encode(query, convert_to_tensor=True)
    paraphrased_embedding = model.encode(paraphrased, convert_to_tensor=True)

    # Cosine similarity for true pair
    true_similarity = util.pytorch_cos_sim(query_embedding, paraphrased_embedding).item()
    true_pair_similarities.append(true_similarity)

    # Create a false pair by pairing with the next row's paraphrased_question
    negative_idx = (idx + 1) % len(test_df)  # Ensure the index is within bounds
    false_paraphrased = str(test_df.loc[negative_idx, 'question'])
    false_paraphrased_embedding = model.encode(false_paraphrased, convert_to_tensor=True)

    # Cosine similarity for false pair
    false_similarity = util.pytorch_cos_sim(query_embedding, false_paraphrased_embedding).item()
    false_pair_similarities.append(false_similarity)

# Plot the distributions of cosine similarities
plt.figure(figsize=(10, 6))

# Histogram for true pairs
plt.hist(true_pair_similarities, bins=30, alpha=0.6, color='blue', label=r'$\cos\_sim(E_{q}, E_{NL_{true}})$')

# Histogram for false pairs
plt.hist(false_pair_similarities, bins=30, alpha=0.6, color='red', label='$\cos\_sim(E_{q}, E_{NL_{false}})$')

# Add labels and legend
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.title('Cosine Similarity Distribution for True and False Pairs')
plt.legend(loc='upper right')

# Show the plot
plt.show()
