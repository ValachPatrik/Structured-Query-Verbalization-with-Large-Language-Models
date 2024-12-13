import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from scipy.stats import linregress
from transformers import BertModel, BertTokenizer
import numpy as np


'''
Given a trained sentence embedding model, this code creates histograms showing bert scores for true and false combinations
'''

def bert_embedding(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Scatter plot function with regression line and correlation coefficient
def scatter_plot_with_regression(df, x_col, y_col, title):
    # Extract x and y data
    x = x_col
    y = y_col

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line = slope * np.array(x) + intercept

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data points', color='blue', alpha=0.6)
    plt.plot(x, line, color='red', label=f'Regression line (r={r_value:.2f})')

    # Plot settings
    plt.xlabel(x_col, fontsize=14)
    plt.ylabel(y_col, fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()



QUERY_ROW_NAME = 'instantiated_query'
QUESTION_ROW_NAME = 'original_question'
TRANSLATION_ROW_NAME = 'best_translation_Q'

# Load the dataset
df = pd.read_csv('../results/1734028434/results.csv')

df = df.iloc[:120]


# Reset index to ensure it starts from 0
df.reset_index(drop=True, inplace=True)

# Load the trained model
model = SentenceTransformer('../MODEL')
#model = SentenceTransformer('bert-base-uncased')

# Initialize BERT model and tokenizer for embeddings
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")


q_NL_scores = []
NL_NL_scores = []

# Calculate cosine similarity for true and false pairs
for idx, row in df.iterrows():
    # Encode true pair (sparql_wikidata and paraphrased_question)
    query = str(row[QUERY_ROW_NAME])
    question = str(row[QUESTION_ROW_NAME])
    translation = str(row[TRANSLATION_ROW_NAME])

    query_embedding_Q = model.encode(query, convert_to_tensor=True)
    translation_embedding_Q = model.encode(translation, convert_to_tensor=True)

    # Bert embeddings
    question_embedding = bert_embedding(question)
    translation_embedding = bert_embedding(translation)

    # Cosine similarity for q and NL
    q_nl_similarity = util.pytorch_cos_sim(query_embedding_Q, translation_embedding_Q).item()
    q_NL_scores.append(q_nl_similarity)

    # Cosine similarity for NL and NL_gt
    N_NL_gt_similarity = util.pytorch_cos_sim(question_embedding, translation_embedding).item()
    NL_NL_scores.append(N_NL_gt_similarity)

scatter_plot_with_regression(df, q_NL_scores, NL_NL_scores, 'Scatter Plot: bert_q_NL vs bert_NL_NL_gt')

