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
def scatter_plot_with_regression(df, x_values, y_values, title):
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)
    line = slope * np.array(x_values) + intercept

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, label='Data points', color='blue', alpha=0.6)
    plt.plot(x_values, line, color='red', label=f'Regression line (r={r_value:.2f})')

    # Plot settings
    plt.xlabel("q_NL_similarity", fontsize=14)
    plt.ylabel("NL_NL_gt_similarity", fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()

QUERY_ROW_NAME = 'instantiated_query'
QUESTION_ROW_NAME = 'original_question'
TRANSLATION_ROW_NAME = 'best_translation_Q'

# Load the dataset
df = pd.read_csv('../results/results_classified.csv')

# (Optional) restrict your dataframe, for example:
df = df.iloc[:120]

# Reset index to ensure it starts from 0
df.reset_index(drop=True, inplace=True)

# Load the trained model
model = SentenceTransformer("../.embedding_models/instantiated_hard_negative_1_epoch")
# If you want to use BERT base model:
# model = SentenceTransformer('bert-base-uncased')

# Initialize BERT model and tokenizer for embeddings (if you need them)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

q_NL_scores = []
NL_NL_scores = []

# Arrays to store q_NL similarity for correct and incorrect translations
q_NL_correct = []
q_NL_wrong = []

for idx, row in df.iterrows():
    print(idx)
    query = str(row[QUERY_ROW_NAME])
    question = str(row[QUESTION_ROW_NAME])
    translation = str(row[TRANSLATION_ROW_NAME])

    # Encode embeddings
    query_embedding_Q = model.encode(query, convert_to_tensor=True)
    translation_embedding_Q = model.encode(translation, convert_to_tensor=True)
    question_embedding = model.encode(question, convert_to_tensor=True)

    question_embedding = bert_embedding(question)
    translation_embedding = bert_embedding(translation)

    # Calculate cosine similarities
    q_nl_similarity = util.pytorch_cos_sim(query_embedding_Q, translation_embedding_Q).item()
    N_NL_gt_similarity = util.pytorch_cos_sim(question_embedding, translation_embedding).item()

    q_NL_scores.append(q_nl_similarity)
    NL_NL_scores.append(N_NL_gt_similarity)

    # If translation_correct is defined (not NaN), separate them into correct/incorrect
    if pd.notna(row.get('translation_correct')):
        if row['translation_correct'] == 1:
            q_NL_correct.append(q_nl_similarity)
        elif row['translation_correct'] == 0:
            q_NL_wrong.append(q_nl_similarity)

# printing accuracy:
acc = len(q_NL_correct)/ (len(q_NL_correct) + len(q_NL_wrong))
print('Manual determined translation accuracy: ', str(acc))

# Scatter plot
scatter_plot_with_regression(df, q_NL_scores, NL_NL_scores,
                             'Scatter Plot: bert_q_NL vs bert_NL_NL_gt')

# Now create a histogram only for rows where translation_correct is present
plt.figure(figsize=(10, 6))

bins = np.linspace(0, 1, 50)  # Choose bins from 0 to 1 for cosine similarity

plt.hist(q_NL_correct, bins=bins, color='green', alpha=0.6,
         label='Correct Translation (1)')
plt.hist(q_NL_wrong, bins=bins, color='red', alpha=0.6,
         label='Incorrect Translation (0)')

plt.title('Histogram of q_NL Similarities by Translation Correctness')
plt.xlabel('q_NL Similarity')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.show()
