import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from scipy.stats import linregress
from transformers import BertModel, BertTokenizer
import numpy as np

'''
Given a trained sentence embedding model, this code creates histograms showing BERT scores 
for true and false combinations
'''

def bert_embedding(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Scatter plot function for correct/incorrect with regression lines
def scatter_plot_with_regression(
    x_correct, y_correct,
    x_wrong, y_wrong,
    title
):
    # Regression for correct points
    if len(x_correct) > 1 and len(y_correct) > 1:
        slope_c, intercept_c, r_value_c, p_value_c, std_err_c = linregress(x_correct, y_correct)
        line_c = slope_c * np.array(x_correct) + intercept_c
    else:
        slope_c, intercept_c, r_value_c = 0, 0, 0
        line_c = []

    # Regression for incorrect points
    if len(x_wrong) > 1 and len(y_wrong) > 1:
        slope_w, intercept_w, r_value_w, p_value_w, std_err_w = linregress(x_wrong, y_wrong)
        line_w = slope_w * np.array(x_wrong) + intercept_w
    else:
        slope_w, intercept_w, r_value_w = 0, 0, 0
        line_w = []

    plt.figure(figsize=(10, 6))

    # Plot correct points
    plt.scatter(x_correct, y_correct, color='green', alpha=0.6, label='Correct')
    if len(line_c) > 0:
        plt.plot(x_correct, line_c, color='green',
                 label=f'Correct line (r={r_value_c:.2f})')

    # Plot incorrect points
    plt.scatter(x_wrong, y_wrong, color='red', alpha=0.6, label='Incorrect')
    if len(line_w) > 0:
        plt.plot(x_wrong, line_w, color='red',
                 label=f'Incorrect line (r={r_value_w:.2f})')

    # Plot settings
    plt.xlabel("q_NL_similarity", fontsize=14)
    plt.ylabel("NL_NL_gt_similarity", fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()

QUERY_ROW_NAME = 'instantiated_query'
QUESTION_ROW_NAME = 'original_question'
#TRANSLATION_ROW_NAME = 'best_translation_Q'
CORRECTNESS_COLUMN = 'translation_correct'


CORRECTNESS_COLUMN = 'gt_correct'
TRANSLATION_ROW_NAME = 'original_question'

# Load the dataset
df = pd.read_csv('../results/gpt4_few_shot/results_gpt4_few_shot.csv')

# (Optional) restrict your dataframe, for example:
df = df.iloc[:50]

# Reset index to ensure it starts from 0
df.reset_index(drop=True, inplace=True)

# Load the trained model
model = SentenceTransformer("../embedding_models/instantiated_ModernBERT")

# Initialize BERT model and tokenizer for embeddings (if you need them)
#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#bert_model = BertModel.from_pretrained("bert-base-uncased")

bert_model = SentenceTransformer("../embedding_models/ModernBERT_untrained")


# Arrays to store q_NL similarity and NL_NL_gt similarity for correct and incorrect translations
q_NL_correct = []
NL_NL_correct = []

q_NL_wrong = []
NL_NL_wrong = []

# We also keep track of q_NL_scores and NL_NL_scores if you need the full distribution
q_NL_scores = []
NL_NL_scores = []

for idx, row in df.iterrows():
    print(idx)
    query = str(row[QUERY_ROW_NAME])
    question = str(row[QUESTION_ROW_NAME])
    translation = str(row[TRANSLATION_ROW_NAME])

    # Encode embeddings with SentenceTransformer
    query_embedding_Q = model.encode(query, convert_to_tensor=True)
    translation_embedding_Q = model.encode(translation, convert_to_tensor=True)

    # Encode embeddings with ModernBERT
    question_embedding = bert_model.encode(question, convert_to_tensor=True)
    translation_embedding = bert_model.encode(translation, convert_to_tensor=True)

    # or normal BERT
    #question_embedding = bert_embedding(question)
    #translation_embedding = bert_embedding(translation)

    # Calculate cosine similarities
    q_nl_similarity = util.pytorch_cos_sim(query_embedding_Q, translation_embedding_Q).item()
    NL_NL_gt_similarity = util.pytorch_cos_sim(question_embedding, translation_embedding).item()

    # For completeness, keep track of them globally
    q_NL_scores.append(q_nl_similarity)
    NL_NL_scores.append(NL_NL_gt_similarity)

    # Only gather correctness data if 'translation_correct' is not NaN
    if pd.notna(row.get(CORRECTNESS_COLUMN)):
        # If translation is correct
        if row[CORRECTNESS_COLUMN] == 1:
            q_NL_correct.append(q_nl_similarity)
            NL_NL_correct.append(NL_NL_gt_similarity)
        # If translation is incorrect
        elif row[CORRECTNESS_COLUMN] == 0:
            q_NL_wrong.append(q_nl_similarity)
            NL_NL_wrong.append(NL_NL_gt_similarity)

# Print accuracy if you used manual labeling
n_correct = len(q_NL_correct)
n_wrong = len(q_NL_wrong)
acc = n_correct / (n_correct + n_wrong) if (n_correct + n_wrong) > 0 else None
print('Manual determined translation accuracy: ', str(acc))

# Print average BERT_NL_GT Score (over entire dataframe, if you want)
print('Average BERT_NL_GT Score: ', np.mean(NL_NL_scores))

# Now plot the scatter plot, showing correct points in green, wrong in red
scatter_plot_with_regression(
    x_correct=q_NL_correct,
    y_correct=NL_NL_correct,
    x_wrong=q_NL_wrong,
    y_wrong=NL_NL_wrong,
    title='Scatter Plot: bert_q_NL vs bert_NL_NL_gt (Correct vs Wrong)'
)

# Create a histogram only for rows where translation_correct is present
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
