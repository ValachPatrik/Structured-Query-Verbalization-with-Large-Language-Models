import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from scipy.stats import linregress
from transformers import BertModel, BertTokenizer
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

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
df = pd.read_csv('../results/gemini_few_shot/results_classified_gemini_few_shot.csv')

# (Optional) restrict your dataframe, for example:
df = df.iloc[:49]

# Reset index to ensure it starts from 0
df.reset_index(drop=True, inplace=True)

# Load the trained model
#model = SentenceTransformer("../embedding_models/instantiated_ModernBERT")
model = SentenceTransformer("../MODEL")
# If you want to use BERT base model:
# model = SentenceTransformer('bert-base-uncased')

# Initialize BERT model and tokenizer for embeddings (if you need them)
#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#bert_model = BertModel.from_pretrained("bert-base-uncased")

bert_model = SentenceTransformer("../embedding_models/ModernBERT_untrained")


q_NL_scores = []
NL_NL_scores = []
manual_scores = []

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
    #question_embedding = model.encode(question, convert_to_tensor=True)

    # Encode embeddings with ModernBERT
    question_embedding = bert_model.encode(question, convert_to_tensor=True)
    translation_embedding = bert_model.encode(translation, convert_to_tensor=True)

    # or normal BERT
    #question_embedding = bert_embedding(question)
    #translation_embedding = bert_embedding(translation)

    # Calculate cosine similarities
    q_nl_similarity = util.pytorch_cos_sim(query_embedding_Q, translation_embedding_Q).item()
    N_NL_gt_similarity = util.pytorch_cos_sim(question_embedding, translation_embedding).item()

    q_NL_scores.append(q_nl_similarity)
    NL_NL_scores.append(N_NL_gt_similarity)

    # If translation_correct is defined (not NaN), separate them into correct/incorrect
    if pd.notna(row.get('translation_correct')):
        manual_scores.append(row['translation_correct'])
        if row['translation_correct'] == 1:
            q_NL_correct.append(q_nl_similarity)
        elif row['translation_correct'] == 0:
            q_NL_wrong.append(q_nl_similarity)

# printing accuracy:
try:
    acc = len(q_NL_correct)/ (len(q_NL_correct) + len(q_NL_wrong))
    print('Manual determined translation accuracy: ', str(acc))
except:
    pass


print('Average BERT_NL_GT Score: ', np.mean(NL_NL_scores))
print('Average BERT_NL_Q Score: ', np.mean(q_NL_scores))

corr_matrix = np.corrcoef(q_NL_scores, manual_scores)
correlation = corr_matrix[0, 1]
print("Correlation of manual assessment and q_NL score:", correlation)



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

# Define the threshold for classification
threshold = 0.73  # You can adjust this value based on your requirements

# Initialize lists to store predicted labels and true labels
predicted_labels = []
true_labels = []

# Iterate over the scores and manual labels to create predicted and true label lists
for score, true_label in zip(q_NL_scores, manual_scores):
    predicted_label = 1 if score >= threshold else 0
    predicted_labels.append(predicted_label)
    true_labels.append(true_label)

# Compute evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, zero_division=0)
recall = recall_score(true_labels, predicted_labels, zero_division=0)

print(f"\nClassification Metrics using threshold = {threshold}:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")

# Optional: Display a confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Incorrect', 'Correct'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Optional: Find the optimal threshold using ROC Curve
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(true_labels, q_NL_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
