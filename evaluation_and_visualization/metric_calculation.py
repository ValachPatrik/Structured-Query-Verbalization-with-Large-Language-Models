import pandas as pd
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.nist_score import sentence_nist
from nltk.metrics.distance import edit_distance
from rouge import Rouge

# Initialize Rouge
rouge = Rouge()

# Define function to calculate ROUGE-L using the rouge library
def calculate_rougeL(row):
    try:
        # Use best_translation_Q as generated output and original_question as reference
        scores = rouge.get_scores(row['best_translation_Q'], row['original_question'])
        return np.round(scores[0]['rouge-l']['f'], 4)
    except Exception as e:
        raise
        return 0.0

# Load the CSV file
df = pd.read_csv('../results/gpt4_300/results_classified.csv')  # replace 'your_file.csv' with your actual file path

# Prepare lists to collect metric scores
bleu_scores = []
nist_scores = []
rouge_scores = []
levenshtein_scores = []

# Compute metrics for each row
for idx, row in df.iterrows():
    reference = str(row['original_question'])
    hypothesis = str(row['best_translation_Q'])

    # Calculate Sentence BLEU
    try:
        bleu = sentence_bleu(
            [reference.split()],
            hypothesis.split(),
            smoothing_function=SmoothingFunction().method4
        )
    except Exception:
        raise
        bleu = 0.0

    # Calculate Sentence NIST
    try:
        nist = sentence_nist([reference.split()], hypothesis.split(), 4)
    except Exception:
        raise
        nist = 0.0

    # Calculate ROUGE-L using the rouge library
    rouge_score = calculate_rougeL(row)

    # Calculate Levenshtein distance
    levenshtein = edit_distance(reference, hypothesis)

    bleu_scores.append(bleu)
    nist_scores.append(nist)
    rouge_scores.append(rouge_score)
    levenshtein_scores.append(levenshtein)

# Add new columns to the DataFrame
df['BLEU'] = bleu_scores
df['NIST'] = nist_scores
df['ROUGE_L'] = rouge_scores
df['Levenshtein'] = levenshtein_scores

# Print average of each metric across all rows
print("Average BLEU:", df['BLEU'].mean())
print("Average NIST:", df['NIST'].mean())
print("Average ROUGE_L:", df['ROUGE_L'].mean())
print("Average Levenshtein:", df['Levenshtein'].mean())

# Filter rows where translation_correct is not null
df_corr = df[df['translation_correct'].notnull()].copy()

# Convert translation_correct to numeric (assuming it contains boolean-like values)
df_corr['translation_correct'] = df_corr['translation_correct'].astype(int)

# Calculate and print correlation of each metric with translation_correct
for metric in ['BLEU', 'NIST', 'ROUGE_L', 'Levenshtein', 'bert_nl_nl_gt_score_Q']:
    correlation = df_corr[metric].corr(df_corr['translation_correct'])
    print(f"Correlation between {metric} and translation_correct: {correlation}")

df.to_csv('all_metrics.csv')
