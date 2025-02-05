import pandas as pd

df = pd.read_csv('../results_nl_to_q/gpt-4o-mini-synthetic-clean/results.csv')

df['sparql_correct'] = pd.to_numeric(df['sparql_correct'], errors='coerce')

existing_entries = df['sparql_correct'].dropna()

num_correct = (existing_entries == 1).sum()

total_entries = existing_entries.shape[0]
accuracy = (num_correct / total_entries) * 100

print(f"Number of correct entries (1s): {num_correct}")
print(f"Total existing entries: {total_entries}")
print(f"Accuracy: {accuracy:.2f}%")
