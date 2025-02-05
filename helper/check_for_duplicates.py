import pandas as pd


df = pd.read_csv('../results/gpt4_rows_6000_to_7300/results.csv')



# define parts you want to compare
first_500 = df.iloc[:500]['instantiated_query']
last_300 = df.iloc[-300:]['instantiated_query']

# Step 3: Process the strings by stripping whitespace and converting to lowercase
first_500_processed = first_500.astype(str).str.strip().str.lower()
last_300_processed = last_300.astype(str).str.strip().str.lower()

# Convert the first 500 entries to a set for faster lookup
first_500_set = set(first_500_processed)

# Step 4: Find how many entries in the last 300 are also in the first 500
common_entries = last_300_processed.isin(first_500_set)
num_common = common_entries.sum()

# Optional: If you want to see which entries are common
common_queries = last_300_processed[common_entries].unique()

print(f"Number of common entries: {num_common}")

# If you want to see the common queries, uncomment the following line:
# print("Common queries:", common_queries)
