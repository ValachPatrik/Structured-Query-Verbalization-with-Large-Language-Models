import pandas as pd
import re


# Function to normalize whitespace
def normalize_whitespace(s):
    return re.sub(r'\s+', ' ', s).strip()


# Define file paths (update these paths if your files are located elsewhere)
lc_quad_file = '../data/v2/lc_quad_preprocessed.csv'
results_classified_file = '../results/gpt4_rows_6000_to_7300/results.csv'
output_file = 'results_classified_with_descriptions.csv'

# Step 1: Read the lc_quad_preprocessed.csv and select rows 6000 to 7300
try:
    lc_df = pd.read_csv(lc_quad_file)
    selected_lc_df = lc_df.iloc[6000:7300].copy()
    print(f"Selected {len(selected_lc_df)} rows from {lc_quad_file}.")
except FileNotFoundError:
    print(f"Error: The file {lc_quad_file} was not found.")
    exit(1)
except Exception as e:
    print(f"An error occurred while reading {lc_quad_file}: {e}")
    exit(1)

# Step 2: Read the results_classified.csv
try:
    results_df = pd.read_csv(results_classified_file)
    print(f"Loaded {len(results_df)} rows from {results_classified_file}.")
except FileNotFoundError:
    print(f"Error: The file {results_classified_file} was not found.")
    exit(1)
except Exception as e:
    print(f"An error occurred while reading {results_classified_file}: {e}")
    exit(1)

# Step 3: Normalize whitespace and lowercase the relevant columns
selected_lc_df['sparql_wikidata_translated'] = selected_lc_df['sparql_wikidata_translated'].astype(
    str).str.lower().apply(normalize_whitespace)
results_df['instantiated_query'] = results_df['instantiated_query'].astype(str).str.lower().apply(normalize_whitespace)

# Step 3.5: Check for duplicates in selected_lc_df
duplicate_keys = selected_lc_df['sparql_wikidata_translated'].duplicated().sum()
if duplicate_keys > 0:
    print(
        f"Found {duplicate_keys} duplicate 'sparql_wikidata_translated' entries. Handling duplicates by keeping the first occurrence.")
    # Option 1: Drop duplicates, keeping the first occurrence
    selected_lc_df = selected_lc_df.drop_duplicates(subset='sparql_wikidata_translated', keep='first')

    # Option 2: Alternatively, you can aggregate descriptions if needed
    # selected_lc_df = selected_lc_df.groupby('sparql_wikidata_translated')['descriptions'].agg('; '.join).reset_index()
else:
    print("No duplicate 'sparql_wikidata_translated' entries found.")

# Step 4: Merge the DataFrames
# Perform a left join to keep all rows from results_df and add matching descriptions
merged_df = results_df.merge(
    selected_lc_df[['sparql_wikidata_translated', 'descriptions']],
    how='left',
    left_on='instantiated_query',
    right_on='sparql_wikidata_translated'
)

# Step 5: Verify that all rows have a matching description
missing_descriptions = merged_df['descriptions'].isna().sum()
if missing_descriptions > 0:
    raise ValueError(
        f"Error: {missing_descriptions} rows in {results_classified_file} did not find a matching description.")
else:
    print("All rows have matching descriptions.")

# Step 6: Add the 'descriptions' column to results_df
results_df['descriptions'] = merged_df['descriptions']

# Optional: Drop the 'sparql_wikidata_translated' column if it's no longer needed
# results_df = results_df.drop(columns=['sparql_wikidata_translated'])

# Step 7: Save the updated DataFrame to a new CSV file
try:
    results_df.to_csv(output_file, index=False)
    print(f"Successfully saved the updated data to {output_file}.")
except Exception as e:
    print(f"An error occurred while saving to {output_file}: {e}")
