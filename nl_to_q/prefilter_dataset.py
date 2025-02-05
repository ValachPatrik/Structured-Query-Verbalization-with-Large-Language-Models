import pandas as pd
from sentence_transformers import SentenceTransformer, util
from torch.nn import Threshold

# Define constants for column names
QUERY_ROW_NAME = 'instantiated_query'
TRANSLATION_ROW_NAME = 'question'
THRESHOLD = 0.8

# Load the dataset
df = pd.read_csv('../results/gpt4_rows_6000_to_7300/results_classified_with_descriptions.csv')

# (Optional) Restrict your dataframe to the first 100 rows
df = df.iloc[:1000]

# Reset index to ensure it starts from 0
df.reset_index(drop=True, inplace=True)

# Load the trained SentenceTransformer model
model = SentenceTransformer("../embedding_models/MODERNBERT_hard_3_epoch_gpt4_translation_1800")

# Initialize a list to store rows that meet the similarity threshold
selected_rows = []

# Iterate over each row in the dataframe
for idx, row in df.iterrows():
    print(f"Processing row {idx + 1}/{len(df)}")  # More informative progress message

    # Extract necessary fields from the row
    query = str(row[QUERY_ROW_NAME])
    translation = str(row[TRANSLATION_ROW_NAME])

    # Encode embeddings using the trained model
    query_embedding = model.encode(query, convert_to_tensor=True)
    translation_embedding = model.encode(translation, convert_to_tensor=True)

    # Calculate cosine similarity between query and translation embeddings
    q_nl_similarity = util.pytorch_cos_sim(query_embedding, translation_embedding).item()

    # Check if similarity exceeds the threshold
    if q_nl_similarity > THRESHOLD:
        # Optionally, you can add the similarity score to the row
        row_data = row.to_dict()
        row_data['q_nl_similarity'] = q_nl_similarity
        selected_rows.append(row_data)

# Create a new dataframe from the selected rows
df_selected = pd.DataFrame(selected_rows)

print('Length after filtering: ', str(len(selected_rows)))

# Save the filtered dataframe to a new CSV file
output_csv_path = 'filtered_results.csv'
df_selected.to_csv(output_csv_path, index=False)

print(f"Filtered data saved to {output_csv_path}")
