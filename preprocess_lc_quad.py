import pandas as pd
from main import map_wikidata_to_natural_language
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

'''
Code to transform the wikidata queries by replacing wikidata ids by names of the entities/predicates and add as new 
row to the dataset
'''


# Load the CSV file into a pandas DataFrame
input_csv = 'lc_quad.csv'
df = pd.read_csv(input_csv)

df = df.iloc[:7000]

### Parallel Solution ##

num_workers = multiprocessing.cpu_count()
tqdm.pandas(desc="Translating SPARQL queries")

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = list(tqdm(executor.map(map_wikidata_to_natural_language, df['sparql_wikidata']), total=len(df), desc="Processing in parallel"))

df['sparql_wikidata_translated'] = results


### Sequential Solution ###
# Apply the function to the 'sparql_wikidata' column with a progress bar
#tqdm.pandas(desc="Translating SPARQL queries")
#df['sparql_wikidata_translated'] = df['sparql_wikidata'].progress_apply(map_wikidata_to_natural_language)

# Save the updated DataFrame to a new CSV file
output_csv = 'lc_quad_preprocessed.csv'
df.to_csv(output_csv, index=False)
print(f"New CSV file with translated SPARQL queries saved as '{output_csv}'")
