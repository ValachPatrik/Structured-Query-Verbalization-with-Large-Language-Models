import pandas as pd
from wiki_translations import map_wikidata_to_natural_language_description
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from datasets import load_dataset
import os

'''
Code to transform the wikidata queries by replacing wikidata ids by names of the entities/predicates and add as new 
row to the dataset
'''
def preprocess_lc_quad(n_queries):
    if not os.path.exists('../data/v2/lc_quad.csv'):
        lc_quad_dataset = load_dataset("lc_quad", trust_remote_code=True)
        df = pd.concat([lc_quad_dataset["train"].to_pandas(), lc_quad_dataset["test"].to_pandas()], ignore_index=True)
        df.to_csv('data/v2/lc_quad.csv', index=False)
        print("LC Quad dataset saved as 'lc_quad.csv'")
    else:
        input_csv = '../data/v2/lc_quad.csv'
        df = pd.read_csv(input_csv)
        print('Loaded from disk')

    df = df.iloc[:n_queries]

    ### Parallel Solution ##

    num_workers = multiprocessing.cpu_count()
    tqdm.pandas(desc="Translating SPARQL queries")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(map_wikidata_to_natural_language_description, df['sparql_wikidata']), total=len(df), desc="Processing in parallel"))

    # Unpack the results into two separate lists
    translated_queries, descriptions = zip(*results)

    df['sparql_wikidata_translated'] = translated_queries
    df['descriptions'] = descriptions

    # Save the updated DataFrame to a new CSV file
    output_csv = '../data/v2/lc_quad_preprocessed.csv'
    df.to_csv(output_csv, index=False)
    print(f"New CSV file with translated SPARQL queries and descriptions saved as '{output_csv}'")


if __name__ == '__main__':
    preprocess_lc_quad(10000)
