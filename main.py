from datasets import load_dataset
import pandas as pd
from wikidata.client import Client
import re
import os

def find_translatable_parts(sparql_query):
    entity_pattern = re.compile(r'wd:Q\d+')
    property_pattern = re.compile(r'wdt:P\d+')

    entity_matches = entity_pattern.findall(sparql_query)
    property_matches = property_pattern.findall(sparql_query)

    return entity_matches + property_matches

def map_wikidata_to_natural_language(sparql_query):
    client = Client()
    
    translatable_parts = find_translatable_parts(sparql_query)
    
    for i in translatable_parts:
        try:
            entity = client.get(i.split(":")[1], load=True)
            sparql_query = sparql_query.replace(i, str(entity.label))
        except Exception as e:
            print(e, end=" ")
            print(i)
    return sparql_query
    

def load_lc():
    if os.path.exists("lc_quad_translated.csv"):
        
        combined_df = pd.read_csv("lc_quad_translated.csv")
        print("Loaded from local")
    else:
        lc_quad_dataset = load_dataset("lc_quad")

        train_df = lc_quad_dataset["train"].to_pandas()
        test_df = lc_quad_dataset["test"].to_pandas()

        combined_df = pd.concat([train_df, test_df], ignore_index=True)

        combined_df["wikidata_translated"] = combined_df["sparql_wikidata"].map(map_wikidata_to_natural_language)
        
        combined_df.to_csv("lc_quad_translated.csv", index=False)
        
        print("Loaded from net")

    return combined_df

combined_df = load_lc()
