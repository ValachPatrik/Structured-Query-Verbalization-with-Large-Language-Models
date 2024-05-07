from datasets import load_dataset
import pandas as pd
from wikidata.client import Client
import re
import os
from typing import *

def find_translatable_parts(sparql_query: str) -> List[str]:
    """
    This function takes a SPARQL query as input and returns a list of all the entities and properties that are translatable.

    Parameters:
    sparql_query (str): The SPARQL query to be processed

    Returns:
    List[str]: A list of all the entities and properties that are translatable in the SPARQL query
    """
    entity_pattern = re.compile(r'wd:Q\d+')
    property_pattern = re.compile(r'wdt:P\d+')

    entity_matches = entity_pattern.findall(sparql_query)
    property_matches = property_pattern.findall(sparql_query)

    return entity_matches + property_matches

def map_wikidata_to_natural_language(sparql_query:str) -> str:
    """
    This function takes a SPARQL query as input and returns a modified SPARQL query where all the entities and properties that are translatable are replaced with their corresponding natural language labels.

    Parameters:
    sparql_query (str): The SPARQL query to be processed.

    Returns:
    str: A modified SPARQL query where all the entities and properties that are translatable are replaced with their corresponding natural language labels.

    Raises:
    Exception: If an error occurs while retrieving the label of an entity or property and leaves the query as is.
    """
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
    

def load_lc() -> pd.DataFrame:
    """
    This function loads the LC quad dataset. If the dataset is already locally available, it loads it from the local storage.
    Otherwise, it fetches the data from the network.
    
    Deletes rows with failed mappings!!!

    Parameters:
    None

    Returns:
    pd.DataFrame: A pandas DataFrame containing the LC quad dataset.
    """
    if os.path.exists("lc_quad_translated.csv"):
        # Load locally if data is already locally ready
        combined_df = pd.read_csv("lc_quad_translated.csv")
        print("Loaded from local")
    else:
        # Load data from net
        lc_quad_dataset = load_dataset("lc_quad")

        # combine both datasets test and train, we dont need this kind of differentiation
        train_df = lc_quad_dataset["train"].to_pandas()
        test_df = lc_quad_dataset["test"].to_pandas()
        combined_df = pd.concat([train_df, test_df], ignore_index=True)

        # Apply KG to map from ambiguous descriptions to NL
        combined_df["wikidata_translated"] = combined_df["sparql_wikidata"].map(map_wikidata_to_natural_language)
        
        # Delete failed rows
        combined_df = combined_df[combined_df["wikidata_translated"] != combined_df["sparql_wikidata"]]
        
        # Save DF for future use to save resources
        combined_df.to_csv("lc_quad_translated.csv", index=False)
        
        print("Loaded from net")

    return combined_df

combined_df = load_lc()
