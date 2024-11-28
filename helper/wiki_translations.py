import re
from typing import *
from wikidata.client import Client
from concurrent.futures import ThreadPoolExecutor, as_completed

def find_translatable_parts(sparql_query: str) -> List[str]:
    entity_pattern = re.compile(r"wd:Q\d+")
    property_pattern = re.compile(r"wdt:P\d+")
    property_pattern2 = re.compile(r"p:P\d+")
    property_pattern3 = re.compile(r"ps:P\d+")
    property_pattern4 = re.compile(r"pq:P\d+")

    # property_pattern = re.compile(r"(wdt|p|ps|pq):P\d+")

    entity_matches = entity_pattern.findall(sparql_query)
    property_matches = property_pattern.findall(sparql_query)
    property_matches2 = property_pattern2.findall(sparql_query)
    property_matches3 = property_pattern3.findall(sparql_query)
    property_matches4 = property_pattern4.findall(sparql_query)

    return (
        entity_matches
        + property_matches
        + property_matches2
        + property_matches3
        + property_matches4
    )


def translate_part(part: str, client: Client) -> str:
    try:
        entity = client.get(part.split(":")[1], load=True)
        return part, str(entity.label)
    except Exception as e:
        print(e, end=" ")
        print(part)
        return part, None


def translate_part_description(part: str, client: Client) -> str:
    try:
        entity = client.get(part.split(":")[1], load=True)
        return part, str(entity.description)
    except Exception as e:
        print(e, end=" ")
        print(part)
        return part, None


def map_wikidata_to_natural_language(sparql_query: str) -> str:
    client = Client()

    translatable_parts = find_translatable_parts(sparql_query)
    translated_parts = {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_part = {
            executor.submit(translate_part, part, client): part
            for part in translatable_parts
        }

        for future in as_completed(future_to_part):
            part = future_to_part[future]
            translated_part = future.result()
            if translated_part[1] is not None:
                translated_parts[translated_part[0]] = f"[{translated_part[1]}]"
            else:
                print("None part!")
    
    for part, translation in translated_parts.items():
        sparql_query = sparql_query.replace(part, translation)
        
    if None in translated_parts.values():
        return None
    
    return sparql_query



def map_wikidata_to_natural_language_description(
    sparql_query: str,
) -> Tuple[str, List[str]]:
    client = Client()

    translatable_parts = find_translatable_parts(sparql_query)
    translated_parts = {}
    descriptions = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks for translation
        future_to_part_translation = {
            executor.submit(translate_part, part, client): part
            for part in translatable_parts
        }
        # Submit tasks for description
        future_to_part_description = {
            executor.submit(translate_part_description, part, client): part
            for part in translatable_parts
        }

        # Process translation results
        for future in as_completed(future_to_part_translation):
            part = future_to_part_translation[future]
            translated_part = future.result()
            if translated_part[1] is not None:
                translated_parts[translated_part[0]] = f"[{translated_part[1]}]"

        # Process description results
        for future in as_completed(future_to_part_description):
            part = future_to_part_description[future]
            translated_description = future.result()
            if translated_description[1] is not None:
                # Add "<label> is <description>" to descriptions
                label = translated_parts.get(part, part)
                descriptions.append(f"{label} is {translated_description[1]}")

    # Replace parts in the query with their translations
    for part, translation in translated_parts.items():
        sparql_query = sparql_query.replace(part, translation)

    return sparql_query, descriptions