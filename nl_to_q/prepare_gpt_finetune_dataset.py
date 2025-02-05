import pandas as pd
import json


# Define the input and output file paths
INPUT_CSV_PATH = '../results/gpt4_rows_6000_to_7300/results_classified_with_descriptions.csv'      # Replace with your actual CSV file path
OUTPUT_JSONL_PATH = 'test_data_clean.jsonl'  # Desired output file path

SYSTEM_MESSAGE = f"""
You are an AI translator for converting natural language questions into SPARQL queries. You answer only
    with SPARQL queries, given descriptions of entities/predicates in queries and a Natural Language Question that needs to
    be translated
Here are some examples:

Query: select distinct ?obj where {{ [Delta Air Lines] [house publication] ?obj . ?obj [instance of] [periodical] }}
Natural Language: What periodical literature does Delta Air Lines use as a mouthpiece?

Query: ASK WHERE {{ [Judi Dench] [award received] [Tony Award for Best Direction of a Play] . [Judi Dench] [award received] [Praemium Imperiale] }}
Natural Language: Did Judi Dench receive the Tony Award for Best Direction of a Play and the Praemium Imperiale?

Query with a blank node:
Query: SELECT ?obj WHERE {{ [Angela Merkel] [position held] ?s . ?s [position held] ?obj . ?s [start time] ?x FILTER (contains(YEAR(?x), '1994')) }}
Natural Language: Which position did Angela Merkel hold on November 10, 1994?

Query: select ?ent where {{ ?ent [instance of] [Class IB flammable liquid] . ?ent [lower flammable limit] ?obj }} ORDER BY DESC(?obj) LIMIT 5 
Natural Language: Which 5 Class IB flammable liquids have the highest lower flammable limit?

Query: SELECT (COUNT(?obj) AS ?value ) {{ [St. Peter's Basilica] [architect] ?obj }}
Natural Language: How many architects were responsible for designing St. Peter's Basilica?
"""

USER_PROMPT_TEMPLATE = """
Context information: {description}

Translate the following natural language question into a SPARQL query; respond only with the SPARQL query. Here is the Natural Language question: '{question}'
"""

def create_jsonl(input_csv, output_jsonl, system_message):
    """
    Reads the input CSV and writes the formatted data to a JSONL file.

    Parameters:
    - input_csv (str): Path to the input CSV file.
    - output_jsonl (str): Path to the output JSONL file.
    - system_message (str): The fixed system message.
    """
    try:
        df = pd.read_csv(input_csv)
        df = df.iloc[1000:1300] # restrict to examples for train or test

        # Check if required columns exist
        required_columns = {'instantiated_query', 'question', 'description'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Input CSV is missing columns: {missing}")

        with open(output_jsonl, 'w', encoding='utf-8') as outfile:
            # Iterate over each row in the DataFrame
            for index, row in df.iterrows():
                # Extract values from the row
                description = row['description']
                question = row['question']
                instantiated_query = row['instantiated_query']

                # Construct the user prompt using the template
                user_prompt = USER_PROMPT_TEMPLATE.format(
                    description=description,
                    question=question
                ).strip()  # Remove leading/trailing whitespace

                # Create the message structure
                message = {
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": instantiated_query}
                    ]
                }

                # Write the JSON object as a single line in the JSONL file
                outfile.write(json.dumps(message) + '\n')

        print(f"Successfully created {output_jsonl} with {len(df)} entries.")

    except FileNotFoundError:
        print(f"Error: The file {input_csv} does not exist.")
    except pd.errors.EmptyDataError:
        print("Error: The input CSV file is empty.")
    except pd.errors.ParserError:
        print("Error: The input CSV file is malformed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Execute the function with the specified paths and system message
if __name__ == "__main__":
    create_jsonl(INPUT_CSV_PATH, OUTPUT_JSONL_PATH, SYSTEM_MESSAGE)
