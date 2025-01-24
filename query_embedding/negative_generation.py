import os
import time
from openai import OpenAI
import openai
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from scipy.stats import variation

PROMPT = (
    """Your task is to change given questions in such a way that they have a different meaning (i.e. their answer will be different) but it still sounds similar (uses mostly the same subjects, verbs)
    There are 2 level of variations: 
    slight - here it is quite difficult to spot the difference
    significant - here the change is more prominent and easier to spot
    Please only respond with the answer( see example below), and nothing else
    ITS OF UTMOST IMPORTANT THAT THE ANSWERS TO THE QUESTIONS WOULD BE DIFFERENT (OR PUT ANOTHER WAY, THE SQL QUERIES REPRESENTING THE QUESTIONS WOULD BE DIFFERENT)
    Example:

    Question: Who is the child of Ranavalona I's husband?
    Variation Level: slight
    Variation: Who is the father of Ranavalona I's husband?

    Question: What award did Danila Kozlovsky receive in 2017?
    Variation Level: significant
    Variation: Did Danila Kozlowsky give more then 10 awards to somebody in 2017 ? 
    
    Question: Who replaced Isabel Martinez de Peron as the President of Argentina?
    Variation Level: significant
    Variation: Did Isabel Martinez de Peron and the President of Argentina meet recently ? 

    Question: {question} ?
    Variation Level: {variation_level}
    Variation: """
)



# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")  # Ensure you have this in your .env

# Configure OpenAI
openai.api_key = openai_api_key

# Configure Gemini
genai.configure(api_key=gemini_api_key)

# Flag to choose between GPT and Gemini
USE_GEMINI = True  # Set to False to use OpenAI GPT

# Function to generate variations using OpenAI GPT
def generate_variations_gpt(question):
    client = OpenAI(api_key=openai_api_key)
    prompt_slight = PROMPT.format(question=question, variation_level='slight')
    prompt_significant = PROMPT.format(question=question, variation_level='significant')

    try:
    #     response = client.chat.completions.create(
    #         model="gpt-4o",
    #         messages=[
    #             {"role": "user", "content": prompt_slight}
    #         ]
    #     )
    #     slight = response.choices[0].message.content
        slight = ''

        # Generate significant variation
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt_slight}
            ]
        )
        significant = response.choices[0].message.content

        return slight, significant

    except Exception as e:
        print(f"Error generating variations for question: {question}\nError: {e}")
        return "ERROR", "ERROR"

# Function to generate variations using Gemini
def generate_variations_gemini(question):
    prompt_slight = PROMPT.format(question=question, variation_level='slight')
    prompt_significant = PROMPT.format(question=question, variation_level='significant')

    model = genai.GenerativeModel("gemini-2.0-flash-exp")

    # Generate slight variation
    #response_slight = model.generate_content(prompt_slight)
    #slight = response_slight.text
    slight = ""

    # Generate significant variation
    response_significant = model.generate_content(prompt_significant)
    significant = response_significant.text

    return slight, significant


# Function to process the CSV
def process_csv(input_file, output_file, use_gemini=False, sleep_time=1):
    # Read the CSV file
    df = pd.read_csv(input_file)

    df = df.iloc[:] # TODO MAKE THE SKIP HERE DYNAMIC


    QUESTION_COLUMN = 'question'

    # Ensure the 'question' column exists
    if QUESTION_COLUMN not in df.columns:
        raise ValueError("The input CSV must contain a 'question' column.")

    # Initialize new columns
    df['negative_slight'] = ""
    df['negative_significant'] = ""

    # Iterate over each row with progress
    i = 0
    for index, row in df.iterrows():
        question = row[QUESTION_COLUMN]
        print(f"Processing row {index + 1}/{len(df)}: {question}")

        try:
            if use_gemini:
                slight, significant = generate_variations_gemini(question)
            else:
                slight, significant = generate_variations_gpt(question)
        except:
            print('ERROR TRYING TO GET VARIATION')
            time.sleep(30)

        print(f"Variation: {significant}")

        # Assign the variations to the DataFrame
        #df.at[index, 'negative_slight'] = slight
        df.at[index, 'negative_significant'] = significant

        # Sleep to avoid overloading the API
        i += 1
        time.sleep(7)
        #if i >295:
        #    break

        # Save the updated CSV
        df.to_csv(output_file, index=False)
        print(f"Processed CSV saved to {output_file}")

# Input and output file paths
input_csv = "../results/1737620314/results.csv"       # Replace with your input file path
output_csv = "output.csv"     # Replace with your desired output file path

# Process the CSV
process_csv(input_csv, output_csv, use_gemini=USE_GEMINI, sleep_time=1)
