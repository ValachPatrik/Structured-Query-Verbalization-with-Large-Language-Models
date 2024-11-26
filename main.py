from helper_functions import *

import requests
import openai
import numpy as np
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform

import warnings
from dotenv import load_dotenv
from openai import OpenAI

import pandas as pd
from sentence_transformers import SentenceTransformer


warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load environment variables from .env file
load_dotenv()

# Initialize BERT model and tokenizer for embeddings
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

q_embedding_model = SentenceTransformer('./all-MiniLM-L6-v2-uninstantiated-wikidata-question')



# Function to call LLaMA3 API
def llama3_generate_translation(llama3_api_endpoint, Q, model, n_shot=1, threshold=1):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a AI translator for converting sparql queries into normal natural language questions<|eot_id|><|start_header_id|>user<|end_header_id|>
            Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: '{Q}' <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    payload = {
        "model": model,
        "prompt": prompt,
    }
    response = requests.post(llama3_api_endpoint, json=payload, stream=True)
    response_text = ""
    for line in response.text.split("\n"):
        try:
            response_text += line.split('response":"')[1].split('","done":')[0]
        except:
            pass


    for _ in range(n_shot):
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are a AI translator for converting sparql queries into normal natural language questions<|eot_id|><|start_header_id|>user<|end_header_id|>
                Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: '{Q}' <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                {response_text}<|eot_id|><|start_header_id|>user<|end_header_id|>
                Reflect on your answer and improve upon it; respond with only the improved question in one sentence only<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        payload = {
            "model": model,
            "prompt": prompt,
        }
        response = requests.post(llama3_api_endpoint, json=payload, stream=True)
        response_text = ""
        for line in response.text.split("\n"):
            try:
                response_text += line.split('response":"')[1].split('","done":')[0]
            except:
                pass
        if threshold == None:
            pass
            break
    print(response_text)
    return response_text


def llama3_compare_translations(llama3_api_endpoint, Q, translations, model):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a AI translator for selecting the best natural language translation of a sparql query.<|eot_id|><|start_header_id|>user<|end_header_id|>
            Select only a single translation from the list of given translations that is the best equivalent of the sparql query. Translations: {translations}, Query: {Q}. Respond with the selected translation only.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    payload = {
        "model": model,
        "prompt": prompt,
    }
    response = requests.post(llama3_api_endpoint, json=payload, stream=True)
    response_text = ""
    for line in response.text.split("\n"):
        try:
            response_text += line.split('response":"')[1].split('","done":')[0]
        except:
            pass
    # print(f"llama response {response_text}")
    return response_text


# Function to call OpenAI GPT API
def gpt_generate_translation(openai_api_key, Q, n_shot=0, threshold=1):
    client = OpenAI(api_key=openai_api_key)
    prompt = f"""You are a AI translator for converting sparql queries into normal natural language questions.
            Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: '{Q}'"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    response_text = response.choices[0].message.content

    for _ in range(n_shot):
        prompt = f"""You are a AI translator for converting sparql queries into normal natural language questions.
                Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: '{Q}'
                {response_text}
                Reflect on your answer and improve upon it; respond with only the improved question in one sentence only."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        response_text = response.choices[0].message.content
        if threshold == None:
            pass
            break
    return response_text


def gpt_compare_translations(openai_api_key, Q, translations):
    client = OpenAI(api_key=openai_api_key)
    prompt = f"""You are a AI translator for selecting the best natural language translation of a sparql query.
            Select only a single translation from the list of given translations that is the best equivalent of the sparql query. Translations: {translations}, Query: {Q}. Respond with the selected translation only."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].text.strip()


# Abstraction for translation model
def translate_query_to_nl(model_type_T, Q, llama3_api_endpoint=None, openai_api_key=None, n_shot=1, threshold=1):
    if model_type_T == "llama3":
        return llama3_generate_translation(llama3_api_endpoint, Q, model=model_type_T, n_shot=n_shot, threshold=threshold)
    elif model_type_T == "gpt":
        return gpt_generate_translation(openai_api_key, Q)
    else:
        raise ValueError(f"Unknown model_type: {model_type_T}")


def compare_query_to_nl(
    model_type_C, Q, translations, llama3_api_endpoint=None, openai_api_key=None
):
    if model_type_C == "llama3":
        return llama3_compare_translations(
            llama3_api_endpoint, Q, translations, model=model_type_C
        )
    elif model_type_C == "gpt":
        return gpt_compare_translations(openai_api_key, Q, translations)
    else:
        raise ValueError(f"Unknown model_type: {model_type_C}")


# Compute BERT embeddings for the text
def bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


def calculate_intra_cluster_distance(embeddings):
    """Calculate the average intra-cluster distance."""
    embeddings_array = np.array(embeddings)
    if embeddings_array.ndim == 3:
        # If we have a 3D array (list of 2D embeddings), we need to reshape it
        embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)
    elif embeddings_array.ndim != 2:
        raise ValueError("Embeddings must be a 2D array or a list of 2D arrays")

    distances = pdist(embeddings_array)
    return np.mean(distances)


def evaluate_distance_filtering_method(embeddings, filtering_method_distance='silhouette'):
    """Evaluate translation quality using specified method."""
    embeddings_array = np.array(embeddings)
    if embeddings_array.ndim == 3:
        embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)
    elif embeddings_array.ndim != 2:
        raise ValueError("Embeddings must be a 2D array or a list of 2D arrays")

    if filtering_method_distance == 'silhouette':
        similarities = cosine_similarity(embeddings_array)
        return np.mean(similarities[np.triu_indices_from(similarities, k=1)])
    elif filtering_method_distance == 'intra_cluster':
        distances = pdist(embeddings_array)
        return np.mean(distances)
    else:
        raise ValueError(f"Unknown evaluation method: {filtering_method_distance}")


def get_model_api(model_type, llama3_api_endpoint=None, openai_api_key=None):
    """Get the appropriate model API configuration."""
    if model_type == "llama3":
        if not llama3_api_endpoint:
            raise ValueError("API endpoint required for llama3")
        return {"api_endpoint": llama3_api_endpoint, "model": model_type}
    elif model_type == "gpt":
        if not openai_api_key:
            raise ValueError("API key required for GPT")
        return {"openai_api_key": openai_api_key}
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def print_embedding_distances(query_embedding, nl_embeddings):
    """Print cosine distances between query embedding and natural language embeddings."""
    print("\nEmbedding Distances (Cosine):")
    for i, nl_embedding in enumerate(nl_embeddings):
        distance = (
            1
            - cosine_similarity(
                query_embedding.reshape(1, -1), nl_embedding.reshape(1, -1)
            )[0][0]
        )
        print(f"Translation {i+1}: {distance:.4f}")

def translate_and_assess(
    model_type_T,
    Q,
    T_A,
    model_type_C,
    filtering_method_distance='silhouette',
    filter_best_translation="distance",
    llama3_api_endpoint=None,
    openai_api_key=None,
    k=3,
    n_shot=1,
    threshold=1,
    NL_gt=None,
    Q_raw=None
):
    NL_embeddings = []
    translations = []
    NL_embeddings_q_model = []

    # Get model API configuration
    model_config = get_model_api(model_type_T, llama3_api_endpoint, openai_api_key)

    # Translate Query through KG
    #Q = map_wikidata_to_natural_language(Q)
    print(Q)

    # Get query embedding
    #query_embedding = bert_embedding(Q)
    if Q_EMBEDDING_MODE == 'instantiated':
        query_embedding = q_embedding_model.encode(Q, convert_to_tensor=False)
    else:
        query_embedding = q_embedding_model.encode(Q_raw, convert_to_tensor=False)

    nl_gt_embedding = bert_embedding(NL_gt)

    # Generate k proposals and compute BERT embeddings
    for i in range(k):
        NL = translate_query_to_nl(model_type_T, Q, **model_config, n_shot=n_shot, threshold=threshold)
        e_i = bert_embedding(NL)
        e_nl_i = q_embedding_model.encode(NL, convert_to_tensor=False)
        NL_embeddings_q_model.append(e_nl_i)
        NL_embeddings.append(e_i)
        translations.append(NL)

    print(f"translations {translations}")

    

    if filter_best_translation == "distance":
        quality_score = evaluate_distance_filtering_method(NL_embeddings, filtering_method_distance=filtering_method_distance)
        print(f"Translation quality score ({filtering_method_distance}): {quality_score}")
        # Determine acceptance based on thresholds
        accept = quality_score >= T_A

        # Choose the best translation
        best_translation_index = np.argmax([evaluate_distance_filtering_method([e], filtering_method_distance=filtering_method_distance) for e in NL_embeddings])
        best_translation = translations[best_translation_index]
        best_translation_embedding = NL_embeddings[best_translation]
    elif filter_best_translation == 'q_nl_score':
        intra_cluster_distance = None
        quality_score = None
        accept = None
        # todo pick the one with highest q_nl_score
        best_translation = translations[0]
        best_translation_embedding = NL_embeddings[0]
        best_translation_embedding_q_model = NL_embeddings_q_model[0]
    elif filter_best_translation == "model_comparison":
        # here we need to get the index of best translation
        best_translation = compare_query_to_nl(model_type_C, Q, translations, **model_config)
        quality_score = 1



    bert_q_nl_score = cosine_similarity(query_embedding.reshape(1, -1), best_translation_embedding_q_model.reshape(1, -1))[0][0]
    bert_nl_nl_gt_score = cosine_similarity(best_translation_embedding.reshape(1, -1), nl_gt_embedding.reshape(1, -1))[0][0]


    return best_translation, accept, quality_score, intra_cluster_distance, bert_q_nl_score, bert_nl_nl_gt_score


def gradient_descent_threshold_optimization(
    model_type_T,
    model_type_C,
    combined_df,
    llama3_api_endpoint,
    openai_api_key,
    initial_T_A=0.1,
    learning_rate=0.01,
    num_iterations=50,
    k=3,
):
    T_A = initial_T_A

    for iteration in range(num_iterations):
        total_quality_score = 0
        total_samples = 0

        for _ , row in combined_df.iterrows():
            Q = row["sparql_wikidata"]
            _, _, quality_score = translate_and_assess(
                model_type_T,
                Q,
                T_A,
                model_type_C,
                llama3_api_endpoint=llama3_api_endpoint,
                openai_api_key=openai_api_key,
                k=k,
            )
            total_quality_score += quality_score
            total_samples += 1

        avg_quality_score = total_quality_score / total_samples

        # Update thresholds
        T_A_gradient = -1 if avg_quality_score < T_A else 1

        T_A = max(0, min(1, T_A + learning_rate * T_A_gradient))

        print(
            f"Iteration {iteration + 1}: T_A = {T_A:.4f}, Avg Quality Score = {avg_quality_score:.4f}"
        )

    return T_A


def load_or_optimize_threshold(
    acceptance_threshold_file,
    model_type_T,
    model_type_C,
    combined_df,
    llama3_api_endpoint,
    openai_api_key,
    initial_T_A=0.1,
    learning_rate=0.01,
    num_iterations=50,
    k=3
):
    """Load the best threshold from file or optimize it using gradient descent."""
    if os.path.exists(acceptance_threshold_file):
        with open(acceptance_threshold_file, "r") as file:
            best_T_A = float(file.read())
    else:
        # Determine threshold using gradient descent
        best_T_A = gradient_descent_threshold_optimization(
            model_type_T,
            model_type_C,
            combined_df,
            llama3_api_endpoint,
            openai_api_key,
            initial_T_A=initial_T_A,
            learning_rate=learning_rate,
            num_iterations=num_iterations,
            k=k,
        )
        with open(acceptance_threshold_file, "w") as file:
            file.write(f"{best_T_A}")

    print(f"Optimized Acceptance Threshold (T_A): {best_T_A}")

    return best_T_A


def load_and_combine_dataset(df_load_limit):
    """
    Load and combine the LC-QuAD dataset from train and test splits.

    Args:
        load_limit (int): Maximum number of rows to load from the dataset

    Returns:
        pandas.DataFrame: Combined dataset limited to specified number of rows
    """
    lc_quad_dataset = load_dataset("lc_quad", trust_remote_code=True)

    # combine both datasets test and train
    train_df = lc_quad_dataset["train"].to_pandas()
    test_df = lc_quad_dataset["test"].to_pandas()
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    return combined_df.iloc[:df_load_limit]

def evaluate_translations(combined_df, model_type_T, best_T_A, model_type_C, llama3_api_endpoint, openai_api_key):
    """
    Evaluate translations for each query in the dataset.

    Args:
        combined_df (pandas.DataFrame): Dataset containing queries to translate
        model_type_T (str): Type of translation model to use
        best_T_A (float): Acceptance threshold
        model_type_C (str): Type of comparison model to use
        llama3_api_endpoint (str): API endpoint for LLaMA3 model
        openai_api_key (str): OpenAI API key
    """
    for index, row in combined_df.iterrows():
        Q = row["sparql_wikidata_translated"]
        print("-" * 50)
        translation, accept, quality_score = translate_and_assess(
            model_type_T,
            Q,
            best_T_A,
            model_type_C,
            filter_best_translation="distance",
            filtering_method_distance="silhouette",
            llama3_api_endpoint=llama3_api_endpoint,
            openai_api_key=openai_api_key,
            k=k,
            NL_gt=row["paraphrased_question"],
            Q_raw=row["sparql_wikidata"],
            results_df
        )

        print(f"Query {index + 1}:")
        print("SPARQL Query:", Q)
        print("Best Translation:", translation)
        print("Accepted:", accept)
        print("Quality Score:", quality_score)
        print(f"Supposed question paraphrased: {row['paraphrased_question']}")
        print(f"Supposed question: {row['question']}")
        print("-" * 50)


        # Store the new row in a dictionary
        new_row = {
            "sparql_wikidata": Q,
            "paraphrased_question": row["paraphrased_question"],
            "question": row["question"],
            "best_translation": translation,
            "bert_q_NL": bert_q_nl_score,
            "bert_NL_NL_gt": bert_nl_nl_gt_score,
            "intra_cluster_distance": intra_cluster_distance
        }

        # Convert to DataFrame and concatenate
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)



        # Saving results_df to disk please
        results_df.to_csv('translation_results.csv')

# Example usage
if __name__ == "__main__":

    ### Parameters ###
    # Initialize SPARQL query Q\
    load_limit = 300

    DATASET_NAME = "lc_quad_preprocessed.csv"
    # whether the q-embedding model works with queries that have wikidata codes replaced (instantiated) by label or not
    Q_EMBEDDING_MODE = 'uninstantiated'
    # LLaMA3 usage
    llama3_api_endpoint = "http://localhost:11434/api/generate"
    # GPT usage
    openai_api_key = "your_openai_api_key"

    # Choose critique model type: "llama3" or "gpt"
    C = "llama3"

    # Choose model type: "llama3" or "gpt" for translation
    # and critique
    model_type_C = "gpt"
    model_type_T = "gpt"  # "llama3" or "gpt"

    # How many answers to sample per query
    k = 1
    ###

    # Check if the combined dataset is saved already
    if os.path.exists(DATASET_NAME):
        # Load from the saved CSV file
        combined_df = pd.read_csv(DATASET_NAME)
        print("Loaded combined dataset from saved file.")
    else:
        # Load the dataset from the source
        lc_quad_dataset = load_dataset("lc_quad", trust_remote_code=True)

        # Combine both datasets: train and test
        train_df = lc_quad_dataset["train"].to_pandas()
        test_df = lc_quad_dataset["test"].to_pandas()
        combined_df = pd.concat([train_df, test_df], ignore_index=True)

        # Save the combined dataframe to a CSV file for future use
        combined_df.to_csv(DATASET_NAME, index=False)
        print("Combined dataset saved to file.")


    combined_df = combined_df.iloc[:load_limit]

    Q = " select distinct ?obj where { wd:Q188920 wdt:P2813 ?obj . ?obj wdt:P31 wd:Q1002697 } "

    # Thresholds for acceptance and rejection
    T_A = 0.6  # Placeholder threshold for acceptance
    best_T_A = T_A

    # Dataframe to store the results
    results_df = pd.DataFrame(columns=["sparql_wikidata", "paraphrased_question", "question", "best_translation", "bert_q_NL", "bert_NL_NL_gt", "intra_cluster_distance"])


    # Evaluate translations
    evaluate_translations(
        combined_df,
        model_type_T,
        best_T_A,
        model_type_C,
        llama3_api_endpoint,
        openai_api_key
    )

