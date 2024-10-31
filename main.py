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

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Initialize BERT model and tokenizer for embeddings
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")


# Function to call LLaMA3 API
def llama3_generate_translation(api_endpoint, Q, model):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a AI translator for converting sparql queries into normal natural language questions<|eot_id|><|start_header_id|>user<|end_header_id|>
            Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: '{Q}' <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    payload = {
        "model": model,
        "prompt": prompt,
    }
    # print(payload)
    response = requests.post(api_endpoint, json=payload, stream=True)
    # print(response)
    response_text = ""
    for line in response.text.split("\n"):
        try:
            response_text += line.split('response":"')[1].split('","done":')[0]
        except:
            pass
    # print("first response")
    # print(response_text)
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a AI translator for converting sparql queries into normal natural language questions<|eot_id|><|start_header_id|>user<|end_header_id|>
            Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: '{Q}' <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            {response_text}<|eot_id|><|start_header_id|>user<|end_header_id|>
            Reflect on your answer and improve upon it; respond with only the improved question in one sentence only<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    payload = {
        "model": model,
        "prompt": prompt,
    }
    # print(payload)
    response = requests.post(api_endpoint, json=payload, stream=True)
    # print(response)
    response_text = ""
    for line in response.text.split("\n"):
        try:
            response_text += line.split('response":"')[1].split('","done":')[0]
        except:
            pass
    # print("second response")
    print(response_text)
    # print(f"llama response {response_text}")
    return response_text


def llama3_compare_translations(api_endpoint, Q, translations, model):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a AI translator for selecting the best natural language translation of a sparql query.<|eot_id|><|start_header_id|>user<|end_header_id|>
            Select only a single translation from the list of given translations that is the best equivalent of the sparql query. Translations: {translations}, Query: {Q}. Respond with the selected translation only.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    payload = {
        "model": model,
        "prompt": prompt,
    }
    # print(payload)
    response = requests.post(api_endpoint, json=payload, stream=True)
    # print(response)
    response_text = ""
    for line in response.text.split("\n"):
        try:
            response_text += line.split('response":"')[1].split('","done":')[0]
        except:
            pass
    # print(f"llama response {response_text}")
    return response_text


# Function to call OpenAI GPT API
def gpt_generate_translation(openai_api_key, Q):
    openai.api_key = openai_api_key
    prompt = f"Translate the SPARQL query: {Q} into natural language."
    response = openai.Completion.create(
        model="gpt-4", prompt=prompt, max_tokens=150  # Choose appropriate model
    )
    return response.choices[0].text.strip()


def gpt_compare_translations(openai_api_key, Q, translations):
    openai.api_key = openai_api_key
    prompt = f"Translate the SPARQL query: {Q} into natural language."
    response = openai.Completion.create(
        model="gpt-4", prompt=prompt, max_tokens=150  # Choose appropriate model
    )
    return response.choices[0].text.strip()


# Abstraction for translation model
def translate_query_to_nl(model_type, Q, api_endpoint=None, openai_api_key=None):
    if model_type == "llama3":
        return llama3_generate_translation(api_endpoint, Q, model=model_type)
    elif model_type == "gpt":
        return gpt_generate_translation(openai_api_key, Q)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def compare_query_to_nl(
    model_type, Q, translations, api_endpoint=None, openai_api_key=None
):
    if model_type == "llama3":
        return llama3_compare_translations(
            api_endpoint, Q, translations, model=model_type
        )
    elif model_type == "gpt":
        return gpt_compare_translations(openai_api_key, Q, translations)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


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


def evaluate_translation_quality(embeddings):
    """Evaluate translation quality using silhouette score."""
    embeddings_array = np.array(embeddings)
    if embeddings_array.ndim == 3:
        embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)
    elif embeddings_array.ndim != 2:
        raise ValueError("Embeddings must be a 2D array or a list of 2D arrays")

    similarities = cosine_similarity(embeddings_array)
    return np.mean(similarities[np.triu_indices_from(similarities, k=1)])


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
    model_type, Q, T_A, T_R, C, api_endpoint=None, openai_api_key=None, k=3
):
    NL_embeddings = []
    translations = []

    # Translate Query through KG
    Q = map_wikidata_to_natural_language(Q)
    print(Q)

    # Get query embedding
    query_embedding = bert_embedding(Q)

    # Generate k proposals and compute BERT embeddings
    for i in range(k):
        NL = translate_query_to_nl(model_type, Q, api_endpoint, openai_api_key)
        e_i = bert_embedding(NL)
        NL_embeddings.append(e_i)
        translations.append(NL)

    print(f"translations {translations}")

    

    # Calculate intra-cluster distance
    intra_cluster_distance = calculate_intra_cluster_distance(NL_embeddings)
    print(f"Intra-cluster distance: {intra_cluster_distance}")

    # Evaluate translation quality
    quality_score = evaluate_translation_quality(NL_embeddings)
    print(f"Translation quality score: {quality_score}")

    # Determine acceptance based on thresholds
    accept = T_A <= quality_score <= T_R

    # Choose the best translation (you can modify this logic if needed)
    best_translation = translations[
        np.argmax([evaluate_translation_quality([e]) for e in NL_embeddings])
    ]
    
    # Print distances between query and translations
    print_embedding_distances(query_embedding, bert_embedding(best_translation))

    return best_translation, accept, quality_score


def gradient_descent_threshold_optimization(
    model_type,
    C,
    combined_df,
    llama3_api_endpoint,
    openai_api_key,
    initial_T_A=0.1,
    initial_T_R=0.9,
    learning_rate=0.01,
    num_iterations=50,
    k=3,
):
    T_A = initial_T_A
    T_R = initial_T_R

    for iteration in range(num_iterations):
        total_quality_score = 0
        total_samples = 0

        for index, row in combined_df.iterrows():
            Q = row["sparql_wikidata"]
            _, _, quality_score = translate_and_assess(
                model_type,
                Q,
                T_A,
                T_R,
                C,
                api_endpoint=llama3_api_endpoint,
                openai_api_key=openai_api_key,
                k=k,
            )
            total_quality_score += quality_score
            total_samples += 1

        avg_quality_score = total_quality_score / total_samples

        # Update thresholds
        T_A_gradient = -1 if avg_quality_score < T_A else 1
        T_R_gradient = 1 if avg_quality_score > T_R else -1

        T_A = max(0, min(1, T_A + learning_rate * T_A_gradient))
        T_R = max(0, min(1, T_R + learning_rate * T_R_gradient))

        print(
            f"Iteration {iteration + 1}: T_A = {T_A:.4f}, T_R = {T_R:.4f}, Avg Quality Score = {avg_quality_score:.4f}"
        )

    return T_A, T_R


# Example usage
if __name__ == "__main__":
    # Initialize SPARQL query Q\
    load_limit = 20

    lc_quad_dataset = load_dataset("lc_quad", trust_remote_code=True)

    # combine both datasets test and train, we dont need this kind of differentiation
    train_df = lc_quad_dataset["train"].to_pandas()
    test_df = lc_quad_dataset["test"].to_pandas()
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    combined_df = combined_df.iloc[:load_limit]

    Q = " select distinct ?obj where { wd:Q188920 wdt:P2813 ?obj . ?obj wdt:P31 wd:Q1002697 } "

    # Thresholds for acceptance and rejection
    T_A = 0.6  # Placeholder threshold for acceptance
    T_R = 0.92  # Placeholder threshold for rejection

    # LLaMA3 usage
    llama3_api_endpoint = "http://localhost:11434/api/generate"

    # GPT usage
    openai_api_key = "your_openai_api_key"

    # Choose critique model type: "llama3" or "gpt"
    C = "llama3"

    # Choose model type: "llama3" or "gpt"
    model_type = "llama3"  # or "gpt"

    # Optimize thresholds using gradient descent
    import os

    best_scores_file = "best_scores.txt"

    if os.path.exists(best_scores_file):
        with open(best_scores_file, "r") as file:
            best_T_A, best_T_R = map(float, file.read().split(","))
    else:
        best_T_A, best_T_R = gradient_descent_threshold_optimization(
            model_type,
            C,
            combined_df,
            llama3_api_endpoint,
            openai_api_key,
            initial_T_A=0.1,
            initial_T_R=0.9,
            learning_rate=0.01,
            num_iterations=50,
            k=3,
        )
        with open(best_scores_file, "w") as file:
            file.write(f"{best_T_A},{best_T_R}")

    print(f"Optimized Acceptance Threshold (T_A): {best_T_A}")
    print(f"Optimized Rejection Threshold (T_R): {best_T_R}")

    # Use optimized thresholds for translation and assessment
    for index, row in combined_df.iterrows():
        Q = row["sparql_wikidata"]
        print("-" * 50)
        translation, accept, quality_score = translate_and_assess(
            model_type,
            Q,
            best_T_A,
            best_T_R,
            C,
            api_endpoint=llama3_api_endpoint,
            openai_api_key=openai_api_key,
            k=3,
        )

        print(f"Query {index + 1}:")
        print("SPARQL Query:", Q)
        print("Best Translation:", translation)
        print("Accepted:", accept)
        print("Quality Score:", quality_score)
        print(f"Supposed question paraphrased: {row['paraphrased_question']}")
        print(f"Supposed question: {row['question']}")
        print("-" * 50)
