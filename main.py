from torch.nn.functional import threshold

from helper.helper_functions import *

import numpy as np
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform

import warnings
from dotenv import load_dotenv

import pandas as pd
from sentence_transformers import SentenceTransformer


warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load environment variables from .env file
load_dotenv()

# Initialize BERT model and tokenizer for embeddings
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

q_embedding_model = SentenceTransformer('./all-MiniLM-L6-v2-uninstantiated-wikidata-question')

# Compute BERT embeddings for the text
def bert_embedding(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def evaluate_distance_method(embeddings, filtering_method_distance='intra_cluster'):
    """Evaluate translation quality using specified method."""
    embeddings_array = np.array(embeddings)
    if embeddings_array.ndim == 3:
        embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)
    elif embeddings_array.ndim != 2:
        raise ValueError("Embeddings must be a 2D array or a list of 2D arrays")

    if filtering_method_distance == 'cosine':
        similarities = cosine_similarity(embeddings_array)
        return np.mean(similarities[np.triu_indices_from(similarities, k=1)])
    elif filtering_method_distance == 'intra_cluster':
        distances = pdist(embeddings_array)
        return np.mean(distances)
    else:
        raise ValueError(f"Unknown evaluation method: {filtering_method_distance}")

def query_translate_and_assess(
    model_type_T,
    Q,
    descriptions,
    T_A,
    model_type_C,
    filtering_method_distance='cosine',
    filter_best_translation_method="distance",
    llama3_api_endpoint=None,
    openai_api_key=None,
    k=2,
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
        NL = translate_query_to_nl(model_type_T, Q, descriptions, **model_config, n_shot=n_shot, threshold=threshold)
        e_i = bert_embedding(NL)
        e_nl_i = q_embedding_model.encode(NL, convert_to_tensor=False)
        NL_embeddings_q_model.append(e_nl_i)
        NL_embeddings.append(e_i)
        translations.append(NL)

    print(f"translations {translations}")
    
    if filter_best_translation_method == "distance":
        quality_score = evaluate_distance_method(NL_embeddings, filtering_method_distance=filtering_method_distance)
        print(f"Translation quality score ({filtering_method_distance}): {quality_score}")
        # Determine acceptance based on thresholds
        accept = quality_score >= T_A
        intra_cluster_distance = quality_score
        # Choose the best translation
        best_translation_index = np.argmax([evaluate_distance_method([e], filtering_method_distance=filtering_method_distance) for e in NL_embeddings])
        best_translation = translations[best_translation_index]
        best_translation_embedding = NL_embeddings[best_translation_index]
        best_translation_embedding_q_model = NL_embeddings_q_model[best_translation_index]
        
        
    elif filter_best_translation_method == 'q_nl_score':
        intra_cluster_distance = None
        quality_score = None
        accept = None
        best_translation_index = np.argmax([cosine_similarity(query_embedding.reshape(1, -1), e.reshape(1, -1))[0][0] for e in NL_embeddings_q_model])
        best_translation = translations[best_translation_index]
        best_translation_embedding = NL_embeddings[best_translation_index]
        best_translation_embedding_q_model = NL_embeddings_q_model[best_translation_index]
        
        
    else:
        # here we need to get the index of best translation
        best_translation = compare_query_to_nl(model_type_C, Q, translations, **model_config)
        accept = None
        intra_cluster_distance = None
        # for now, embedding again
        best_translation_embedding = bert_embedding(best_translation)
        best_translation_embedding_q_model = q_embedding_model.encode(best_translation, convert_to_tensor=False)
        quality_score = None



    bert_q_nl_score = cosine_similarity(query_embedding.reshape(1, -1), best_translation_embedding_q_model.reshape(1, -1))[0][0]
    bert_nl_nl_gt_score = cosine_similarity(best_translation_embedding.reshape(1, -1), nl_gt_embedding.reshape(1, -1))[0][0]


    return best_translation, accept, quality_score, intra_cluster_distance, bert_q_nl_score, bert_nl_nl_gt_score

def evaluate_dataset(combined_df, model_type_T, best_T_A, model_type_C, filter_best_translation_method, filtering_method_distance, llama3_api_endpoint, openai_api_key):
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

    # Dataframe to store the results
    results_df = pd.DataFrame(columns=["sparql_wikidata", "paraphrased_question", "question", "best_translation", "bert_q_NL", "bert_NL_NL_gt", "intra_cluster_distance"])


    for index, row in combined_df.iterrows():
        Q = row["sparql_wikidata_translated"]
        descriptions = row["descriptions"]
        print("-" * 50)
        translation, accept, quality_score, intra_cluster_distance, bert_q_nl_score, bert_nl_nl_gt_score = query_translate_and_assess(
            model_type_T,
            Q,
            descriptions,
            best_T_A,
            model_type_C,
            filter_best_translation_method=filter_best_translation_method,
            filtering_method_distance=filtering_method_distance,
            llama3_api_endpoint=llama3_api_endpoint,
            openai_api_key=openai_api_key,
            k=k,
            n_shot=1,
            threshold=best_T_A,
            NL_gt=row["paraphrased_question"],
            Q_raw=row["sparql_wikidata"])

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
        results_df.to_csv('data/v2/translation_results.csv')

# Example usage
if __name__ == "__main__":

    ### Parameters ###
    # Initialize SPARQL query Q\
    load_limit = 300

    dataset_name = "data/v2/lc_quad_preprocessed.csv"
    # whether the q-embedding model works with queries that have wikidata codes replaced (instantiated) by label or not
    Q_EMBEDDING_MODE = 'uninstantiated'
    # LLaMA3 usage
    llama3_api_endpoint = "http://localhost:11434/api/generate"
    # GPT usage
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Choose model type: "llama3" or "gpt" for translation
    # and critique
    model_type_C = "gpt"
    model_type_T = "gpt"  # "llama3" or "gpt"

    # How many answers to sample per query
    k = 3
    
    # Filtering method for best translation
    filter_best_translation_method = "distance" # "distance" or "q_nl_score" or None(nl compare)
    # Filtering method for distance
    filtering_method_distance = "intra_cluster" # if distance is used, then "cosine" or "intra_cluster"

    # Check if the combined dataset is saved already
    combined_df = load_or_combine_dataset(dataset_name)
    combined_df = combined_df.iloc[:load_limit]

    best_T_A = load_or_optimize_threshold(
        "data/v2/acceptance_threshold.txt",
        model_type_T,
        model_type_C,
        combined_df,
        llama3_api_endpoint,
        openai_api_key
    )

    # Evaluate translations
    evaluate_dataset(
        combined_df,
        model_type_T,
        best_T_A,
        model_type_C,
        filter_best_translation_method,
        filtering_method_distance,
        llama3_api_endpoint,
        openai_api_key
    )

