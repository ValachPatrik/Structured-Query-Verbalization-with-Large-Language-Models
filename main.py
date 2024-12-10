import json

#from scipy.special import result
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

import time

import uuid


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

    # Result dictionary to be rea
    result_dict = {}
    all_embedding_dict = {}

    # Get model API configuration
    model_config = get_model_api(model_type_T, llama3_api_endpoint, openai_api_key)

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
        #NL = 'THIS IS A TEST TRANSLATION'
        e_i = bert_embedding(NL)
        e_nl_i = q_embedding_model.encode(NL, convert_to_tensor=False)
        NL_embeddings_q_model.append(e_nl_i)
        NL_embeddings.append(e_i)
        translations.append(NL)

    # Calculate mean pairwise distance or mean cosine similarity
    avg_pairwise_distance = evaluate_distance_method(NL_embeddings, filtering_method_distance='intra_cluster')
    avg_cosine_similarity = evaluate_distance_method(NL_embeddings, filtering_method_distance='cosine')

    #if filter_best_translation_method == "distance":
    if True:
        # Choose the best translation based on the one that has closest cosinesim to median embedding
        # based on related work "ScienceBenchmark: A Complex Real-World Benchmark for
        # Evaluating Natural Language to SQL Systems"

        # Get median embedding of k translations
        median_embedding = np.median(np.stack(NL_embeddings), axis=0)

        best_translation_index = np.argmax([cosine_similarity(median_embedding, e) for e in NL_embeddings])
        best_translation_dist = translations[best_translation_index]
        best_translation_embedding_dist = NL_embeddings[best_translation_index]
        best_translation_embedding_q_model_dist = NL_embeddings_q_model[best_translation_index]
        
        
    #elif filter_best_translation_method == 'q_nl_score':
    if True:
        best_translation_index = np.argmax([cosine_similarity(query_embedding.reshape(1, -1), e.reshape(1, -1))[0][0] for e in NL_embeddings_q_model])
        best_translation_Q = translations[best_translation_index]
        best_translation_embedding_Q = NL_embeddings[best_translation_index]
        best_translation_embedding_q_model_Q = NL_embeddings_q_model[best_translation_index]
        

    if True:
    #else:
        # here we need to get the index of best translation
        best_translation_llm_critique = compare_query_to_nl(model_type_C, Q, translations, **model_config)
        #best_translation_llm_critique = 'Just a test critique answer'

    # for now, embedding again
        best_translation_embedding_llm_critique = bert_embedding(best_translation_llm_critique)
        best_translation_embedding_q_model_llm_critique = q_embedding_model.encode(best_translation_llm_critique, convert_to_tensor=False)


    #if not filter_best_translation_method == 'llm_critique':
    #    best_translation_llm_critique = compare_query_to_nl(model_type_C, Q, translations, **model_config)
    #    best_translation_embedding_llm_critique = bert_embedding(best_translation_llm_critique)


    # Calculate Metrics for the three methods
    bert_q_nl_score_Q = cosine_similarity(query_embedding.reshape(1, -1), best_translation_embedding_q_model_Q.reshape(1, -1))[0][0]
    bert_nl_nl_gt_score_Q = cosine_similarity(best_translation_embedding_Q.reshape(1, -1), nl_gt_embedding.reshape(1, -1))[0][0]

    bert_q_nl_score_dist = cosine_similarity(query_embedding.reshape(1, -1), best_translation_embedding_q_model_dist.reshape(1, -1))[0][0]
    bert_nl_nl_gt_score_dist = cosine_similarity(best_translation_embedding_dist.reshape(1, -1), nl_gt_embedding.reshape(1, -1))[0][0]

    bert_q_nl_score_critique = cosine_similarity(query_embedding.reshape(1, -1), best_translation_embedding_q_model_llm_critique.reshape(1, -1))[0][0]
    bert_nl_nl_gt_score_critique = cosine_similarity(best_translation_embedding_llm_critique.reshape(1, -1), nl_gt_embedding.reshape(1, -1))[0][0]


    # Bert Scores for NL and NL_gt as well as NL and q
    result_dict['bert_q_nl_score_Q'] = float(bert_q_nl_score_Q)
    result_dict['bert_nl_nl_gt_score_Q'] = float(bert_nl_nl_gt_score_Q)
    result_dict['bert_q_nl_score_dist'] = float(bert_q_nl_score_dist)
    result_dict['bert_nl_nl_gt_score_dist'] = float(bert_nl_nl_gt_score_dist)
    result_dict['bert_q_nl_score_critique'] = float(bert_q_nl_score_critique)
    result_dict['bert_nl_nl_gt_score_critique'] = float(bert_nl_nl_gt_score_critique)

    # Embeddings for query and GT
    all_embedding_dict['q_embedding'] = query_embedding.tolist()
    all_embedding_dict['NL_gt_embedding'] = nl_gt_embedding.tolist()

    # All translations and its embeddding
    all_embedding_dict['NL_embeddings'] = [arr.tolist() for arr in NL_embeddings]
    all_embedding_dict['NL_embeddings_q_model'] = [arr.tolist() for arr in NL_embeddings_q_model]
    all_embedding_dict['translations'] = translations

    # Saving the best translations per filter and its embedding
    result_dict['best_translation_Q'] = best_translation_Q
    all_embedding_dict['best_translation_Q'] = best_translation_Q
    all_embedding_dict['best_translation_embedding_Q'] = best_translation_embedding_Q.tolist()
    result_dict['best_translation_dist'] = best_translation_dist
    all_embedding_dict['best_translation_dist'] = best_translation_dist
    all_embedding_dict['best_translation_embedding_dist'] = best_translation_embedding_dist.tolist()
    result_dict['best_translation_llm_critique'] = best_translation_llm_critique
    all_embedding_dict['best_translation_llm_critique'] = best_translation_llm_critique
    all_embedding_dict['best_translation_embedding_llm_critique'] = best_translation_embedding_llm_critique.tolist()

    # Filter metrics distance and cossim
    result_dict['avg_pairwise_distance'] = float(avg_pairwise_distance)
    result_dict['avg_cosine_similarity'] = float(avg_cosine_similarity)



    return result_dict, all_embedding_dict

def evaluate_dataset(combined_df, model_type_T, best_T_A, model_type_C, filter_best_translation_method,
                     filtering_method_distance, llama3_api_endpoint, openai_api_key, config):
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
    results_df = pd.DataFrame(columns=[
        "query_id", "original_query", "instantiated_query", "original_question",
        "bert_q_nl_score_Q", "bert_nl_nl_gt_score_Q",
        "bert_q_nl_score_dist", "bert_nl_nl_gt_score_dist",
        "bert_q_nl_score_critique", "bert_nl_nl_gt_score_critique",
        "best_translation_Q", "best_translation_dist",
        "best_translation_llm_critique",
        "avg_pairwise_distance", "avg_cosine_similarity"
    ])

    result_json = {}
    result_json['config'] = config
    result_json['queries'] = []

    # create result folder and filenames
    timestamp = str(int(time.time()))
    save_folder_name = os.path.join('results', timestamp)
    save_folder_name_queries = os.path.join(save_folder_name, 'queries')

    os.makedirs(save_folder_name, exist_ok=True)
    os.makedirs(save_folder_name_queries, exist_ok=True)

    # File paths for CSV and JSON
    csv_file_path = os.path.join(save_folder_name, "results.csv")

    for index, row in tqdm(combined_df.iterrows(), total=len(combined_df)):

        query_id = str(uuid.uuid4())

        Q = row["sparql_wikidata_translated"]
        descriptions = row["descriptions"]
        translation_result_dict, all_embeddings_dict = query_translate_and_assess(
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
            NL_gt=row["question"],
            Q_raw=row["sparql_wikidata"])

        # Adding Info about the query
        translation_result_dict['original_query'] = row['sparql_wikidata']
        translation_result_dict['instantiated_query'] = row['sparql_wikidata_translated']
        translation_result_dict['query_id'] = query_id
        translation_result_dict['original_question'] = row['question']


        results_df = pd.concat([results_df, pd.DataFrame([translation_result_dict])], ignore_index=True)
        results_df.to_csv(csv_file_path)

        with open(os.path.join(save_folder_name_queries, f"{query_id}.json"), 'w') as json_file:
            json.dump(all_embeddings_dict, json_file, indent=4)

# Example usage
if __name__ == "__main__":

    # Load Params from config
    config = load_config('config.json')

    ### Parameters ###
    # Initialize SPARQL query Q\
    load_limit = config['n_queries']
    dataset_name = config["dataset_name"]
    # whether the q-embedding model works with queries that have wikidata codes replaced (instantiated) by label or not
    Q_EMBEDDING_MODE = config['q_embedding_mode']
    # LLaMA3 usage
    llama3_api_endpoint =config["llama3_api_endpoint"]
    # GPT usage
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # Choose model type: "llama3" or "gpt" for translation
    # and critique
    model_type_C = config["model_type_C"]
    model_type_T = config["model_type_T"]  # "llama3" or "gpt"
    # How many answers to sample per query
    k = config["k"]
    # Filtering method for best translation
    filter_best_translation_method = config["filter_best_translation_method"] # "distance" or "q_nl_score" or "llm_critique"(nl compare)
    # Filtering method for distance
    filtering_method_distance = config["filtering_method_distance"] # if distance is used, then "cosine" or "intra_cluster"

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
        openai_api_key,
        config=config
    )

