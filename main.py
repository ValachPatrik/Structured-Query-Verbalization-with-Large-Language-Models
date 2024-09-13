from helper_functions import *

import requests
import openai
import numpy as np
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

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
    #print(f"llama response {response_text}")
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
    #print(f"llama response {response_text}")
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
    
def compare_query_to_nl(model_type, Q, translations, api_endpoint=None, openai_api_key=None):
    if model_type == "llama3":
        return llama3_compare_translations(api_endpoint, Q, translations, model=model_type)
    elif model_type == "gpt":
        return gpt_compare_translations(openai_api_key, Q, translations)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# Compute BERT embeddings for the text
def bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


# Calculate average pairwise distance between embeddings
def calculate_average_pairwise_embedding_distance(embeddings):
    distances = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            # print(len([embeddings[i], embeddings[j]]))
            distances.append(
                cosine_similarity(
                    embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1)
                )[0][0]
            )
            # print(distances[-1])

    print(distances)
    upper_tri = np.triu(distances, k=1)
    return upper_tri.mean()


# Function to fine-tune threshold based on training data (ground truth translations)
def tune_thresholds(train_data, T_A, T_R):
    # Example train_data is a list of (Q, true_NL) tuples
    for Q, true_NL in train_data:
        e_q = bert_embedding(Q)
        e_true = bert_embedding(true_NL)
        distance = cosine_similarity([e_q], [e_true])
        # Update T_A, T_R based on the distances observed during training
        # Placeholder for tuning logic
    return T_A, T_R

# Main function for translating and assessing SPARQL query
def translate_and_assess(
    model_type, Q, T_A, T_R, C, api_endpoint=None, openai_api_key=None, k=3
):
    NL_embeddings = []
    translations = []
    accept = None

    # Translate Query through KG
    Q = map_wikidata_to_natural_language(Q)
    print(Q)

    # Generate k proposals and compute BERT embeddings
    for i in range(k):
        # print(i)
        NL = translate_query_to_nl(model_type, Q, api_endpoint, openai_api_key)
        e_i = bert_embedding(NL)
        NL_embeddings.append(e_i)
        translations.append(NL)

    # print(f"nl embeddings {NL_embeddings}")
    # for i in range(len(NL_embeddings)):
    # print(i)
    # print(NL_embeddings[i])
    print(f"translations {translations}")

    # Calculate average intra-cluster embedding distance
    mu_d = calculate_average_pairwise_embedding_distance(NL_embeddings)
    print(f"mu_d {mu_d}")

    if mu_d < T_A:
        accept = True
    elif mu_d > T_R:
        accept = False

    # print(accept)
    # Option 1: Use query embedding to choose the best translation
    if accept:
        e_q = bert_embedding(Q)
        distances = [
            cosine_similarity(e_q.reshape(1, -1), e_i.reshape(1, -1))[0][0]
            for e_i in NL_embeddings
        ]
        best_index = np.argmax(distances)
        final_translation_bert = translations[best_index]
    # Option 2: Use critique model LLM to choose the best translation
    if accept:
        final_translation_llm = compare_query_to_nl(C, Q, translations, api_endpoint, openai_api_key)
        pass

    return final_translation_bert, final_translation_llm, accept


# Example usage
if __name__ == "__main__":
    # Initialize SPARQL query Q\
    load_limit = 30

    lc_quad_dataset = load_dataset("lc_quad", trust_remote_code=True)

    # combine both datasets test and train, we dont need this kind of differentiation
    train_df = lc_quad_dataset["train"].to_pandas()
    test_df = lc_quad_dataset["test"].to_pandas()
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    combined_df = combined_df.iloc[:load_limit]

    Q = " select distinct ?obj where { wd:Q188920 wdt:P2813 ?obj . ?obj wdt:P31 wd:Q1002697 } "

    # Thresholds for acceptance and rejection
    T_A = 0.5  # Placeholder threshold for acceptance
    T_R = 0.8  # Placeholder threshold for rejection

    # LLaMA3 usage
    llama3_api_endpoint = "http://localhost:11434/api/generate"

    # GPT usage
    openai_api_key = "your_openai_api_key"

    # Choose critique model type: "llama3" or "gpt"
    C = "llama3"

    # Choose model type: "llama3" or "gpt"
    model_type = "llama3"  # or "gpt"

    # Translate and assess

    for index, row in combined_df.iterrows():
        Q = row[
            "sparql_wikidata"
        ]  # Assuming 'sparql_query' is the column name for SPARQL queries
        print("-" * 50)
        translation_bert, translation_llm, accept = translate_and_assess(
            model_type,
            Q,
            T_A,
            T_R,
            C,
            api_endpoint=llama3_api_endpoint,
            openai_api_key=openai_api_key,
            k=3,
        )

        print(f"Query {index + 1}:")
        print("SPARQL Query:", Q)
        print("Final Translation Bert:", translation_bert)
        print("Final Translation LLM:", translation_llm)
        print("Accepted:", accept)
        print(f"supposed question paraphrased: {row['paraphrased_question']}")
        print(f"supposed question: {row['question']}")
        print("-" * 50)

    # Optionally, tune thresholds using a training dataset
    # train_data = [(Q1, true_NL1), (Q2, true_NL2), ...]
    # T_A, T_R = tune_thresholds(train_data, T_A, T_R)
