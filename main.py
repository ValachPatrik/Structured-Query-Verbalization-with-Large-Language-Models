import os
import re
import requests
import json
import pandas as pd
from typing import *
from wikidata.client import Client
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu

def find_translatable_parts(sparql_query: str) -> List[str]:
    entity_pattern = re.compile(r'wd:Q\d+')
    property_pattern = re.compile(r'wdt:P\d+')

    entity_matches = entity_pattern.findall(sparql_query)
    property_matches = property_pattern.findall(sparql_query)

    return entity_matches + property_matches

def map_wikidata_to_natural_language(sparql_query:str) -> str:
    client = Client()
    
    translatable_parts = find_translatable_parts(sparql_query)
    
    for i in translatable_parts:
        try:
            entity = client.get(i.split(":")[1], load=True)
            sparql_query = sparql_query.replace(i, str(entity.label))
        except Exception as e:
            print(e, end=" ")
            print(i)
            return None
    return sparql_query
    

def load_lc(load_limit: int) -> pd.DataFrame:
    if os.path.exists("lc_quad_translated.csv"):
        # Load locally if data is already locally ready
        combined_df = pd.read_csv("lc_quad_translated.csv").head(load_limit)
        print("Loaded from local")
    else:
        # Load data from net
        lc_quad_dataset = load_dataset("lc_quad")

        # combine both datasets test and train, we dont need this kind of differentiation
        train_df = lc_quad_dataset["train"].to_pandas()
        test_df = lc_quad_dataset["test"].to_pandas()
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        combined_df = combined_df.head(load_limit)

        # Apply KG to map from ambiguous descriptions to NL
        combined_df["wikidata_translated"] = combined_df["sparql_wikidata"].map(map_wikidata_to_natural_language)
        
        # Delete failed rows
        combined_df = combined_df[combined_df["wikidata_translated"] != None]
        
        # Save DF for future use to save resources
        combined_df.to_csv("lc_quad_translated.csv", index=False)
        
        print("Loaded from net, load limit is {load_limit}")
    print(f"Loaded {len(combined_df.index)} rows")
    # for i in range(5):
    #     print(combined_df.at[i, 'question'])
    #     print(combined_df.at[i, 'paraphrased_question'])
    #     print(combined_df.at[i, 'wikidata_translated'])
    return combined_df






def generate_response_llama(sparql_query: str, model: str, api_endpoint: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt + sparql_query + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    }
    response = requests.post(api_endpoint, json=payload, stream=True)
    response_text = ""
    for line in response.text.split("\n"):
        try:
            response_text += line.split('response":"')[1].split('","done":')[0]
        except:
            pass
    print(response_text)
    return response_text
    
def use_llm(df: pd.DataFrame, model: str, api_endpoint: str, load_limit: int, prompt: str) -> pd.DataFrame:
    if os.path.exists(f"lc_quad_translated_{model}.csv"):
        # Load locally if data is already locally ready
        df = pd.read_csv(f"lc_quad_translated_{model}.csv").head(load_limit)
        print("Loaded from local model responses")
    else:
        df = df.head(load_limit)
        df[f'{model}_response'] = df['wikidata_translated'].apply(lambda x: generate_response_llama(x, model, api_endpoint, prompt))
    
        df.to_csv(f"lc_quad_translated_{model}.csv", index=False)
        print(f"Saved {len(df.index)} rows with model responses")
    
    print(f"Loaded {len(df.index)} rows")
    return df





def eval_manual(df: pd.DataFrame, model: str, size_manual_eval: int) -> pd.DataFrame:
    if os.path.exists(f"lc_quad_translated_{model}_evaluated.csv"):
        # Load locally if data is already locally ready
        print(f"Evaluated file already exists. Do you want to overwrite/add manual eval? (y/n)")
        answer = input()
        if answer == "y":
            df = eval_manual_logic(df, model, size_manual_eval)
        else:
            df = pd.read_csv(f"lc_quad_translated_{model}_evaluated.csv")
    else:
        df = eval_manual_logic(df, model, size_manual_eval)
    
    df.to_csv(f"lc_quad_translated_{model}_evaluated.csv", index=False)
    print("Saved manual eval")
    return df

def eval_manual_logic(df: pd.DataFrame, model: str, size_manual_eval: int) -> pd.DataFrame:
    print("Creating manual evaluation")
    print(min(size_manual_eval, len(df.index)))
    for i in range(min(size_manual_eval, len(df.index))):
        print(i)
        print(f"Dataset: {df.at[i, 'question']}, {df.at[i, 'paraphrased_question']}")
        print(f"Model: {df.at[i, f'{model}_response']}")
        print("Are these two the same? (y/n) ", end="")
        response = input()
        print()
        if response == "y":
            df.at[i, 'eval_manual'] = 1
        else:
            df.at[i, 'eval_manual'] = 0
    print("Manual eval created")
    return df



def eval_llm(df: pd.DataFrame, model: str, api_endpoint: str, size_manual_eval: int) -> pd.DataFrame:
    df = df.head(size_manual_eval)
    if os.path.exists(f"lc_quad_translated_{model}_evaluated.csv"):
        # Load locally if data is already locally ready
        print(f"Evaluated file already exists. Do you want to overwrite/add llm eval? (y/n)")
        answer = input()
        if answer == "y":
            df['eval_llm'] = df.apply(lambda x: eval_llm_logic(x['paraphrased_question'], x[f'{model}_response'], model, api_endpoint), axis=1)
        else:
            df = pd.read_csv(f"lc_quad_translated_{model}_evaluated.csv").head(size_manual_eval)
    else:
        df['eval_llm'] = df.apply(lambda x: eval_llm_logic(x["paraphrased_question"], x[f"{model}_response"], model, api_endpoint), axis=1)
    
    df.to_csv(f"lc_quad_translated_{model}_evaluated.csv", index=False)
    print("Saved llm eval")
    return df
    
def eval_llm_logic(dataset: str, model_response: str, model: str, api_endpoint: str) -> int:
    payload = {
        "model": model,
        "prompt": "Respond with a single word. Are the following sentences semantically the same?\n" + str(dataset) + "\n" + str(model_response)
    }
    response = requests.post(api_endpoint, json=payload, stream=True)
    response_text = ""
    for line in response.text.split("\n"):
        try:
            response_text += line.split('response":"')[1].split('","done":')[0]
        except:
            pass

    print(response_text)
    if response_text[:3].lower() == "yes":
        return 1
    elif response_text[:2].lower() == "no":
        return 0
    else:
        print("Something went wrong")
        print(response_text)
        return 0



def eval_mlp_bert(df: pd.DataFrame, model: str, load_limit: int) -> pd.DataFrame:
    df = df.head(load_limit)
    if os.path.exists(f"lc_quad_translated_{model}_evaluated.csv"):
        # Load locally if data is already locally ready
        print(f"Evaluated file already exists. Do you want to overwrite/add bert eval? (y/n)")
        answer = input()
        if answer == "y":
            df['eval_bert'] = df.apply(lambda x: eval_bert_logic(x["paraphrased_question"], x[f"{model}_response"]), axis=1)
        else:
            df = pd.read_csv(f"lc_quad_translated_{model}_evaluated.csv").head(load_limit)
    else:
        df['eval_bert'] = df.apply(lambda x: eval_bert_logic(x["paraphrased_question"], x[f"{model}_response"]), axis=1)
    
    df.to_csv(f"lc_quad_translated_{model}_evaluated.csv", index=False)
    print("Saved bert eval")
    return df

def eval_bert_logic(dataset: str, model_response: str):
    # scorer = BERTScorer(model_type='bert-base-uncased')
    # P, R, F1 = scorer.score([dataset], [model_response])
    # print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
    # return (P, R, F1)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    inputs1 = tokenizer(dataset, return_tensors="pt", padding=True, truncation=True)
    inputs2 = tokenizer(model_response, return_tensors="pt", padding=True, truncation=True)

    outputs1 = model(**inputs1)
    outputs2 = model(**inputs2)

    embeddings1 = outputs1.last_hidden_state.mean(dim=1).detach().numpy()
    embeddings2 = outputs2.last_hidden_state.mean(dim=1).detach().numpy()

    similarity = np.dot(embeddings1, embeddings2.T) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))

    return similarity[0][0]



def eval_mlp_bleu(df: pd.DataFrame, model: str, load_limit: int) -> pd.DataFrame:
    df = df.head(load_limit)
    if os.path.exists(f"lc_quad_translated_{model}_evaluated.csv"):
        # Load locally if data is already locally ready
        print(f"Evaluated file already exists. Do you want to overwrite/add bleu eval? (y/n)")
        answer = input()
        if answer == "y":
            df['eval_bleu'] = df.apply(lambda x: eval_bleu_logic(x["paraphrased_question"], x["question"], x[f"{model}_response"]), axis=1)
        else:
            df = pd.read_csv(f"lc_quad_translated_{model}_evaluated.csv").head(load_limit)
    else:
        df['eval_bleu'] = df.apply(lambda x: eval_bleu_logic(x["paraphrased_question"], x["question"], x[f"{model}_response"]), axis=1)
    
    df.to_csv(f"lc_quad_translated_{model}_evaluated.csv", index=False)
    print("Saved bleu eval")
    return df

def eval_bleu_logic(dataset1: str, dataset2: str, model_response: str):
    dataset = [dataset1, dataset2]
    reference = [question.split(" ") for question in dataset]
    predictions = model_response.split(" ")
    score = sentence_bleu(reference, predictions) # TODO may need to play with weights
    return score



def accuracy(correct, total):
    return correct/total
def precision(true_positives, false_positives):
    return true_positives / (true_positives + false_positives)
def recall(true_positives, false_negatives):
    return true_positives / (true_positives + false_negatives)
def f1(precision, recall):
    return 2*precision*recall / (precision + recall)






# Constants
load_limit = 100
model = "llama3"
api_endpoint = "http://localhost:11434/api/generate"
prompt_translate = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a AI translator for converting sparql queries into normal natural language questions<|eot_id|><|start_header_id|>user<|end_header_id|>
"Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: select distinct ?obj where { Delta Air Lines house publication ?obj . ?obj instance of periodical }<|eot_id|><|start_header_id|>assistant<|end_header_id|>
What periodical literature does Delta Air Lines use as a moutpiece?<|eot_id|>
<|start_header_id|>user<|end_header_id|>
SELECT ?answer WHERE { Ranavalona I of Madagascar spouse ?X . ?X father ?answer}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
What is the name of Ranavalona I's husband's child?<|eot_id|><|start_header_id|>user<|end_header_id|>
ASK WHERE { Jeff Bridges occupation Lane Chandler . Jeff Bridges occupation photographer }<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Are Jeff Bridges and Lane Chandler both photographers?<|eot_id|><|start_header_id|>user<|end_header_id|>
"""
#"Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: "
size_manual_eval = 30

# Load data
df = load_lc(load_limit)
# Run LLM
df = use_llm(df, model, api_endpoint, load_limit, prompt_translate)
# Evaluate
    # 1 - manual 0/1
df = eval_manual(df, model, size_manual_eval)
    # 2 - LLM
df = eval_llm(df, model, api_endpoint, size_manual_eval)
    # 3 - MLP
        # BERT
df = eval_mlp_bert(df, model, load_limit)
        # BLEU
df = eval_mlp_bleu(df, model, load_limit)


eval_manual_llm_same = df.head(size_manual_eval)[(df['eval_manual'] == df['eval_llm'])].shape[0]
eval_manual_llm_true_positives = df.head(size_manual_eval)[(df['eval_manual'] == 1) & (df['eval_llm'] == 1)].shape[0]
eval_manual_llm_false_positives = df.head(size_manual_eval)[(df['eval_manual'] == 0) & (df['eval_llm'] == 1)].shape[0]
eval_manual_llm_false_negatives = df.head(size_manual_eval)[(df['eval_manual'] == 1) & (df['eval_llm'] == 0)].shape[0]
eval_manual_llm_true_negatives = df.head(size_manual_eval)[(df['eval_manual'] == 0) & (df['eval_llm'] == 0)].shape[0]
eval_llm_total_1 = df[df['eval_llm'] == 1].shape[0]

llm_accuracy = accuracy(eval_manual_llm_same, size_manual_eval)
llm_precision = precision(eval_manual_llm_true_positives, eval_manual_llm_false_positives)
llm_recall = recall(eval_manual_llm_true_positives, eval_manual_llm_false_negatives)
llm_f1 = f1(llm_precision, llm_recall)

print("LLM")
print(f"Accuracy: {llm_accuracy}")
print(f"Precision: {llm_precision}")
print(f"Recall: {llm_recall}")
print(f"F1 Score: {llm_f1}")
print(f"Totally evaluated as correct by model: {eval_llm_total_1} out of {df.shape[0]}")

eval_bert_min = df['eval_bert'].min()
eval_bert_max = df['eval_bert'].max()
eval_bert_average = df['eval_bert'].mean()
eval_bert_median = df['eval_bert'].median()
eval_bert_mode = df['eval_bert'].mode()[0]

print("Eval BERT Statistics")
print(f"Min: {eval_bert_min}")
print(f"Max: {eval_bert_max}")
print(f"Average: {eval_bert_average}")
print(f"Median: {eval_bert_median}")
print(f"Mode: {eval_bert_mode}")

# Plot BERT values
plt.hist(df['eval_bert'], bins=10)
plt.xlabel('BERT Values')
plt.ylabel('Frequency')
plt.title('Distribution of BERT Values')
plt.show()

eval_bleu_min = df['eval_bleu'].min()
eval_bleu_max = df['eval_bleu'].max()
eval_bleu_average = df['eval_bleu'].mean()
eval_bleu_median = df['eval_bleu'].median()
eval_bleu_mode = df['eval_bleu'].mode()[0]



print("Eval BLEU Statistics")
print(f"Min: {eval_bleu_min}")
print(f"Max: {eval_bleu_max}")
print(f"Average: {eval_bleu_average}")
print(f"Median: {eval_bleu_median}")
print(f"Mode: {eval_bleu_mode}")

# Plot BLEU values
plt.hist(df['eval_bleu'], bins=10)
plt.xlabel('BLEU Values')
plt.ylabel('Frequency')
plt.title('Distribution of BLEU Values')
plt.show()