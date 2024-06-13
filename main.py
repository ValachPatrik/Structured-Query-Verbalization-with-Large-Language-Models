import os
import re
import requests
import json
import math
import pandas as pd
from typing import *
from wikidata.client import Client
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import matplotlib.pyplot as plt
import nltk
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
        
        print(f"Loaded from net, load limit is {load_limit}")
    print(f"Loaded {len(combined_df.index)} rows")
    # print('/DEBUG/')
    # for i in range(1, 31):
    #     print(combined_df.at[len(combined_df.index) - i, 'question'])
    #     print(combined_df.at[len(combined_df.index) - i, 'paraphrased_question'])
    #     print(combined_df.at[len(combined_df.index) - i, 'wikidata_translated'])
    # print('//DEBUG//')
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
    # print('/DEBUG/')
    # for i in range(31, 61):
    #     print("Respond with a single word. Are the following sentences semantically the same?:")
    #     print(df.at[len(df.index) - i, 'paraphrased_question'])
    #     print(df.at[len(df.index) - i, f'{model}_response'] + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>')  
    #     print('XXXXX.<|eot_id|><|start_header_id|>user<|end_header_id|>') 
    # print('//DEBUG//')
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



def eval_llm(df: pd.DataFrame, model: str, api_endpoint: str, size_manual_eval: int, prompt: str) -> pd.DataFrame:
    df = df.head(size_manual_eval)
    if os.path.exists(f"lc_quad_translated_{model}_evaluated.csv"):
        # Load locally if data is already locally ready
        print(f"Evaluated file already exists. Do you want to overwrite/add llm eval? (y/n)")
        answer = input()
        if answer == "y":
            df['eval_llm'] = df.apply(lambda x: eval_llm_logic(x['paraphrased_question'], x[f'{model}_response'], model, api_endpoint, prompt), axis=1)
        else:
            df = pd.read_csv(f"lc_quad_translated_{model}_evaluated.csv").head(size_manual_eval)
    else:
        df['eval_llm'] = df.apply(lambda x: eval_llm_logic(x["paraphrased_question"], x[f"{model}_response"], model, api_endpoint, prompt), axis=1)
    
    df.to_csv(f"lc_quad_translated_{model}_evaluated.csv", index=False)
    print("Saved llm eval")
    return df
    
def eval_llm_logic(dataset: str, model_response: str, model: str, api_endpoint: str, prompt: str) -> int:
    payload = {
        "model": model,
        "prompt": prompt + str(dataset) + "\n" + str(model_response) + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
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
    if type(dataset1) != str:
        print(dataset1)
        dataset1 = "n/a"
    if type(dataset2) != str:
        print(dataset2)
        dataset2 = "n/a"
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

def stats(df, size_manual_eval):
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
    print(f"Second LLM run made {eval_manual_llm_false_positives + eval_manual_llm_false_negatives} mistakes")
    print(f"Actual good results are {eval_manual_llm_true_positives} out of {size_manual_eval}")

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




# Constants
prompt_translate = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a AI translator for converting sparql queries into normal natural language questions<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: SELECT DISTINCT ?sbj ?sbj_label WHERE { ?sbj instance of physical phenomenon . ?sbj rdfs:label ?sbj_label . FILTER(CONTAINS(lcase(?sbj_label), 'surface')) . FILTER (lang(?sbj_label) = 'en') } LIMIT 25<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Let me know physical marvel whose title has the word surface in it.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: ASK WHERE { Algeria total fertility rate ?obj filter(?obj > 3.4284) }<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Is the entire richness rate of Algeria more noteworthy than 3.4284?<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: SELECT ?obj WHERE { John de Mol p:P166 ?s . ?s ps:P166 ?obj . ?s pq:P585 ?x filter(contains(YEAR(?x),'2011')) }<|eot_id|><|start_header_id|>assistant<|end_header_id|>
in 2011 John de Mol won which award?<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: select distinct ?obj where { Sasha Grey contributed to creative work ?obj . ?obj instance of musical duo }<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Say the melodic related work highlighting the work of Sasha Grey.<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: select distinct ?obj where { Deadpool performer ?obj . ?obj instance of human }<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Who was the lead performing artist for the motion picture Deadpool?<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: SELECT ?answer WHERE { Ultima III: Exodus part of the series ?X . ?X input device ?answer}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Who organizes the arrange of Ultima III: Departure?<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: SELECT DISTINCT ?sbj ?sbj_label WHERE { ?sbj instance of Greek deity . ?sbj sibling Zeus . ?sbj rdfs:label ?sbj_label . FILTER(CONTAINS(lcase(?sbj_label), 'poseidon')) . FILTER (lang(?sbj_label) = 'en') } LIMIT 25<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Let me know the Greek deity of kin of Zeus which contains the word poseidon in it's name?<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: select distinct ?obj where { Alexander McQueen employer ?obj . ?obj instance of business }<|eot_id|><|start_header_id|>assistant<|end_header_id|>
What's the name of Alexander McQueen's business?<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: SELECT ?answer WHERE { Shigeno Yasutsugu student of ?X . ?X student ?answer}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Who was the instructor that administered Shigeno Yasutsugu?<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: SELECT DISTINCT ?sbj ?sbj_label WHERE { ?sbj instance of empire . ?sbj coat of arms royal coat of arms of the United Kingdom . ?sbj rdfs:label ?sbj_label . FILTER(CONTAINS(lcase(?sbj_label), 'british')) . FILTER (lang(?sbj_label) = 'en') } LIMIT 25<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Title an realm that contains the word "british" in its name<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: SELECT ?answer WHERE { Grande Odalisque depicts ?answer . ?answer direction relative to location backwards}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
What does the portray Review Odalisque speak to which has DRTL backwards?<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: SELECT ?answer WHERE { Qur’an author ?X . ?X relative ?answer}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Give me the name of the brother in law of the writer of Quran.<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: SELECT DISTINCT ?sbj ?sbj_label WHERE { ?sbj instance of horse breed . ?sbj rdfs:label ?sbj_label . FILTER(STRSTARTS(lcase(?sbj_label), 'z')) . FILTER (lang(?sbj_label) = 'en') } LIMIT 25<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Tell me each and every horse breed whose identify begins with the letter z<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: SELECT ?answer WHERE { Minigun ammunition ?answer . ?answer conflict Vietnam War}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
What is the ammunition of the weapon, that has been used in the conflict in Southeast Asia?<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: SELECT ?answer WHERE { Lily Tomlin award received ?X . ?X winner ?answer}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Who is {champ} of {prize granted} {Lily Tomlin} ?TARD ?<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: SELECT ?answer WHERE { New York Centaurs home venue ?X . ?X architect ?answer}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
who engineering firm of domestic field of modern york centaurs?<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: SELECT ?obj WHERE { hydrogen peroxide p:P2054 ?s . ?s ps:P2054 ?obj . ?s pq:P2076 ?x filter(contains(?x,'20.0')) }<|eot_id|><|start_header_id|>assistant<|end_header_id|>
What is hydrogen peroxide's density at twenty degrees?<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: ASK WHERE { Merion Golf Club slope rating ?obj filter(?obj = 149) }<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Is the slope rating of the Merion Golf Club 149?<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: select distinct ?obj where { Janet Jackson record label ?obj . ?obj instance of record label }<|eot_id|><|start_header_id|>assistant<|end_header_id|>
What is the record label signed by Janet Jackson?<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: SELECT ?obj WHERE { Hans Krebs p:P166 ?s . ?s ps:P166 ?obj . ?s pq:P585 ?x filter(contains(YEAR(?x),'1966')) }<|eot_id|><|start_header_id|>assistant<|end_header_id|>
What is {grant gotten} of {Hans Krebs} where {point in time} is {1966-0-0} ?<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: select distinct ?sbj where { ?sbj fictional universe described in The Matrix . ?sbj instance of fictional universe }<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Mention the fictional universe described or included in The Matrix.<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: SELECT DISTINCT ?sbj ?sbj_label WHERE { ?sbj instance of film character . ?sbj from narrative universe Marvel Universe . ?sbj rdfs:label ?sbj_label . FILTER(STRSTARTS(lcase(?sbj_label), 'w')) . FILTER (lang(?sbj_label) = 'en') } LIMIT 25<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Give me a movie personality from a fictional universe, such as Marvel comics that starts with a W.<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: SELECT ?answer WHERE { Infinite Jest narrative location ?answer . ?answer shares border with Quincy}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
What is {located next to Quincy,} in the {novel, Infinite Jest}?<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: ASK WHERE { Aleister Crowley educated at Trinity College . Aleister Crowley educated at Eton College }<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Did Aleister Crowley receive his education at Trinity College and Eton College?<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: 
"""
#"Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: "
prompt_is_equal = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a AI comparator of sentences comparing if their semantic is the same.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
Does Vest-vassdraget have a throughput of 2697.672?
Does the water flow rate of Vest-vassdraget equal 2697.672?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Yes.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
What is the title of a Cayenne Pepper that too has dates?
What is the pepper, named after which, has an origin dating back to 1664?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
No.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
How is the naval artillery wirh the smallest firing range called?
What are the five naval artillery pieces, listed in order from shortest to longest effective firing range?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Yes.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
Where at Passy Cementery is Fernandel buried?
What is the date of death for Fernandel, buried at Passy Cemetery?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
No.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
What is hooked up by way of Archbishop of Canterbury, who is a male?
Who is the male founder of the office of the Archbishop of Canterbury?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Yes.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
What is a purpose of dying that begins with the letter "p" and can be located on a CT scan?
What are the names of causes of death that have been diagnosed through a CT scan and start with P?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Yes.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
Which borders of borders of Riverside have a begin date of 1984-0-0?
When was Riverside founded, which is bordered by another city that started or began operating in 1984?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
No.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
On what date was Triple Crown Trophy given to Secretariat?
What is the year {Triple Crown winner} won the award?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Yes.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
Are there more than survivors of the Charkhi Dadri mid-air collision
Was there no survivor in the mid-air collision that occurred at Charkhi Dadri?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Yes.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
Former champion Francisco Alarcon gave what award to Art Spiegelaman?
What award did Francisco X. Alarcón win, as won by Art Spiegelman?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Yes.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
Not applicable
What ancient writings have an edition or translation known as the Septuagint?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
No.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
When was the companion separated Nero in 9-6-68?
Who was emperor when Nero died on June 9, 68?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Yes.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
The football association regulates what organization?
What international governing body regulates the sport of association football?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Yes.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
In what year was Nicaragua's population 3.87732 million?
What is the latitude of Nicaragua?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
No.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
Located in the Central District, what is the county seat whose twin cities include Feodosiya?
What is the capital of Central District that has a twin city or administrative partnership with Feodosiia?<|eot_id|><|start_header_id|>assistant<|end_header_id|>        
No.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
Where did the Golden Horde live in?
What was the country ruled by the Golden Horde?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Yes.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
Of the century breaks of the Colm Gilcreest rise to less than 9.6?
Did Colm Gilcreest make centuries in his career that are less than 9.6?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Yes.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
What majestic state supplanted the Kingdom of Incredible Britain?
What country replaced the Kingdom of Great Britain?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Yes.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
Which species is the ecia139-4120 protein found in?
In which environment or ecosystem is ECIAI39_4120, a hypothetical protein, typically found?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Yes.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
What kind of career in the screenwriting field does Grigori Kozintsev have?
What is the job or profession that Grigori Kozintsev was involved in, specifically focused on writing for films?<|eot_id|><|start_header_id|>assistant<|end_header_id|>  
Yes.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
What year did Emmerson Mnangagwa start at the University of Zambia?
What is the date of graduation or academic award received by Emmerson Mnangagwa from the University of Zambia?<|eot_id|><|start_header_id|>assistant<|end_header_id|>    
Yes.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
In Indonesia, is the average life expectancy 55.3528?
Is the average lifespan of people in Indonesia exactly 55 years and 20 months?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
No.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
Where is the geographic center of Michigan using the gravity center of the surface?
What is the geographic point that serves as the central location for the state boundaries of Michigan?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Yes.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
What product that contains paraffin wax has the lowest usage per capita?
What are the top 5 products, ranked by their per capita consumption, that contain or use paraffin wax as a component?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Yes.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
On what date did Pablo Picasso end his partnership with Fernade Oliver?
What is the date of marriage between Pablo Picasso and Fernande Olivier?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
No.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
What was the date that The St. Louis Literary Award was aquired by James Thomas Farrell.
What is the year that James Thomas Farrell received the St. Louis Literary Award?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Yes.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
What country has the highest taxes?
What are the top 5 countries by their highest individual income tax rates, listed in descending order?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Yes.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
Is 11 the maximum age of St. Peters Junior School?
What is the maximum age for students at St Peter's Junior School, which is 11 years old?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
No.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
Who is Hank Azaria married to ?
Who is Hank Azaria's unmarried partner and who is his spouse?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
No.<|eot_id|><|start_header_id|>user<|end_header_id|>
Respond with a single word. Are the following sentences semantically the same?:
Did Brittany Murphy have USA citizenship?
Was Brittany Murphy a citizen of the United States of America?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Yes.<|eot_id|><|start_header_id|>user<|end_header_id|>
"""
#"Respond with a single word. Are the following sentences semantically the same?\n"
model = "llama3"
api_endpoint = "http://localhost:11434/api/generate"
load_limit = 100
size_manual_eval = 40


# Load data
df = load_lc(load_limit)
# Run LLM
df = use_llm(df, model, api_endpoint, load_limit, prompt_translate)
# Evaluate
    # 1 - manual 0/1
df = eval_manual(df, model, size_manual_eval)
    # 2 - LLM
df = eval_llm(df, model, api_endpoint, size_manual_eval, prompt_is_equal)
    # 3 - MLP
        # BERT
df = eval_mlp_bert(df, model, load_limit)
        # BLEU
df = eval_mlp_bleu(df, model, load_limit)

stats(df, size_manual_eval)



# XXX The 41-70 are used to train second LLM run - DO NOT USE FOR ACTUAL RUN
# XXX The 71-100 are used to train first LLM run - DO NOT USE FOR ACTUAL RUN