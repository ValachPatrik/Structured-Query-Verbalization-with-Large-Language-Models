# STRUCTURED QUERY VERBALIZATION WITH LARGE LANGUAGE MODELS
This project has been developed by Patrik Valach (patrik.valach@tum.de) & Louisa Siebel (ge94seq@mytum.de) under the supervision of M.Sc Tim Schwabe in the Technical University of Munich (TUM) course Data Engineering led by Prof. Maribel Acosta.

This project aims to provide a way of generating data for training LLMs for the specific use case of creating SQL queries from natural language.
This is done by doing the process in reverse, as there are many SQL query examples but not enough natural language ones. Thus, creating an LLM translator proven to give correct translations can be further used to create more data for training.

![image](https://github.com/ValachPatrik/dataengineering/assets/82080194/e5443f27-1384-4318-9da4-1b3de8eb86b7)

## The dataset
The dataset used for creating the prompt and testing is the lc_quad dataset https://huggingface.co/datasets/mohnish/lc_quad
The columns relevant to our purpose are 'question', 'paraphrased_question', 'sparql_wikidata'
There are 24k examples to be used to create and test our idea.
'sparql_wikidata' comes with wikidata entities and properties that are numbers. Those needed to be remapped into their natural language representation.

![image](https://github.com/ValachPatrik/dataengineering/assets/82080194/242f037d-2eac-4421-bc68-7fe16f0fc5ce)

Progress is saved after the mapping.

## Translation using LLM
The prompt is automatically generated from the last <200> rows of the dataset to showcase the model how to respond properly. Given our computing constraints and the formulation of the prompt we found these 200 rows to be sufficient, however an increase could pottentially improve the model's performance for a larger portion of the dataset.

The model used: llama3 8b
https://ollama.com/library/llama3

Room for improvement:
- Use a more powerful model like llama3 70b
- Increase the prompt size

![image](https://github.com/ValachPatrik/dataengineering/assets/82080194/590921ee-fa2d-4244-a4de-602a693b1862)

Progress is saved after the translation.

## Evaluation
To find the best filtering metric for incorrect translations of the data, 4 different evaluating methods were tested, we selected the best one with a threshold that automatically filters out all incorrect translations.\
The conclusion on how we decided on the final metric has been based on a smaller subset of manual evaluations.

Progress is saved after each evaluation, and the user is prompted if he wants to rerun it.

### Manual Evaluation
A manual evaluation was conducted on the same data as the automated evaluation methods.\
40 pairs of sentences were checked for semantic equivalence.\
These true/false results could then be used as the ground truth to evaluate the automated methods.\
The verdict is that about ~75% are correct from all the translations. Now, we need a way to filter the false ones automatically.

<img src="https://github.com/ValachPatrik/dataengineering/assets/82080194/69ef1ad9-fb95-48f3-b093-7e44b71411ea" alt="image" width="300"/>

### LLM comparison - Failed Evaluation
Automated asking LLM to compare the semantic difference for a true/false matrix.\
The prompt template is generated automatically and then requires the user to compare the two questions to train llama to determine if two sentences have the same semantic meaning.
After that, the dataset and translations can be compared using this approach.

Unfortunately, the LLM comparison does not succeed in correctly comparing the sentences. Even though it gains a 0.6 F1 score, there are too many false negatives; thus, we observed it actually flagging only 33% of pairs correctly. This is a far cry from the 75% we determined by manual.

<img src="https://github.com/ValachPatrik/dataengineering/assets/82080194/65fdd856-e0e2-4416-8544-7d888d119f00" alt="image" width="362"/>
<img src="https://github.com/ValachPatrik/dataengineering/assets/82080194/86e6510d-5c5c-418e-aa9a-76e87b29c541" alt="image" width="300"/>

Room for improvement:
- Train llama for the comparison with more examples in the prompt
- Automatically generate the prompt by asking the model to give a set of semantically same sentences, potentially improving the comparison quality.

### BLEU - Failed Evaluation
Using BLEU from https://huggingface.co/spaces/evaluate-metric/bleu \
It gives a value between [0,1] to measure how many transformations are needed to create the same sentence.

Min: 0.0 \
Max: 0.8153551038173115 \
Average: 0.08747699091088278 

<img src="https://github.com/ValachPatrik/Structured-Query-Verbalization-with-Large-Language-Models/assets/82080194/3ef0c3db-4d31-447f-b354-42b77270da23" alt="image" width="300"/>

The BLEU metric has not correctly identified which ones are the same.

Room for improvement:
- Change up the weights

### BERT - Winner
Using BLEU from https://huggingface.co/docs/transformers/en/model_doc/bert \
Converts text to a vector embedding. \
Semantically indifferent text will give embeddings that are close to each other. \
It gives a value between [0,1] to measure this closeness.

Min: 0.41958645 \
Max: 0.9939645 \
Average: 0.8257805704999999 \
Median: 0.8335725700000001 \
Mode: 0.41958645 

<img src="https://github.com/ValachPatrik/Structured-Query-Verbalization-with-Large-Language-Models/assets/82080194/81ca9710-339f-4e06-976d-14eb5ac34e05" alt="image" width="300"/>

### Display Statistics
Code provides statistics for each evaluation and creates graphs for BERT, BLEU, and the final subset of filtered manual evaluation.

## Final Filtering
The final filter of translated prompts is done with the BERT metric. \
The exact value has been set to 0.87 to eliminate all incorrectly translated prompts and ensure we don't create any faulty data.\
The final number of examples is approximately a third of the number of translated prompts.\
The threshold could be lowered to increase this number of outputted translations, which would also introduce occasional faulty examples.\
Unless another model or bigger prompt is used to move the threshold down while maintaining accuracy.

![image](https://github.com/ValachPatrik/Structured-Query-Verbalization-with-Large-Language-Models/assets/82080194/e22b440b-c0df-48fb-91cf-7b91fa1cb7f1)
![image](https://github.com/ValachPatrik/Structured-Query-Verbalization-with-Large-Language-Models/assets/82080194/2a4f0911-6841-4e70-a7cd-53093d44ec2b)



## How to use
If you use llama3 as we did, you must run it locally (https://ollama.com/library/llama3) or provide a different API endpoint.\
After that, run the script and follow its instructions to create the final filtered dataset.\
Remove the example CSV and TXT files to start fresh without our data run.\
Each file is a progress save in the creation of the final dataset.\
Change the parameters at the bottom of the script to suit your needs.\
Modify what is needed for your use case (dataset, columns, model, number of examples, prompts, filter metric,...)
