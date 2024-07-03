# STRUCTURED QUERY VERBALIZATION WITH LARGE LANGUAGE MODELS
This project has been developed by Patrik Valach & Louisa Siebel under the supervision of Tim Schwabe in the Technical University of Munich (TUM) course Data Engineering led by Prof. Maribel Acosta.

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
The prompt is automatically generated from the last <200> rows of the dataset to showcase the model how to respond properly. This can be increased further to make the model potentially perform correctly for the bigger portion of the dataset. However, in between our computing resource constraints and the effectiveness of the prompt, we have landed at the sweet spot of 200 examples.

The model used: llama3 8b
https://ollama.com/library/llama3

We selected a free model we could run locally, llama3 8b.
The prompt is called on the dataset; we have set the limit to 200 in the code due to computing restraints. 

Room for improvement:
- Use a more powerful model like llama3 70b
- Increase the prompt size

![image](https://github.com/ValachPatrik/dataengineering/assets/82080194/590921ee-fa2d-4244-a4de-602a693b1862)

## Evaluation
We used 4 different evaluating methods and selected the best one to filter out to only correct results that can be used to train LLMs.

### Manual Evaluation
A manual evaluation is created to judge other evaluations to be able to tell 100% how our metrics perform.
We did the first 40 examples.
The user is asked if two sentences are semantically the same and thus can be used to train LLMs.
The verdict is that about ~75% are correct from all the translations. Now, we need a way to filter the false ones automatically.

<img src="https://github.com/ValachPatrik/dataengineering/assets/82080194/69ef1ad9-fb95-48f3-b093-7e44b71411ea" alt="image" width="300"/>

### LLM comparison - Failed Evaluation
The same logic as in manual evaluation is used, but now it's automated using LLM to scale.
The prompt template is generated automatically and then requires the user to compare the two to train llama to determine if two sentences have the same semantic meaning.
After that, the dataset and translations can be compared using this approach.

Unfortunately, the LLM comparison does not succeed in correctly comparing the sentences. Even though it gains a 0.6 F1 score, there are too many false negatives; thus, we observed it actually flagging only 33% correctly. This is a far cry from the 75% we determined by manual.

<img src="https://github.com/ValachPatrik/dataengineering/assets/82080194/65fdd856-e0e2-4416-8544-7d888d119f00" alt="image" width="362"/>
<img src="https://github.com/ValachPatrik/dataengineering/assets/82080194/86e6510d-5c5c-418e-aa9a-76e87b29c541" alt="image" width="300"/>

Room for improvement:
- Train llama for the comparison with more examples in the prompt
- Automatically generate the prompt by asking the model to give a set of semantically same sentences, potentially improving the comparison quality.

### BERT - Winner
Using BLEU from https://huggingface.co/docs/transformers/en/model_doc/bert \
Converts text to vector \
Semantically indifferent text will give vectors that are close to each other \

Min: 0.41958645 \
Max: 0.9939645 \
Average: 0.8257805704999999 \
Median: 0.8335725700000001 \
Mode: 0.41958645 \
<img src="https://github.com/ValachPatrik/dataengineering/assets/82080194/a25418b2-b630-435d-8cf5-a7e7ccf237e0" alt="image" width="300"/>

### BLEU - Failed Evaluation
Using BLEU from https://huggingface.co/spaces/evaluate-metric/bleu \

Min: 0.0 \
Max: 0.8153551038173115 \
Average: 0.08747699091088278 \

<img src="https://github.com/ValachPatrik/dataengineering/assets/82080194/adb6629a-9f44-4011-b79c-b2dce23c8e1b" alt="image" width="300"/>

The BLEU metric has not correctly identified which ones are the same.

Room for improvement:
- Change up the weights

## Final Filtering
The final filter of translated prompts is done with the BERT metric. \
The exact value has been set to 0.88 to eliminate all incorrectly translated prompts and ensure we don't create faulty data.\
The final amount is approximately halved from the number of translated prompts, but that is the cost of having high quality.\
It could be lowered to increase the amount, but that would also introduce occasional faulty examples.\
Also, training the translation LLM better or choosing a stronger model could help reduce this threshold.

## Contact
Patrik Valach patrik.valach@tum.de

Louisa Siebel ge94seq@mytum.de
