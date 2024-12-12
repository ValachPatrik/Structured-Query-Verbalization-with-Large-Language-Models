---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:12000
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: SELECT ?obj WHERE { wd:Q213417 p:P115 ?s . ?s ps:P115 ?obj . ?s
    pq:P582 ?x filter(contains(YEAR(?x),'1922')) }
  sentences:
  - What city does Curtis Institute of Music reside in, and has wards in Cameron County?
  - What is  home place  of  New York Yankees  that is  end time  is 1922 - 0 - 0
    ?
  - What is William of Rubruck's date of birth?
- source_sentence: SELECT ?obj WHERE { wd:Q131252 p:P112 ?s . ?s ps:P112 ?obj . ?s
    pq:P3831 wd:Q1968442 }
  sentences:
  - '"Who is {victim} of {Battle of Ramillies}, which has {time zone} is {Central
    European Time} ?"'
  - Is the damage cost of the Tulsa Massacre 30000000?
  - What's the taxon source of Sichuan pepper?
- source_sentence: 'SELECT DISTINCT ?sbj ?sbj_label WHERE { ?sbj wdt:P31 wd:Q1307214
    . ?sbj rdfs:label ?sbj_label . FILTER(CONTAINS(lcase(?sbj_label), ''unicameralism''))
    . FILTER (lang(?sbj_label) = ''en'') } LIMIT 25 '
  sentences:
  - What is the asteroid with the lowest rotation period whose site of astronomical
    discovery is Bishop Observatory ?
  - What award did Ken Thompson receive on January 1st, 1983?
  - What are the form of government which start with the letter unicameralism
- source_sentence: SELECT ?answer WHERE { wd:Q855289 wdt:P641 ?X . ?X wdt:P2283 ?answer}
  sentences:
  - How many birth name are for Sarah Bernhardt?
  - B. B. King died from a stroke in his brain.
  - IS THE G FACTOR OF PROTON EQUALS 5.585694713
- source_sentence: SELECT (COUNT(?obj) AS ?value ) { wd:Q209050 wdt:P937 ?obj }
  sentences:
  - What is the monomer of polyvinyl chloride
  - How is the Gospel of John exemplar?
  - What is an electrical connector system that is published by USB Implementers Forum
    with the word "usb" in its name?
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy
- cosine_accuracy_threshold
- cosine_f1
- cosine_f1_threshold
- cosine_precision
- cosine_recall
- cosine_ap
model-index:
- name: SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: binary-classification
      name: Binary Classification
    dataset:
      name: test binary eval
      type: test-binary-eval
    metrics:
    - type: cosine_accuracy
      value: 0.85
      name: Cosine Accuracy
    - type: cosine_accuracy_threshold
      value: 0.4129094183444977
      name: Cosine Accuracy Threshold
    - type: cosine_f1
      value: 0.8611111111111112
      name: Cosine F1
    - type: cosine_f1_threshold
      value: 0.4129094183444977
      name: Cosine F1 Threshold
    - type: cosine_precision
      value: 0.8017241379310345
      name: Cosine Precision
    - type: cosine_recall
      value: 0.93
      name: Cosine Recall
    - type: cosine_ap
      value: 0.9166469813448259
      name: Cosine Ap
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision fa97f6e7cb1a59073dff9e6b13e2715cf7475ac9 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'SELECT (COUNT(?obj) AS ?value ) { wd:Q209050 wdt:P937 ?obj }',
    'What is an electrical connector system that is published by USB Implementers Forum with the word "usb" in its name?',
    'How is the Gospel of John exemplar?',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Binary Classification

* Dataset: `test-binary-eval`
* Evaluated with [<code>BinaryClassificationEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.BinaryClassificationEvaluator)

| Metric                    | Value      |
|:--------------------------|:-----------|
| cosine_accuracy           | 0.85       |
| cosine_accuracy_threshold | 0.4129     |
| cosine_f1                 | 0.8611     |
| cosine_f1_threshold       | 0.4129     |
| cosine_precision          | 0.8017     |
| cosine_recall             | 0.93       |
| **cosine_ap**             | **0.9166** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 12,000 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                          | sentence_1                                                                        | label                                                          |
  |:--------|:------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                              | string                                                                            | float                                                          |
  | details | <ul><li>min: 24 tokens</li><li>mean: 49.27 tokens</li><li>max: 104 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 16.46 tokens</li><li>max: 44 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.52</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                            | sentence_1                                                                              | label            |
  |:------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------|:-----------------|
  | <code> select distinct ?sbj where { ?sbj wdt:P195 wd:Q641 . ?sbj wdt:P31 wd:Q860861 } </code>         | <code>Who written the prequel of The Structures of Everyday Life?</code>                | <code>0.0</code> |
  | <code>SELECT ?answer WHERE { wd:Q205662 wdt:P37 ?X . ?X wdt:P282 ?answer}</code>                      | <code>Who is the president and CEO of BP?</code>                                        | <code>0.0</code> |
  | <code>SELECT ?ans_1 ?ans_2 WHERE { wd:Q389735 wdt:P2293 ?ans_1 . wd:Q389735 wdt:P2176 ?ans_2 }</code> | <code>What is {played by] of {computer network} that {painting of} is {network}?</code> | <code>0.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss | test-binary-eval_cosine_ap |
|:------:|:----:|:-------------:|:--------------------------:|
| 0.124  | 93   | -             | 0.7887                     |
| 0.248  | 186  | -             | 0.8644                     |
| 0.372  | 279  | -             | 0.8926                     |
| 0.496  | 372  | -             | 0.8968                     |
| 0.62   | 465  | -             | 0.9090                     |
| 0.6667 | 500  | 0.1687        | -                          |
| 0.744  | 558  | -             | 0.9143                     |
| 0.868  | 651  | -             | 0.9157                     |
| 0.992  | 744  | -             | 0.9167                     |
| 1.0    | 750  | -             | 0.9166                     |


### Framework Versions
- Python: 3.11.5
- Sentence Transformers: 3.3.0
- Transformers: 4.46.2
- PyTorch: 2.5.1+cpu
- Accelerate: 1.1.1
- Datasets: 3.1.0
- Tokenizers: 0.20.3

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->