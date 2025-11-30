#region Imports
import numpy as np
import pandas as pd
import pickle as pkl
import asyncio
from tqdm.autonotebook import tqdm
import os
import logging
from typing import List
from datasets import load_dataset
from google.genai import types
from hyperparams import HyperParams
from tree_objects import SemanticNode, InferSample
from llm_apis import GenAIAPI, VllmAPI
from prompts import get_traversal_prompt_response_constraint, get_reranking_prompt
from utils import (
    setup_logger, 
    compute_node_registry,
    get_all_leaf_nodes_with_path, 
    get_node_id, 
    post_process, 
    save_exp, 
    load_exp,
    init_wandb_logging,
    finish_wandb_logging,
    wandb_log_iteration_metrics,
    wandb_log_reranking_metrics,
    wandb_log_final_summary,
)
np.random.seed(42)
#endregion

#region Setup
hp = HyperParams.from_args()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = f'{BASE_DIR}/results/{hp.DATASET}/{hp.SUBSET}/'
os.makedirs(RESULTS_DIR, exist_ok=True)
logger = setup_logger('lattice_runner', f"{RESULTS_DIR}/{hp}.log", logging.INFO)

# Initialize wandb logging
run_name = init_wandb_logging(hp, RESULTS_DIR)
logger.info(f"Initialized wandb run: {run_name}")
#endregion

#region Data loading
if os.path.exists(f'{BASE_DIR}/data/{hp.DATASET}/{hp.SUBSET}/documents.jsonl'):
    docs_df = pd.read_json(f'{BASE_DIR}/data/{hp.DATASET}/{hp.SUBSET}/documents.jsonl', lines=True, dtype={'id': str})
    examples_df = pd.read_json(f'{BASE_DIR}/data/{hp.DATASET}/{hp.SUBSET}/examples.jsonl', lines=True, dtype={'gold_ids': List[str]})
    examples_df['gold_ids'] = examples_df['gold_ids'].apply(lambda x: [str(i) for i in x])
else:
    docs_df = pd.DataFrame(load_dataset('xlangai/BRIGHT', 'documents', split=hp.SUBSET))
    examples_df = pd.DataFrame(load_dataset('xlangai/BRIGHT', 'examples', split=hp.SUBSET))
    
doc_id_to_content = {docs_df.iloc[i].id: docs_df.iloc[i].content for i in range(len(docs_df))}

tree_dict = pkl.load(open(f'{BASE_DIR}/trees/{hp.DATASET}/{hp.SUBSET}/tree-{hp.TREE_VERSION}.pkl', 'rb'))
semantic_root_node = SemanticNode().load_dict(tree_dict) if isinstance(tree_dict, dict) else tree_dict
node_registry = compute_node_registry(semantic_root_node)
all_leaf_nodes = get_all_leaf_nodes_with_path(semantic_root_node)
doc_id_to_path = {get_node_id(leaf.id, docs_df): path for leaf, path in all_leaf_nodes}
#endregion

#region Setup LLM API and Eval Samples
if hp.LLM_API_BACKEND == 'genai': 
    llm_api = GenAIAPI(hp.LLM, logger=logger, timeout=hp.LLM_API_TIMEOUT, max_retries=hp.LLM_API_MAX_RETRIES)
elif hp.LLM_API_BACKEND == 'vllm': 
    llm_api = VllmAPI(hp.LLM, logger=logger, timeout=hp.LLM_API_TIMEOUT, max_retries=hp.LLM_API_MAX_RETRIES, base_url=','.join([f"http://localhost:{8000+i}/v1" for i in range(4)]))
else: raise ValueError(f'Unknown LM API backend: {hp.LLM_API_BACKEND}')

llm_api_kwargs = {
    'max_concurrent_calls': hp.LLM_MAX_CONCURRENT_CALLS,
    'response_mime_type': 'application/json',
    'response_schema': get_traversal_prompt_response_constraint(bool(hp.REASONING_IN_TRAVERSAL_PROMPT)),
    'staggering_delay': hp.LLM_API_STAGGERING_DELAY,
    # 'temperature': 0.8,
    'thinking_config': types.ThinkingConfig(thinking_budget=hp.REASONING_IN_TRAVERSAL_PROMPT),
}

if hp.LLM_API_BACKEND == 'vllm':
    llm_api_kwargs.pop('response_mime_type')
    llm_api_kwargs.pop('thinking_config')
    llm_api_kwargs.pop('response_schema')

if hp.LOAD_EXISTING and os.path.exists(f'{RESULTS_DIR}/all_eval_sample_dicts-{hp}.pkl'):
  all_eval_samples, all_eval_metric_dfs = load_exp(RESULTS_DIR, hp, semantic_root_node, node_registry, logger)
  logger.info(f'Loaded existing experiment with {len(all_eval_samples)} eval samples and {len(all_eval_metric_dfs)} eval metric dfs')
  if len(all_eval_samples) > 0:
    eval_metric_df = pd.DataFrame([sample.compute_eval_metrics(k=10) for sample in all_eval_samples])
    logger.info('; '.join([f'{k}: {eval_metric_df[k].mean():.2f}' for k in eval_metric_df.columns]))
else: 
  all_eval_samples, all_eval_metric_dfs = [], []
  for i in range(min(examples_df.shape[0], hp.NUM_EVAL_SAMPLES)):
    gold_paths = [doc_id_to_path[doc_id] for doc_id in examples_df.iloc[i]['gold_ids'] if doc_id in doc_id_to_path]
    if len(gold_paths) < len(examples_df.iloc[i]['gold_ids']):
        logger.warning(f"Some gold IDs for example {i} not found in document paths.")
    sample = InferSample(
        semantic_root_node,
        node_registry,
        hp=hp,
        logger=logger,
        query=examples_df.iloc[i]['query'][:hp.MAX_QUERY_CHAR_LEN],
        gold_paths=gold_paths,
        excluded_ids_set=set(examples_df.iloc[i]['excluded_ids']),
        )
    all_eval_samples.append(sample)
  assert not any([sample.prediction_tree.excluded for sample in tqdm(all_eval_samples)])
  
logger.info('Hyperparams:\n'+'\n'.join([f'{k}:\t{v}' for k, v in vars(hp).items()]))
#endregion

#region Run Retrieval Loop
async def retrieval_loop_step():  # Make the function asynchronous
    inputs = [sample.get_step_prompts() for sample in all_eval_samples]
    indptr = np.cumsum([0, *[len(x) for x in inputs]])
    flat_inputs = [y for x in inputs for y in x]
    flat_prompts, flat_slates = list(zip(*flat_inputs))
    slates = [flat_slates[indptr[j]:indptr[j+1]] for j in range(len(inputs))]

    flat_responses = await llm_api.run_batch(flat_prompts, **llm_api_kwargs)
    flat_response_jsons = [post_process(output, return_json=True) for output in tqdm(flat_responses)]
    response_jsons = [flat_response_jsons[indptr[j]:indptr[j+1]] for j in range(len(inputs))]

    for sample, sample_slates, sample_response_jsons in tqdm(zip(all_eval_samples, slates, response_jsons), total=len(all_eval_samples), desc='Updating samples'):
      sample.update(sample_slates, sample_response_jsons)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
try:
    for i in tqdm(range(len(all_eval_metric_dfs), hp.NUM_ITERS)):
        logger.info(f'-------------------- Iter {i} --------------------')
        loop.run_until_complete(retrieval_loop_step())
        eval_metric_df = pd.DataFrame([sample.compute_eval_metrics(k=10) for sample in all_eval_samples])
        all_eval_metric_dfs.append(eval_metric_df)
        
        # Log metrics
        wandb_log_iteration_metrics(eval_metric_df, i)
        logger.info('; '.join([f'{k}: {eval_metric_df[k].mean():.2f}' for k in eval_metric_df.columns]))
        save_exp(RESULTS_DIR, hp, llm_api, all_eval_samples, all_eval_metric_dfs, allow_overwrite=True)  
        logger.info('-'*50)
finally:
    loop.close()
#endregion

#region Reranking (Optional)
async def rerank_predictions():
    logger.info('Starting reranking process...')
    
    def get_sample_rerank_prompt(sample):
        return get_reranking_prompt(sample.query, [x.desc for x, s in sample.get_top_predictions(100)], hp=hp, logger=logger, topk=10)

    def process_sample_rerank_response(sample, response):
        try:
          ranking = post_process(response, return_json=True)['ranking']
          top_preds = [x[0] for x in sample.get_top_predictions(100)]
          for rank, idx in enumerate(ranking):
              top_preds[idx].inverse_rank = 1/(rank+1)
        except Exception as e:
          logger.error(f'Error processing rerank response for query "{sample.query}": {e}')

    all_rerank_prompts, all_rerank_constraints = list(zip(*[get_sample_rerank_prompt(sample) for sample in all_eval_samples]))
    all_rerank_responses = await llm_api.run_batch(
        all_rerank_prompts, 
        max_concurrent_calls=hp.LLM_MAX_CONCURRENT_CALLS, 
        response_mime_type='application/json', 
        response_schema=all_rerank_constraints[0]
    )
    
    for sample, response in zip(all_eval_samples, all_rerank_responses):
        process_sample_rerank_response(sample, response)
    
    default_rel_fn = all_eval_samples[0].get_rel_fn(leaf=True)
    rerank_rel_fn = lambda x: (x.inverse_rank if hasattr(x, 'inverse_rank') else 0, default_rel_fn(x))
    rerank_eval_metric_df = pd.DataFrame([sample.compute_eval_metrics(k=10, rel_fn=rerank_rel_fn) for sample in all_eval_samples])
    
    # Log reranking metrics to wandb
    wandb_log_reranking_metrics(rerank_eval_metric_df)
    
    logger.info('After reranking: '+'; '.join([f'{k}: {rerank_eval_metric_df[k].mean():.2f}' for k in rerank_eval_metric_df.columns]))
    
    return rerank_eval_metric_df

# Run reranking if enabled
if hasattr(hp, 'RERANK') and hp.RERANK:
    rerank_eval_metric_df = asyncio.run(rerank_predictions())
    save_exp(RESULTS_DIR, hp, llm_api, all_eval_samples, all_eval_metric_dfs + [rerank_eval_metric_df], allow_overwrite=True)
else:
    logger.info('Reranking disabled, skipping...')

# Log final summary metrics and finish wandb run
if all_eval_metric_dfs and all_eval_samples:
    wandb_log_final_summary(all_eval_samples)

finish_wandb_logging(logger)
#endregion