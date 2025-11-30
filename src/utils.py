import json
import wandb
import os
import pickle as pkl
import pandas as pd
from json_repair import repair_json
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import textwrap
from IPython.display import display, HTML
from urllib.parse import quote
import logging
from jsonschema import validate, ValidationError
from google import genai

PICKLE_ALLOWED_CLASSES = [bool, int, float, complex, str, bytes, bytearray, list, tuple, dict, set, frozenset, np.ndarray]

#region Eval metric functions
def compute_ndcg(sorted_preds: list, gold: list, k: int = 10) -> float:
    if not sorted_preds or not gold:
        return 0.0

    sorted_preds = sorted_preds[:k]
    dcg = 0
    for i, pred_item in enumerate(sorted_preds):
        if pred_item in gold:
            # Assign a relevance of 1 if the item is in the gold list
            relevance = 1
            # The rank is i + 1, and we use log2(rank + 1) for the discount
            dcg += relevance / np.log2(i + 2)

    # Calculate IDCG
    # The ideal ranking would have all relevant items from the gold set at the top.
    num_relevant_items = len(gold)
    # The number of items to consider for IDCG is the minimum of k and the number of relevant items.
    ideal_k = min(k, num_relevant_items)

    idcg = 0
    for i in range(ideal_k):
        # In the ideal case, all top items have a relevance of 1
        relevance = 1
        idcg += relevance / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg

def compute_recall(sorted_preds: list, gold: list, k: int = 10) -> float:
    if not sorted_preds or not gold:
        return 0.0

    sorted_preds = sorted_preds[:k]
    num_relevant_items = len(gold)
    num_relevant_retrieved = sum([1 for pred_item in sorted_preds if pred_item in gold])
    return num_relevant_retrieved / num_relevant_items
#endregion

def chain_path_rel_fn(cur_rel, path_rel, relevance_chain_factor):
    return ((cur_rel*relevance_chain_factor) + (path_rel*(1-relevance_chain_factor))) if cur_rel > 0.09 else cur_rel

def get_all_leaf_nodes_with_path(node, path=[]):
  if len(node.child) == 0:
    return [(node, path)]
  else:
    ret = []
    for i, child in enumerate(node.child):
      ret += get_all_leaf_nodes_with_path(child, path + [i])
    return ret

def get_node_id(id, docs_df):
  try:
    if isinstance(id, str):
      if id.startswith('['):
        return id.split(' ', 1)[1]
      else:
        return id
    elif isinstance(id, int) or isinstance(id, np.int64) or isinstance(id, np.int32):
      return docs_df.id.iloc[id]
    else:
      raise Exception(f'Unknown id type {type(id)}')
  except Exception as e:
    print(f'Exception {e} for id {id}')
    return None

#region Saving and loading results helper functions
def save_exp(RESULTS_DIR, hp, llm_api, eval_samples, all_eval_metric_dfs, allow_overwrite=False, save_llm_api_history=False):
  def sanitize_dict(d):
    if isinstance(d, dict):
      return {k: sanitize_dict(v) for k, v in d.items() if any([isinstance(v, allowed) for allowed in PICKLE_ALLOWED_CLASSES])}
    elif isinstance(d, list):
      return [sanitize_dict(v) for v in d if any([isinstance(v, allowed) for allowed in PICKLE_ALLOWED_CLASSES])]
    else:
      return d

  eval_dump_path = f'{RESULTS_DIR}/all_eval_sample_dicts-{hp}.pkl'
  eval_metrics_dump_path = f'{RESULTS_DIR}/all_eval_metrics-{hp}.pkl'
  llm_api_history_dump_path = f'{RESULTS_DIR}/llm_api_history-{hp}.pkl'

  all_eval_sample_dicts = [sanitize_dict(sample.to_dict()) for sample in eval_samples]
  if os.path.exists(eval_dump_path) and (not allow_overwrite):
    user_input = input(f'Dump path exists, override (y/n)?')
    assert user_input.lower() == 'y'

  pkl.dump(all_eval_sample_dicts, open(eval_dump_path, 'wb'))
  # print(f'Saved predictions to {eval_dump_path}')

  pd.concat(all_eval_metric_dfs, axis=1, keys=[f'Iter {i}' for i in range(len(all_eval_metric_dfs))]).to_pickle(eval_metrics_dump_path)
  # print(f'Saved metrics to {eval_metrics_dump_path}')

  if save_llm_api_history:
    pkl.dump(llm_api.history, open(llm_api_history_dump_path, 'wb'))

def load_exp(RESULTS_DIR, hp, semantic_root_node, node_registry, logger, hp_str=None):
  if hp_str is None:
    hp_str = str(hp)
  from tree_objects import InferSample
  eval_dump_path = f'{RESULTS_DIR}/all_eval_sample_dicts-{hp_str}.pkl'
  eval_metrics_dump_path = f'{RESULTS_DIR}/all_eval_metrics-{hp_str}.pkl'

  if not os.path.exists(eval_dump_path) or not os.path.exists(eval_metrics_dump_path):
     logger.warning(f'No existing dump found at {eval_dump_path} or {eval_metrics_dump_path}, starting fresh')
     return [], []

  all_eval_sample_dicts = pkl.load(open(eval_dump_path, 'rb'))
  eval_samples = [InferSample(semantic_root_node, node_registry, hp, logger, excluded_ids_set=d.get('excluded_ids_set', None)).load_dict(d) for d in all_eval_sample_dicts]
  
  for sample in eval_samples:
     sample.post_load_processing()

  # Load the concatenated DataFrame and split it back into a list
  concatenated_df = pd.read_pickle(eval_metrics_dump_path)
  
  # Extract individual DataFrames from the multi-level columns
  all_eval_metric_dfs = []
  if hasattr(concatenated_df.columns, 'levels'):  # Check if it has MultiIndex columns
    for iter_key in concatenated_df.columns.levels[0]:
      df = concatenated_df[iter_key]
      all_eval_metric_dfs.append(df)
  else:
    # Fallback if it's not a MultiIndex (single iteration case)
    all_eval_metric_dfs = [concatenated_df]

  return eval_samples, all_eval_metric_dfs
#endregion

#region Setup logging
def setup_logger(name, log_file_name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    while logger.handlers:
        handler = logger.handlers.pop()
        logger.removeHandler(handler)
        handler.close()

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if os.path.exists(log_file_name):
        print(f"Log file already exists: {os.path.abspath(log_file_name)}, appending to it.")
    else:
        print(f"Creating new log file: {os.path.abspath(log_file_name)}")
    file_handler = logging.StreamHandler(open(log_file_name, 'a'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def init_wandb_logging(hp, results_dir, mode_override=None):
    """Initialize wandb logging with project configuration."""
    wandb.init(
        project="lattice-retrieval",
        name=f"{hp.SUBSET}_{hp.LLM}_{hp.TREE_VERSION}_{hp.SUFFIX}",
        config=vars(hp),
        dir=results_dir,
        mode=mode_override,
    )
    return wandb.run.name

def wandb_log_iteration_metrics(eval_metric_df, iteration):
    """Log metrics for a single iteration with proper step tracking."""
    # Log mean metrics for this iteration with iteration as the step
    metrics_dict = {}
    for k in eval_metric_df.columns:
        metrics_dict[k] = eval_metric_df[k].mean()
    
    # Log with iteration as the step to create proper x-axis progression
    wandb.log(metrics_dict, step=iteration)

def wandb_log_reranking_metrics(rerank_eval_metric_df, step=None):
    """Log reranking evaluation metrics as a table."""
    # Create table data with metric names and their mean values
    table_data = []
    for k in rerank_eval_metric_df.columns:
      table_data.append([k, rerank_eval_metric_df[k].mean()])
    
    # Create wandb table
    rerank_table = wandb.Table(
      columns=["metric", "value"],
      data=table_data
    )
    
    # Log table with step if provided
    if step is not None:
      wandb.log({"reranking_metrics": rerank_table}, step=step)
    else:
      wandb.log({"reranking_metrics": rerank_table})

def wandb_log_final_summary(all_eval_samples, final_step=None):
    """Log final summary metrics and sample table."""
    pass
    # final_metrics = pd.DataFrame([sample.compute_eval_metrics(k=10) for sample in all_eval_samples])
    
    # # Log final metrics with a clear prefix
    # summary_metrics = {}
    # for k in final_metrics.columns:
    #     summary_metrics[f"final_{k}"] = final_metrics[k].mean()
    
    # # Use final_step if provided to maintain step consistency
    # if final_step is not None:
    #     wandb.log(summary_metrics, step=final_step)
    # else:
    #     wandb.log(summary_metrics)
    
    # # Log evaluation samples as wandb table for analysis
    # eval_table_data = []
    # for i, sample in enumerate(all_eval_samples[:10]):  # Log first 10 samples
    #     eval_table_data.append([
    #         i,
    #         sample.query[:100],  # Truncate query for readability
    #         len(sample.gold_paths),
    #         len(sample.get_top_predictions(10))
    #     ])
    
    # eval_table = wandb.Table(
    #     columns=["sample_id", "query", "num_gold_paths", "num_predictions"],
    #     data=eval_table_data
    # )
    # wandb.log({"evaluation_samples": eval_table})

def finish_wandb_logging(logger):
    """Finish wandb run and log completion."""
    wandb.finish()
    logger.info("Wandb run finished")
#endregion

#region JSON utility functions
def recursive_key_search(obj, key):
  if isinstance(obj, dict):
    if key in obj:
      return obj[key]
    for k, v in obj.items():
      result = recursive_key_search(v, key)
      if result is not None:
        return result
  if isinstance(obj, list):
    for item in obj:
      result = recursive_key_search(item, key)
      if result is not None:
        return result
  return None

import re
def post_process(output, return_json=False, extract_key=None):
  try:
    if return_json:
      try:
        return json.loads(output)
      except json.JSONDecodeError:
        pass

    output_text = output['response'] if isinstance(output, dict) else output
    if '```json' in output_text:
      output_text = re.findall(r"(?:```json\s*)(.+)(?:```)", output_text, re.DOTALL)[-1]
    output_json = repair_json(output_text, return_objects=True)
    if return_json: return output_json
    output_text = recursive_key_search(output, extract_key)
    return output_text
  except KeyError as k:
    print(f'Encountered error {k} in post processing: {output}')
    return None

def validate_genai_response_constraint(response_text, constraint):
    try:
        # Convert GenAI schema to JSON Schema (only lowercase type values)
        if isinstance(constraint, genai.types.Schema):
          constraint = constraint.to_json_dict()

        schema_str = json.dumps(constraint)
        schema_str = schema_str.replace('"STRING"', '"string"').replace('"INTEGER"', '"integer"').replace('"ARRAY"', '"array"').replace('"OBJECT"', '"object"').replace('"NUMBER"', '"number"').replace('"BOOLEAN"', '"boolean"')
        schema = json.loads(schema_str)
        
        # Parse and validate
        data = json.loads(response_text)
        validate(instance=data, schema=schema)
        return True, None
    except (json.JSONDecodeError, ValidationError) as e:
        return False, str(e)
#endregion

def compute_node_registry(semantic_root_node):
    def compute_node_path(node, path = ()):
        node.path = path
        for idx, child in enumerate(node.child):
            compute_node_path(child, (*path, idx))
    compute_node_path(semantic_root_node)

    def compute_node_num_leaves(node):
        if (not node.child) or (len(node.child) == 0):
            node.num_leaves = 1
        else:
            node.num_leaves = sum([compute_node_num_leaves(child) for child in node.child])
        return node.num_leaves
    compute_node_num_leaves(semantic_root_node)

    def compute_node_registry(node, node_registry = []):
        node_registry.append(node)
        for child in node.child:
            compute_node_registry(child, node_registry)
        return node_registry

    node_registry = []
    node_registry_inv_map = {}
    node_registry = compute_node_registry(semantic_root_node, node_registry)
    for idx, node in enumerate(node_registry):
        node.registry_idx = idx
        node_registry_inv_map[node.path] = idx

    node_parent_registry_idx_map = [-1 for _ in range(len(node_registry))]
    for node in node_registry:
        if node.child:
            for child in node.child:
                node_parent_registry_idx_map[child.registry_idx] = node.registry_idx
    return node_registry

#region Prediction tree visualization utils
def wrap_text_for_plotly(text, width=120):
    """
    Wraps a long string to a specified width and formats it for
    Plotly hover labels by replacing newlines with <br> tags.
    """
    if not text:
        return ""
    lines = []
    for single_line in text.split('\n'):
        wrapped_lines = textwrap.wrap(single_line, width=width, break_long_words=True)
        if not wrapped_lines:
            lines.append('')
        else:
            lines.extend(wrapped_lines)
    return '<br>'.join(lines)

def visualize_sample(sample, width=1400, height=1000, save_path=None, max_step=1e6):
    """
    Creates an interactive Plotly visualization with query text and highlighted gold paths.
    """
    root_node = sample.prediction_tree
    gold_paths = sample.gold_paths
    excluded_ids_set = sample.excluded_ids_set
    query_text = sample.query[:400].replace('\n', '  ') + ('...' if len(sample.query) > 1000 else '')
    # --- 1. Pre-process gold paths to identify nodes and edges for highlighting ---
    gold_node_paths = set()
    gold_edges = set()
    if gold_paths:
        for path in gold_paths:
            current_path_tuple = ()
            for index in path:
                # The path of the child node
                child_path_tuple = (*current_path_tuple, index)
                # Add the node to the set of gold nodes
                gold_node_paths.add(child_path_tuple)
                # Add the edge connecting the parent to this node
                gold_edges.add((current_path_tuple, child_path_tuple))
                # Move to the next level
                current_path_tuple = child_path_tuple

    # --- 2. Traverse the tree to gather data for plotting ---
    # Coordinates and styles for regular vs. gold edges
    reg_edge_x, reg_edge_y = [], []
    gold_edge_x, gold_edge_y = [], []
    red_edge_x, red_edge_y = [], []

    # Node properties
    node_x, node_y = [], []
    node_hover_text = []
    node_color = []
    node_border_color = []
    node_border_width = []
    node_symbols = []
    node_indices_text = []

    positions = {}
    level_height = 100  # Vertical distance between levels

    # First, find all nodes at each level
    levels = []
    if root_node:
        level = [root_node]
        while level:
            levels.append(level)
            next_level = []
            for node in level:
                if node.child:
                    for c in node.child:
                        if c.creation_step <= max_step:
                        # if True:
                            next_level.append(c)
            level = next_level

    # Now, calculate positions level by level with full-width distribution
    y_pos = 0
    # Use half of the plot width for calculations to center at x=0
    plot_half_width = width / 2.2
    for level in levels:
        num_nodes_level = len(level)
        for i, node in enumerate(level):
            if node.creation_step <= max_step:
            # if True:
                if num_nodes_level == 1:
                    # A single node on a level is always centered
                    x_pos = 0
                else:
                    # Distribute nodes evenly from -plot_half_width to +plot_half_width
                    fraction = i / (num_nodes_level - 1) # A value from 0.0 to 1.0
                    x_pos = (fraction * 2 - 1) * plot_half_width # Map to [-plot_half_width, +plot_half_width]
                positions[node.path] = (x_pos, -y_pos)
        y_pos += level_height

    # Build plot elements
    queue = [root_node]
    visited_paths = {root_node.path}
    while queue:
        node = queue.pop(0)
        if node.path not in positions: continue
        if node.creation_step > max_step: continue

        x, y = positions[node.path]
        node_x.append(x)
        node_y.append(y)

        node_border_color.append('#FFD700' if node.path in gold_node_paths else 'darkgrey')
        node_border_width.append(2)

        node_symbols.append('diamond' if node.is_leaf else 'circle')
        node_indices_text.append('R' if not node.path else str(node.path[-1]))

        wrapped_desc = wrap_text_for_plotly(node.desc)
        wrapped_reasoning = wrap_text_for_plotly(str(node.reasoning) if not isinstance(node.reasoning, list) else '\n'.join(map(str, node.reasoning)))
        child_relevances = sorted([(i, v) for i, v in enumerate(node.child_relevances)], key=lambda x: x[1], reverse=True) if node.child_relevances else []
        child_relevances = wrap_text_for_plotly('; '.join([f'{i}: {v}' for i, v in child_relevances]))
        if node.is_leaf or node.path_relevance > 0:
          calibrated_relevance = getattr(node, 'calibrated_relevance', None)
          node.overall_relevance = float(chain_path_rel_fn(max(0, node.calibrated_relevance), node.parent.path_relevance, 0.5)) if hasattr(node, 'calibrated_relevance') and node.parent else float(node.path_relevance)
          node_hover_text.append(
              f"<b>Path:</b> {node.path}<br>"
              f"<b>Path Relevance:</b> {node.overall_relevance:.3f}<br>"
            #   f"<b>Path Relevance:</b> {node.path_relevance:.3f}<br>"
              f"<b>Calibrated Relevance:</b> {calibrated_relevance:.3f}<br>"
              f"<b>Local Relevance:</b> {node.local_relevance:.3f}<br>"
              f"<b>Child Relevance:</b> {child_relevances}<br><br>"
              f"<b>Description:</b><br>{wrapped_desc}<br><br>"
              f"<b>Reasoning:</b><br>{wrapped_reasoning}"
          )
        else:
          node_hover_text.append('')
        node_color.append(node.overall_relevance if hasattr(node, 'overall_relevance') else node.path_relevance)

        if node.child:
            for child_node in node.child:
                 if child_node.path not in visited_paths and child_node.path in positions:
                    child_x, child_y = positions[child_node.path]
                    edge_tuple = (node.path, child_node.path)
                    if edge_tuple in gold_edges:
                        gold_edge_x.extend([x, child_x, None])
                        gold_edge_y.extend([y, child_y, None])
                    elif isinstance(child_node.id, str) and bool(excluded_ids_set) and (child_node.id.split('] ', 1)[-1] in excluded_ids_set):
                        red_edge_x.extend([x, child_x, None])
                        red_edge_y.extend([y, child_y, None])
                    else:
                        reg_edge_x.extend([x, child_x, None])
                        reg_edge_y.extend([y, child_y, None])
                    visited_paths.add(child_node.path)
                    queue.append(child_node)

    # --- 3. Create Plotly traces ---
    reg_edge_trace = go.Scatter(x=reg_edge_x, y=reg_edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    red_edge_trace = go.Scatter(x=red_edge_x, y=red_edge_y, line=dict(width=0.5, color='#FF0000'), hoverinfo='none', mode='lines')
    gold_edge_trace = go.Scatter(x=gold_edge_x, y=gold_edge_y, line=dict(width=2, color='#FFD700'), hoverinfo='none', mode='lines')
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        # Set mode to draw both markers and the text inside them
        mode='markers+text',
        # Use hovertext for the detailed tooltip
        hovertext=node_hover_text,
        hoverinfo='text',
        # Use text for the visible index number
        text=node_indices_text,
        # Center the text and set a readable font
        textposition='middle center',
        textfont=dict(
            family='sans serif',
            size=8,
            color='white'
        ),
        marker=dict(
            showscale=True, colorscale='YlGn', color=node_color, size=15,
            symbol=node_symbols,
            colorbar=dict(thickness=15, title=dict(text='Relevance')),
            line=dict(color=node_border_color, width=node_border_width)
        )
    )

    # --- 4. Assemble the figure and layout ---
    layout_annotations = []
    title = wrap_text_for_plotly(f'<b>Prediction Tree</b> for <b>Query</b>: {query_text}', width=200)
    fig = go.Figure(data=[reg_edge_trace, red_edge_trace, gold_edge_trace, node_trace],
                 layout=go.Layout(
                    width=width,
                    height=height,
                    title=dict(text=title, font=dict(size=12), x=0.05, xanchor='left', y=0.97, yanchor='top', pad=dict(t=0, b=0)),
                    showlegend=False, hovermode='closest',
                    margin=dict(b=5, l=5, r=100, t=80),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    annotations=layout_annotations,
                    # set background to white
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    hoverlabel=dict(
                        # bgcolor='white',
                        font_size=12,
                        align="left",
                    ),
                ))

    fig.show()
    # fig.write_image(f"{save_path.replace('html', 'png')}")

    html_string = fig.to_html(full_html=True, include_plotlyjs='cdn')
    if save_path:
        with open(save_path, 'w') as f:
            f.write(html_string)
        print(f'Saved plot HTML to {save_path}')
    encoded_html = quote(html_string)
    link = f'<a href="data:text/html;charset=utf-8,{encoded_html}" target="_blank">Click here to open plot in new tab</a>'
    display(HTML(link))
    return fig
#endregion