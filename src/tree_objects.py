import numpy as np
from functools import partial
from json_repair import repair_json
from prompts import get_traversal_prompt, get_reranking_prompt
from utils import compute_ndcg, compute_recall, recursive_key_search, chain_path_rel_fn
from calib_utils import CalibModel, get_bimodal_gmm_intrsxn

class SemanticNode:
    desc: str = ''
    id = None
    embs = None
    child: list = []

    def __init__(self, id=None, desc='', child = [], embs = None):
        self.desc = desc
        self.child = child
        self.id = id
        self.embs = embs

    @property
    def is_leaf(self):
        return self.num_children == 0

    @property
    def num_children(self):
        return len(self.child)

    def get_all_leaf_nodes(self):
        if self.is_leaf:
            return [self]
        else:
            leaf_nodes = []
            for child in self.child:
                leaf_nodes.extend(child.get_all_leaf_nodes())
            return leaf_nodes

    def to_dict(self):
        return {
            **{k: v for k, v in self.__dict__.items() if k != 'child'},
            'child': [x.to_dict() for x in self.child] if self.child else None,
        }

    def load_dict(self, d):
        self.__dict__.update(**{k: v for k, v in d.items() if k != 'child'})
        if ('child' in d) and (d['child'] is not None):
            self.child = [SemanticNode().load_dict(x) for x in d['child']]
        return self

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        SEP = '\n\n'
        return f'ID: {self.id}, Num children: {len(self.child)}, Description: {self.desc}{SEP}First 4 Children:{SEP}{SEP.join(["[" + str(i) + ", " + str(x.id) + ", " + str(len(x.child)) + " children] " + x.desc for i, x in enumerate(self.child[:4])])}'

def is_excluded(node, excluded_ids_set):
    return isinstance(node.id, str) and (node.id.split('] ', 1)[-1].strip() in excluded_ids_set)

class MaskedSemanticNode:
    _excluded = None
    _child = None

    def __init__(self, node, excluded_ids_set = set()):
        self.semantic_node = node
        self.excluded_ids_set = excluded_ids_set

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        SEP = '\n\n'
        return f'ID: {self.id}, Num children: {len(self.child)}, Description: {self.desc}{SEP}First 4 Children:{SEP}{SEP.join(["[" + str(i) + ", " + str(x.id) + ", " + str(len(x.child)) + " children] " + x.desc for i, x in enumerate(self.child[:4])])}'

    def get_all_leaf_nodes(self):
        if self.is_leaf:
            return [self]
        else:
            leaf_nodes = []
            for child in self.child:
                leaf_nodes.extend(child.get_all_leaf_nodes())
            return leaf_nodes

    @property
    def num_leaves(self):
        return self.semantic_node.num_leaves

    @property
    def child(self):
        if self._child is None:
            self._child = [MaskedSemanticNode(x, self.excluded_ids_set) for x in self.semantic_node.child]
        return self._child

    @property
    def excluded(self):
        if self._excluded is None:
            # all_children_excluded = (self.num_children > 0) and (all([child.excluded for child in self.child]))
            # all_children_excluded = (self.num_children > 0) and (all([is_excluded(child, self.excluded_ids_set) for child in self.semantic_node.child]))
            self._excluded = is_excluded(self.semantic_node, self.excluded_ids_set)
            if self._excluded:
               return self._excluded

            for child in self.child:
                if not child.excluded:
                    self._excluded = False
                    break

        return self._excluded

    @property
    def desc(self):
        return self.semantic_node.desc if not self.excluded else 'No Description.'

    @property
    def id(self):
        return self.semantic_node.id

    @property
    def registry_idx(self):
        return self.semantic_node.registry_idx

    @property
    def num_children(self):
        return len(self.child)

    @property
    def embs(self):
        self.semantic_node.embs

    @property
    def is_leaf(self):
        return self.num_children == 0
  
class PredictionNode(object):
  def __init__(self, semantic_node, parent=None, local_relevance=0, path=None, creation_step=None, excluded_ids_set = set(), relevance_chain_factor=None):
    self.semantic_node = semantic_node
    self.parent = parent
    self.excluded = semantic_node.excluded if hasattr(semantic_node, 'excluded') else is_excluded(semantic_node, excluded_ids_set)
    self.local_relevance = local_relevance * (not self.excluded)
    self.calibrated_relevance = self.local_relevance
    self.relevance_chain_factor = relevance_chain_factor
    self.path_relevance = chain_path_rel_fn(self.local_relevance, self.parent.path_relevance, self.relevance_chain_factor) if self.parent else self.local_relevance
    self.path = path
    self.excluded_ids_set = excluded_ids_set
    self.creation_step = creation_step

    # To be initialized later in instantiate_children
    self.child = None
    self.child_relevances = None
    self.reasoning = ''

    self.SAVE_LIST = ['excluded', 'relevance_chain_factor', 'local_relevance', 'calibrated_relevance', 'path_relevance', 'path', 'creation_step', 'child_relevances', 'reasoning']

  def __str__(self):
    string = f'Path: {self.path} || Path Relevance: {self.path_relevance} || Local Relevance: {self.local_relevance} || Calibrated Relevance: {self.calibrated_relevance} || Predicted: {self.predicted} || Description: {self.desc[:100]}'
    if self.predicted:
      string += f'\nChild Relevances: {self.child_relevances}'
    return string

  def __repr__(self):
    return self.__str__()

  def __lt__(self, other):
        return self.path_relevance < other.path_relevance

  @property
  def num_children(self):
    return len(self.semantic_node.child)

  @property
  def num_leaves(self):
    return self.semantic_node.num_leaves

  @property
  def is_leaf(self):
    return self.num_children == 0

  @property
  def registry_idx(self):
    return self.semantic_node.registry_idx

  @property
  def desc(self):
    return self.semantic_node.desc

  @property
  def predicted(self):
    return (self.child_relevances is not None)

  @property
  def child_desc(self):
    return [x.desc for x in self.semantic_child]

  @property
  def semantic_child(self):
    return self.semantic_node.child

  # call after making prediction
  def instantiate_children(self, child_relevances, reasoning, creation_step):
    assert self.child is None
    self.reasoning = reasoning
    self.child_relevances = child_relevances
    self.child = []
    for i, child in enumerate(self.semantic_node.child):
      self.child.append(PredictionNode(
          semantic_node=child,
          local_relevance=child_relevances[i],
          path=(*self.path, i),
          creation_step=creation_step,
          excluded_ids_set=self.excluded_ids_set,
          parent=self,
          relevance_chain_factor=self.relevance_chain_factor))

  def to_dict(self):
    return {
        **{k: v for k, v in self.__dict__.items() if k in self.SAVE_LIST},
        'child': [x.to_dict() for x in self.child] if self.child else None,
    }

  def load_dict(self, sample_dict, parent=None):
    self.__dict__.update({k: v for k, v in sample_dict.items() if k in self.SAVE_LIST})
    child = sample_dict.get('child', None)
    self.parent = parent
    self.child = [PredictionNode(self.semantic_child[i]).load_dict(child_dict, self) for i, child_dict in enumerate(child)] if child else None
    return self

  @property
  def id(self):
    return self.semantic_node.id
  
class InferSample(object):
  def __init__(self, semantic_root_node, node_registry, hp, logger, query='', gold_paths=[], excluded_ids_set=None, max_rerank_size=20):
    self.query = query
    self.gold_paths = gold_paths
    self.excluded_ids_set = excluded_ids_set
    self.semantic_root_node = semantic_root_node
    if (self.excluded_ids_set is not None) and (len(self.excluded_ids_set) > 0):
      self.semantic_root_node = MaskedSemanticNode(semantic_root_node, excluded_ids_set)
    self.max_beam_size = hp.MAX_BEAM_SIZE
    self.max_rerank_size = max_rerank_size
    self.num_iters = 0
    self.search_with_path_relevance = hp.SEARCH_WITH_PATH_RELEVANCE
    self.num_leaf_calib = hp.NUM_LEAF_CALIB
    self.logger = logger

    self.get_traversal_prompt = partial(get_traversal_prompt, hp=hp, logger=logger, return_constraint=False)
    self.get_reranking_prompt = partial(get_reranking_prompt, hp=hp, logger=logger)

    self.prediction_tree = PredictionNode(self.semantic_root_node, parent=None, local_relevance=1.0, path=(), creation_step=0, excluded_ids_set=excluded_ids_set, relevance_chain_factor=hp.RELEVANCE_CHAIN_FACTOR)
    self.beam_state_paths = [[self.prediction_tree]]
    self.beam_state_paths_history = []
    self.frontier = [self.prediction_tree]

    self.calib_model = CalibModel(len(node_registry), tau=hp.PL_TAU)
    self.node_registry = node_registry
    self.relevance_chain_factor = hp.RELEVANCE_CHAIN_FACTOR

    self.SAVE_LIST = ['query', 'gold_paths', 'max_beam_size', 'max_rerank_size', 'num_iters', 'search_with_path_relevance', 'num_leaf_calib', 'excluded_ids_set']

  def __str__(self):
    string = f'Query: {self.query}\n\nBeam state paths:'
    for state_path in self.beam_state_paths:
      string += f'\n{state_path[-1].path}'
    return string

  def __repr__(self):
    return self.__str__()

  def to_dict(self):
    return {**{k: v for k, v in self.__dict__.items() if k in self.SAVE_LIST},
            'prediction_tree': self.prediction_tree.to_dict(),
            'calib_model': self.calib_model.to_dict(),}

  def load_dict(self, sample_dict):
    self.__dict__.update({k: v for k, v in sample_dict.items() if k in self.SAVE_LIST})
    self.prediction_tree = self.prediction_tree.load_dict(sample_dict['prediction_tree'])
    self.calib_model = self.calib_model.load_dict(sample_dict.get('calib_model', {}))
    self.beam_state_paths = [[self.prediction_tree]]
    self.beam_state_paths_history = []

    if self.num_iters > 0:
        self.update_next_beam_states(self.get_rel_fn())

    assert ((self.excluded_ids_set is None) or (len(self.excluded_ids_set) == 0)) or (isinstance(self.semantic_root_node, MaskedSemanticNode) and (self.semantic_root_node.excluded_ids_set == self.excluded_ids_set)), f'Excluded ids set mismatch: {self.excluded_ids_set} vs {self.semantic_root_node.excluded_ids_set if isinstance(self.semantic_root_node, MaskedSemanticNode) else "None"}'
    return self

  def post_load_processing(self):
    if not self.calib_model.trained: 
      self.calib_model.fit()
      self.update_relevances(self.prediction_tree)

  @property
  def beam_size(self):
    return len(self.beam_state_paths)

  def get_rerank_step_prompt(self):
    self.beam_state_paths = []
    for i in range(self.max_beam_size):
      if len(self.frontier) == 0:
        break
      self.beam_state_paths.append([self.frontier.pop()])
    doc_list = [x for node in self.beam_state_paths for x in node[-1].child_desc]
    prompt, constraint = self.get_reranking_prompt(self.query, doc_list, topk=min(self.max_rerank_size, len(doc_list)))
    return prompt, constraint

  def process_rerank_step_response(self, rerank_step_response):
    response_json = repair_json(rerank_step_response, return_objects=True)
    reasoning = response_json['reasoning']
    ranking = response_json['ranking']
    flat_index_to_nested_index = [(i, j) for i, path in enumerate(self.beam_state_paths) for j, x in enumerate(path[-1].child_desc)]
    nested_relevances = [[0 for _ in range(path[-1].num_children)] for path in self.beam_state_paths]
    for rank, ind in enumerate(ranking):
      i, j = flat_index_to_nested_index[ind]
      nested_relevances[i][j] = 1/np.log2(rank+2)
    for i, path in enumerate(self.beam_state_paths):
      if path[-1].child:
        for c in path[-1].child:
          c.local_relevance = nested_relevances[i][c.path[-1]]
          c.path_relevance = chain_path_rel_fn(c.local_relevance, c.parent.path_relevance)
      else:
        path[-1].instantiate_children(nested_relevances[i], reasoning, creation_step=self.num_iters+1)

    self.num_iters += 1
    self.beam_state_paths_history.append(self.beam_state_paths)
    for ind in ranking[::-1]:
      i, j = flat_index_to_nested_index[ind]
      new_node = self.beam_state_paths[i][-1].child[j]
      if not new_node.is_leaf:
        self.frontier.append(new_node)

  def get_all_explored_nodes_at_level(self, node, level):
    if level == 0:
      return [node]

    ret_list = []
    if node.child:
      for child in node.child:
        ret_list += self.get_all_explored_nodes_at_level(child, level-1)
    return ret_list

  def get_best_node_at_level(self, cur_level):
    all_nodes_at_cur_level = self.get_all_explored_nodes_at_level(self.prediction_tree, cur_level)
    if len(all_nodes_at_cur_level) > 0:
      best_node_at_cur_level = max(all_nodes_at_cur_level, key=lambda x: x.path_relevance)
      return best_node_at_cur_level
    else:
      return None

  def get_step_prompts(self):
    inputs = []
    top_preds = self.get_top_predictions(k=None, rel_fn=self.get_rel_fn(return_calibrated_rel=True))
    assert all([s <= 1.0001 for _, s in top_preds]), f'Top predictions not normalized: {top_preds}'
    if len(top_preds) > self.num_leaf_calib:
      theta = np.array([max(s, -1) for _, s in top_preds])
      th = get_bimodal_gmm_intrsxn(theta)
      p = [((4**(2*s)) if s > th else 1e-4) for _, s in top_preds]
      # p = topk(theta, k=self.num_leaf_calib, alpha=10)
      p = np.array(p) / np.sum(p)
      sampled_inds = np.random.choice(np.arange(len(top_preds)), size=self.num_leaf_calib, replace=False, p=p)
      top_preds = [top_preds[i] for i in sampled_inds]      
    top_preds_slate = [(x.desc, x.registry_idx) for x, _ in top_preds]

    for state_path in self.beam_state_paths:
      cur_state = state_path[-1]
      slate = [(child.desc, child.registry_idx) for child in cur_state.semantic_node.child]
      leaf_cluster = all([(len(child.child)==0) for child in cur_state.semantic_node.child])
      anchor = None; anchor_score = None
      if self.num_leaf_calib:
        if leaf_cluster:
          slate += top_preds_slate
        else:
          cur_level = len(cur_state.path)
          best_node_at_cur_level = self.get_best_node_at_level(cur_level)
          if (best_node_at_cur_level is not None) and (cur_level > 0):
            slate += [(best_node_at_cur_level.desc, best_node_at_cur_level.registry_idx)]

      desc_list = [x[0] for x in slate]
      prompt = self.get_traversal_prompt(self.query, desc_list, leaf_cluster=leaf_cluster, anchor=anchor, anchor_score=anchor_score)
      inputs.append((prompt, [x[1] for x in slate]))
    return inputs

  def process_beam_response_jsons(self, beam_slates, beam_response_jsons):
    assert len(beam_response_jsons) == self.beam_size
    for b, (state_path, slate, response_json) in enumerate(zip(self.beam_state_paths, beam_slates, beam_response_jsons)):
      cur_state = state_path[-1]
      cur_semantic_node = cur_state.semantic_node

      reasoning = recursive_key_search(response_json, 'reasoning')
      relevance_scores = recursive_key_search(response_json, 'relevance_scores')
      try:
        relevance_scores = {slate[int(k)]: float(v)/100 for k, v in relevance_scores}
      except Exception as e:
        self.logger.error(f'Error parsing relevance scores: {relevance_scores}, slate: {slate} with error {e}')
        relevance_scores = None

      if relevance_scores:
        self.calib_model.add(relevance_scores)
        cur_node_child_rels = [relevance_scores.get(c.registry_idx, 0) for c in cur_semantic_node.child]
        cur_state.instantiate_children(cur_node_child_rels, reasoning, creation_step=len(self.beam_state_paths_history)+1)
    self.calib_model.fit()
    self.update_relevances(self.prediction_tree)

  def update_relevances(self, node):
    if node.parent:
      node.calibrated_relevance = float(self.calib_model.theta[node.registry_idx])

    if node.child:
      for child in node.child:
        self.update_relevances(child)

  def get_all_expandable_paths(self, node, history=[]):
    if not node.predicted:
      return [] if node.is_leaf else [history + [node]]

    ret_list = []
    for child in node.child:
      ret_list += self.get_all_expandable_paths(child, history + [node])

    return ret_list

  def get_all_candidate_nodes_of_size(self, node, levels, min_size, max_size):
    if node.is_leaf:
      return []

    ret_list = []
    if not node.predicted:
    #   if min_size <= node.num_leaves <= max_size:?
      assert isinstance(levels, (list, set)), f'Level should be a list or set, got {type(levels)}'
      if len(node.path) in levels:
          ret_list.append(node)

    if node.child:
      for child in node.child:
        ret_list += self.get_all_candidate_nodes_of_size(child, levels, min_size, max_size)

    return ret_list

  def get_top_candidates(self, levels, min_size, max_size, k=None, rel_fn=None):
    rel_fn = rel_fn or self.get_rel_fn()
    all_candidates = self.get_all_candidate_nodes_of_size(self.prediction_tree, levels, min_size, max_size)
    all_candidates = [(x, rel_fn(x)) for x in all_candidates]
    return sorted(all_candidates, key=lambda x: x[1], reverse=True)[:k]

  def get_all_predicted_leaves(self, node):
    if node.is_leaf:
      return [node]

    if not node.predicted:
      return []

    ret_list = []
    for child in node.child:
      ret_list += self.get_all_predicted_leaves(child)
    return ret_list

  def get_top_predictions(self, k=None, rel_fn=None):
    rel_fn = rel_fn or self.get_rel_fn(leaf=True)
    all_predicted_leaves = [(x, rel_fn(x)) for x in self.get_all_predicted_leaves(self.prediction_tree)]
    return sorted(all_predicted_leaves, key=lambda x: x[1], reverse=True)[:k]

  def compute_eval_metrics(self, k=10, rel_fn=None):
    rel_fn = rel_fn or self.get_rel_fn(leaf=True)
    sorted_paths = [list(x[0].path) for x in self.get_top_predictions(rel_fn=rel_fn)]
    return {
        f'nDCG@{k}': compute_ndcg(sorted_paths[:k], self.gold_paths, k=k)*100,
        f'Recall@{k}': compute_recall(sorted_paths[:k], self.gold_paths, k=k)*100,
        f'Recall@{100}': compute_recall(sorted_paths[:100], self.gold_paths, k=100)*100,
        f'Recall@all': compute_recall(sorted_paths, self.gold_paths, k=len(sorted_paths))*100,
        f'Coverage': len(sorted_paths),
    }

  def get_rel_fn(self, leaf=False, return_calibrated_rel=False):
    def rel_fn(x):
      x = x[-1] if isinstance(x, list) else x
      parent_rel = x.parent.path_relevance if (x.parent is not None) else 1.0
      local_rel = x.local_relevance
      path_rel = x.path_relevance
      calibrated_rel = x.calibrated_relevance
      alpha = self.relevance_chain_factor if (self.relevance_chain_factor is not None) else 0.5
      combined_rel = (1-alpha) * parent_rel + alpha * calibrated_rel
      # combined_rel = x.path_relevance if self.search_with_path_relevance else x.calibrated_relevance
      if return_calibrated_rel:
        return calibrated_rel
      return combined_rel if leaf else path_rel

    return rel_fn

  def update_next_beam_states(self, rel_fn):
    all_expandable_paths = self.get_all_expandable_paths(self.prediction_tree)
    topk_expandable_paths = sorted(all_expandable_paths, key=rel_fn, reverse=True)[:self.max_beam_size]
    self.beam_state_paths_history.append(self.beam_state_paths)
    self.beam_state_paths = [x for x in topk_expandable_paths]

  def update(self, beam_slates, beam_response_jsons, rel_fn=None):
    rel_fn = rel_fn or self.get_rel_fn()
    self.process_beam_response_jsons(beam_slates, beam_response_jsons)
    self.update_next_beam_states(rel_fn)
    self.num_iters += 1

  def get_expandable_paths_rerank_prompts(self, rel_fn = None):
    rel_fn = rel_fn or self.get_rel_fn()
    all_expandable_paths = self.get_all_expandable_paths(self.prediction_tree)
    topk_expandable_paths = sorted(all_expandable_paths, key=rel_fn, reverse=True)[:self.max_rerank_size]
    prompt, constraint = get_reranking_prompt(self.query, [x[-1].desc for x in topk_expandable_paths], topk=self.max_beam_size)
    return topk_expandable_paths, prompt, constraint

  def process_expandable_paths_rerank_response(self, topk_expandable_paths, rerank_response):
    ranking = repair_json(rerank_response, return_objects=True)['ranking']
    ranked_expandable_paths = [topk_expandable_paths[i] for i in ranking]
    self.beam_state_paths_history.append(self.beam_state_paths)
    self.beam_state_paths = [x for x in ranked_expandable_paths]