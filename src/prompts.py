import re
import numpy as np
from google.genai.types import Schema, Type, Content

#region Tree traversal prompt with explicit reasoning
TREE_TRAVERSAL_PROMPT_V5_CLUSTER = '''You are an intelligent search agent navigating a hierarchical semantic tree of topics. Your mission is to predict the most promising candidates to find the answer to the user's query using the relevance definition below.

**Relevance Definition:** {relevance_defintion}

---

## USER QUERY

{query}

---

## CANDIDATES

Here are the candidates, each is identified by a unique `node_id` provided at the very start in [] (e.g., [0]).

{child_node_options}

---

## YOUR EVALUATION TASK
1.  First, identify the essential problem in the query.
2.  Think step by step to reason about why each candidate is relevant or irrelevant (based on the relevance definition). Provide this analysis in the `reasoning` field.
3.  Rank these passages based on their relevance to the query. Provide your ranking in the `ranking` field.
4.  Assign a relevance score from 0 to 100 (based on the relevance definition and the ranking). Provide relevances in the `relevance_scores` field.

---

## OUTPUT FORMAT
You must provide your response as a single, clean JSON object. The JSON should have three keys: `reasoning`, `ranking`, and `relevance_scores`.

* `reasoning`: This must be a **string**.
* `ranking`: This must be an **array of integers** representing the order of the candidates.
* `relevance_scores`: This must be an **array of arrays** where each inner array contains [node_id, relevance_score]. For example: [[0, 85], [1, 92], [2, 73]].

---

Prompt ID: {prompt_id} (Ignore this, it is just for watermarking purposes)

## YOUR RESPONSE
'''

TREE_TRAVERSAL_PROMPT_V5_LEAF = '''You are an intelligent search agent evaluating a set of candidate passages. Your mission is to predict the most relevant passages to find the answer to the user's query using the relevance definition below.

**Relevance Definition:** {relevance_defintion}. A passage is considered relevant if its text, read in isolation, directly and substantively contributes to answering the user's query. It must contain actual information, not just references or titles. A passage that only mentions the query's topic but provides no specific details is considered irrelevant

---

## USER QUERY

{query}

---

## CANDIDATES

Here are the candidates, each is identified by a unique `node_id` provided at the very start in [] (e.g., [0]).

{child_node_options}

---

## YOUR EVALUATION TASK
1.  First, identify the essential problem in the query.
2.  Think step by step to reason about why each passage is relevant or irrelevant (based on the relevance definition). Provide this analysis in the `reasoning` field.
3.  Rank these passages based on their relevance to the query. Provide your ranking in the `ranking` field.
4.  Assign a relevance score from 0 to 100. The score must reflect how well the passage, *on its own*, substantively answers the query.
    * **High scores** should be reserved for passages that contain a direct answer or a critical piece of the answer.
    * **Low scores** must be given to passages that are merely titles, section headings, table of contents, or descriptions of other content. These are not useful as they do not contain the answer themselves.
    * Provide these scores in the `relevance_scores` field.
---

## OUTPUT FORMAT
You must provide your response as a single, clean JSON object. The JSON should have three keys: `reasoning`, `ranking`, and `relevance_scores`.

* `reasoning`: This must be a **string**.
* `ranking`: This must be an **array of integers** representing the order of the candidates.
* `relevance_scores`: This must be an **array of arrays** where each inner array contains [node_id, relevance_score]. For example: [[0, 85], [1, 92], [2, 73]].

---

Prompt ID: {prompt_id} (Ignore this, it is just for watermarking purposes)

## YOUR RESPONSE
'''
#endregion

#region Define `get_traversal_prompt`
def get_content_proto_size(text):
  return len(text.encode('utf-8'))

def get_desc_str_from_list(desc_list, max_char_len=None):
  return ''.join(["[{}]. {}\n\n".format(i, re.sub('\n+', ' ', doc[:max_char_len])) for i, doc in enumerate(desc_list)])

def get_relevance_definition(subset):
    subset = subset.lower()
    RELEVANCE_DEFINITIONS = {
        'leetcode': '''The relevance between queries and positive documents is defined by whether the coding problem (i.e., query) involves the same algorithm and/or data structure. The queries and documents are problems and solutions from LeetCode. The problem descriptions are used as queries Q, and the positive documents D+Q are solved problems (with solutions) that were annotated as similar problems by LeetCode.''',
        'theoremqa_questions': '''A query is relevant to a document if the document references the same/similar theorem used in the query.''',
        'pony': '''The relevance between queries and positive documents is defined by whether the coding problem (i.e., query) requires the corresponding syntax documentation.''',
        'stackexchange': '''A document is considered relevant to a query if it can be cited in an accepted or highly voted answer that helps reason through the query with critical concepts or theories.''',
        'scifact': '''A document is relevant if it contains sufficient, non-redundant, and minimal evidence (rationale sentences) that can be used to determine the veracity (either *support* OR *refutation*) of a specific scientific claim. The query is a scientific claim, and the documents are abstracts of scientific papers.''',
        'nq': '''A document is relevant to a query if it contains the answer to the question posed in the query. The query is a natural language web search question, and the documents are passages from wikipedia that may contain the answer.''',
        "fiqa": '''A document is considered relevant if it contains the specific information needed to answer the associated question.''',
        'scidocs': '''A query is a source scientific paper (represented by its title and abstract) and a document is a candidate paper from the corpus. A gold document is defined as relevant because it is either directly cited by the query paper or co-viewed (accessed in the same user session) with the query paper in user activity logs.''',
    }
    RELEVANCE_DEFINITIONS['aops'] = RELEVANCE_DEFINITIONS['theoremqa_questions']
    RELEVANCE_DEFINITIONS['theoremqa_theorems'] = RELEVANCE_DEFINITIONS['theoremqa_questions']
    RELEVANCE_DEFINITION = RELEVANCE_DEFINITIONS[subset] if subset in RELEVANCE_DEFINITIONS else RELEVANCE_DEFINITIONS['stackexchange']
    return RELEVANCE_DEFINITION

def get_traversal_prompt_response_constraint(require_reasoning=True, return_dict=True):
    if not return_dict:
      tree_traversal_response_schema = Schema(
          type=Type.OBJECT,
          properties=({
            "reasoning": Schema(
                  type=Type.STRING,
                  description="Step-by-step analysis of why each document is relevant or irrelevant based on the relevance definition"
              )} if require_reasoning else {}) | {
              "ranking": Schema(
                  type=Type.ARRAY,
                  items=Schema(type=Type.INTEGER),
                  description="Array of integers representing the order of the candidates by their node_id",
                  minItems=1,
              ),
              "relevance_scores": Schema(
                  type=Type.ARRAY,
                  items=Schema(
                      type=Type.ARRAY,
                      items=Schema(type=Type.INTEGER),
                      description="Tuple of [node_id, relevance_score] where relevance_score is 0-100"
                  ),
                  description="Array of tuples, each containing [node_id, relevance_score]",
                  minItems=1,
              )
          },
          required=(["reasoning"] if require_reasoning else []) + ["ranking", "relevance_scores"]
      )
      return tree_traversal_response_schema
    else:
      response_dict = {
        "type": "object",
        "properties": ({
            "reasoning": {
                  "type": "string",
                  "description": "Step-by-step analysis of why each document is relevant or irrelevant based on the relevance definition"
              }} if require_reasoning else {}) | {
              "ranking": {
                  "type": "array",
                  "items": {"type": "integer"},
                  "description": "Array of integers representing the order of the candidates by their node_id",
                  "minItems": 1,
              },
              "relevance_scores": {
                  "type": "array",
                  "items": {
                      "type": "array",
                      "items": {"type": "integer"},
                      "description": "Tuple of [node_id, relevance_score] where relevance_score is 0-100"
                  },
                  "description": "Array of tuples, each containing [node_id, relevance_score]",
                  "minItems": 1, 
              }
          },
          "required": (["reasoning"] if require_reasoning else []) + ["ranking", "relevance_scores"]
      }
      return response_dict

def get_traversal_prompt(query, child_desc_list, hp, logger, return_constraint=True, **kwargs):
  max_desc_char_len = None
  constraint = get_traversal_prompt_response_constraint() if return_constraint else None
  relevance_definition = get_relevance_definition(hp.SUBSET)

  while True:
    args = {
        'query': query.replace('\n','  '),
        'child_node_options': get_desc_str_from_list(child_desc_list, max_desc_char_len)
    }

    args['relevance_defintion'] = relevance_definition
    args['prompt_id'] = np.random.randint(10_000_000)
    assert 'leaf_cluster' in kwargs, 'leaf boolean must be provided for version 5'
    prompt = TREE_TRAVERSAL_PROMPT_V5_LEAF.format(**args) if kwargs['leaf_cluster'] else TREE_TRAVERSAL_PROMPT_V5_CLUSTER.format(**args)

    if bool(hp.MAX_PROMPT_PROTO_SIZE) and (get_content_proto_size(prompt) > hp.MAX_PROMPT_PROTO_SIZE):
      max_desc_char_len = hp.MAX_DOC_DESC_CHAR_LEN if max_desc_char_len is None else max_desc_char_len - 100
      logger.debug(f'proto size of prompt is {get_content_proto_size(prompt)}, decreasing max_desc_char_len to {max_desc_char_len}')
    else:
      return (prompt, constraint) if return_constraint else prompt
#endregion

#region Define `get_reranking_prompt`
def get_reasoned_ranking_genai_schema(topk: int) -> dict:
    if not isinstance(topk, int) or topk <= 0:
        raise ValueError("topk must be a positive integer")

    schema = {
        "type": "object",
        "properties": {
            "_reasoning": {
                "type": "string",
                "description": "Step-by-step analysis of why each document is relevant or irrelevant to the query."
            },
            "ranking": {
                "type": "array",
                "description": f"A list of exactly {topk} passage IDs, ranked from most to least relevant.",
                "items": {
                    "type": "integer",
                    "description": "The ID of a passage."
                },
                # "minItems": topk,
                # "maxItems": topk
            }
        },
        "required": [
            "_reasoning",
            "ranking"
        ],
    }
    return schema

def get_reranking_prompt(query, docs, hp, logger, topk=10, return_constraint=True, max_prompt_proto_size=None):
  max_desc_char_len = None
  while True:
    doc_string = get_desc_str_from_list(docs, max_desc_char_len)
    cur_query = query.replace('\n','  ')
    prompt = (
          f'The following passages are related to query: {cur_query}\n\n'
          f'{doc_string}\n\n'
          f'1.  First, identify the essential problem in the query.\n'
          f'2.  Think step by step to reason about why each document is relevant or irrelevant. Provide this analysis in the `reasoning` field.\n'
          f'3.  Rank these passages based on their relevance to the query.\n'
          f'4.  Output the ranking result of passages as a list of {topk} integer IDs, where the first element is the ID of the most relevant '
          f'passage, the second element is the ID of the second most relevant passage, etc. Place this list in the `ranking` field.\n\n'
          f'Please output your response as a single JSON object with two keys: "reasoning" (a string) and "ranking" (a list of {topk} integers).\n'
          f'Example Format:\n'
          f'{{\n'
          f'  "reasoning": "Passage X is highly relevant because... Passage Y is less relevant because...",\n'
          f'  "ranking": [integer_id_1, integer_id_2, ..., integer_id_{topk}]\n'
          f'}}\n'
          f'```'
      )
    if bool(max_prompt_proto_size) and (get_content_proto_size(prompt) > max_prompt_proto_size):
        max_desc_char_len = hp.MAX_DOC_DESC_CHAR_LEN if max_desc_char_len is None else max_desc_char_len - 100
        logger.debug(f'proto size of prompt is {get_content_proto_size(prompt)}, decreasing max_desc_char_len to {max_desc_char_len}')
    else:
      return (prompt, get_reasoned_ranking_genai_schema(topk)) if return_constraint else prompt
#endregion
