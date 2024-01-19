import networkx as nx
import numpy as np
import pandas as pd
import re
from networkx.algorithms import community
from typing import List, Dict, Union
from loguru import logger

def cosine_similarity(a: Union[List[float], np.ndarray], b: Union[List[float], np.ndarray]) -> np.ndarray:
  if isinstance(a, list): a = np.array(a)
  if isinstance(b, list): b = np.array(b)

  # check if a and b are 2D matrices
  assert len(a.shape) == 2, 'Matrices must be 2D'
  assert len(b.shape) == 2, 'Matrices must be 2D'

  # check if the matrices are the same shape
  assert a.shape == b.shape, 'Matrices must be the same shape'

  norm_a = np.linalg.norm(a, axis=1, keepdims=True)
  norm_b = np.linalg.norm(b, axis=1, keepdims=True)
  return (a @ b.T / (norm_a * norm_b))

def create_sentences(segments, MIN_WORDS, MAX_WORDS):
  # Combine the non-sentences together
  sentences = []

  is_new_sentence = True
  sentence_length = 0
  sentence_num = 0
  sentence_segments = []

  for i in range(len(segments)):
    if is_new_sentence == True: is_new_sentence = False

    # Append the segment
    sentence_segments.append(segments[i])
    segment_words = segments[i].split(' ')
    sentence_length += len(segment_words)
    
    # If exceed MAX_WORDS, then stop at the end of the segment
    # Only consider it a sentence if the length is at least MIN_WORDS
    if (sentence_length >= MIN_WORDS and segments[i][-1] == '.') or sentence_length >= MAX_WORDS:
      sentence = ' '.join(sentence_segments)
      sentences.append({
        'sentence_num': sentence_num,
        'text': sentence,
        'sentence_length': sentence_length
      })
      # Reset
      is_new_sentence = True
      sentence_length = 0
      sentence_segments = []
      sentence_num += 1

  return sentences

def create_chunks(sentences, CHUNK_LENGTH, STRIDE):

  sentences_df = pd.DataFrame(sentences)
  
  chunks = []
  for i in range(0, len(sentences_df), (CHUNK_LENGTH - STRIDE)):
    chunk = sentences_df.iloc[i:i+CHUNK_LENGTH]
    chunk_text = ' '.join(chunk['text'].tolist())
    
    chunks.append({
      'start_sentence_num': chunk['sentence_num'].iloc[0],
      'end_sentence_num': chunk['sentence_num'].iloc[-1],
      'text': chunk_text,
      'num_words': len(chunk_text.split(' '))
    })
    
  chunks_df = pd.DataFrame(chunks)
  return chunks_df.to_dict(orient='records')  # return as list of each row as a dict with keys as column names

def parse_title_summary_results(results):
  out = []
  for text in results:
    title_pattern = r"Title: (.*)\n"
    summary_pattern = r"Summary:\n((?:- .*\n)*)"

    title = re.search(title_pattern, text).group(1)
    summary = re.findall(summary_pattern, text)[0].split('\n')
    summary = [item.strip('- ') for item in summary if item]
    out.append({
      'title': title,
      'summary': ' '.join(summary)
    })
  return out

def get_topics(title_similarity, num_topics = 8, bonus_constant = 0.25, min_size = 3):

  proximity_bonus_arr = np.zeros_like(title_similarity)
  for row in range(proximity_bonus_arr.shape[0]):
    for col in range(proximity_bonus_arr.shape[1]):
      if row == col:
        proximity_bonus_arr[row, col] = 0
      else:
        proximity_bonus_arr[row, col] = 1/(abs(row-col)) * bonus_constant
        
  title_similarity += proximity_bonus_arr

  title_nx_graph = nx.from_numpy_array(title_similarity)

  desired_num_topics = num_topics
  # Store the accepted partitionings
  topics_title_accepted = []

  resolution = 0.7
  resolution_step = 0.01
  iterations = 40

  # Find the resolution that gives the desired number of topics
  topics_title = []
  _ = 0
  while len(topics_title) not in [desired_num_topics, desired_num_topics + 1, desired_num_topics + 2]:
    topics_title = community.louvain_communities(title_nx_graph, weight = 'weight', resolution = resolution)
    resolution += resolution_step
    _ += 1
    if _ > 1000:
      msg = (
        'breaking because cnt > 1000\n'
        f'len(topics_title): {len(topics_title)}\n'
        f'resolution: {resolution}\n'
        f'topics_title: {topics_title}'
        f'title_similarity: {title_similarity}\n'
      )
      logger.error(msg)
      break
  topic_sizes = [len(c) for c in topics_title]
  sizes_sd = np.std(topic_sizes)
  # modularity = community.modularity(title_nx_graph, topics_title, weight = 'weight', resolution = resolution)

  lowest_sd_iteration = 0
  # Set lowest sd to inf
  lowest_sd = float('inf')

  for i in range(iterations):
    topics_title = community.louvain_communities(title_nx_graph, weight = 'weight', resolution = resolution)
    # modularity = community.modularity(title_nx_graph, topics_title, weight = 'weight', resolution = resolution)
    
    # Check SD
    topic_sizes = [len(c) for c in topics_title]
    sizes_sd = np.std(topic_sizes)
    
    topics_title_accepted.append(topics_title)
    
    if sizes_sd < lowest_sd and min(topic_sizes) >= min_size:
      lowest_sd_iteration = i
      lowest_sd = sizes_sd
      
  # Set the chosen partitioning to be the one with highest modularity
  topics_title = topics_title_accepted[lowest_sd_iteration]
  print(f'Best SD: {lowest_sd}, Best iteration: {lowest_sd_iteration}')
  
  topic_id_means = [sum(e)/len(e) for e in topics_title]
  # Arrange title_topics in order of topic_id_means
  topics_title = [list(c) for _, c in sorted(zip(topic_id_means, topics_title), key = lambda pair: pair[0])]
  # Create an array denoting which topic each chunk belongs to
  chunk_topics = [None] * title_similarity.shape[0]
  for i, c in enumerate(topics_title):
    for j in c:
      chunk_topics[j] = i
            
  return {
    'chunk_topics': chunk_topics,
    'topics': topics_title
    }

def json_to_md(data: List[Dict[str, str]]) -> str:
    # create markdown with title as h3 and text as p
  ret = ''
  for d in data:
    ret += f'### {d["title"]}\n'
    ret += f'{d["summary"]}\n\n'

  return ret