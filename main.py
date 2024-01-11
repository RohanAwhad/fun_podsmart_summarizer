# [Reference]: https://towardsdatascience.com/summarize-podcast-transcripts-and-long-texts-better-with-nlp-and-ai-e04c89d3b2cb

from datetime import datetime
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import networkx as nx
from networkx.algorithms import community

import json
from nltk.tokenize import sent_tokenize

from src import utils
from src.encoder_service import encode
from src.pdf_reader import pdf_to_text
from src.summarizer import summarize_stage_1, summarize_stage_2

with open('chapter_8_biocomputing_reading.txt', 'r') as f: text = f.read()



# split into sentences
sentences = sent_tokenize(text)
# further split by comma
segments = [sent.split(',') for sent in sentences]
# flaten
segments = [x for sent in segments for x in sent]


sentences = utils.create_sentences(segments, MIN_WORDS=20, MAX_WORDS=80)
chunks = utils.create_chunks(sentences, CHUNK_LENGTH=5, STRIDE=1)
chunks_text = [chunk['text'] for chunk in chunks]

if not (os.path.exists('stage_1_summaries.txt') and os.path.exists('stage_1_titles.txt')):
  #raise Exception('shouldnt be here')
  # Run Stage 1 Summarizing
  stage_1_outputs = summarize_stage_1(chunks_text)['stage_1_outputs']
  # Split the titles and summaries
  stage_1_summaries = [e['summary'] for e in stage_1_outputs]
  stage_1_titles = [e['title'] for e in stage_1_outputs]
  num_1_chunks = len(stage_1_summaries)

  with open('stage_1_summaries.txt', 'w') as f:
    for s in stage_1_summaries:
      f.write(s + '\n---\n')

  with open('stage_1_titles.txt', 'w') as f:
    for s in stage_1_titles:
      f.write(s + '\n---\n')
else:
  with open('stage_1_summaries.txt', 'r') as f:
    stage_1_summaries = f.read().split('---')[:-1]
  with open('stage_1_titles.txt', 'r') as f:
    stage_1_titles = f.read().split('---')[:-1]

  stage_1_summaries = [s.strip() for s in stage_1_summaries]
  stage_1_titles = [s.strip() for s in stage_1_titles]

  stage_1_summaries = [s for s in stage_1_summaries if s != '']
  stage_1_titles = [s for s in stage_1_titles if s != '']

  stage_1_outputs = [{'title': t, 'summary': s} for t, s in zip(stage_1_titles, stage_1_summaries)]
  num_1_chunks = len(stage_1_summaries)


# summary and title embeddings
summary_embeds = encode(stage_1_summaries).numpy()
title_embeds = encode(stage_1_titles).numpy()


# Get similarity matrix between the embeddings of the chunk summaries
summary_similarity_matrix = np.zeros((num_1_chunks, num_1_chunks))
summary_similarity_matrix[:] = np.nan

for row in range(num_1_chunks):
  for col in range(row, num_1_chunks):
    # Calculate cosine similarity between the two vectors
    similarity = 1- cosine(summary_embeds[row], summary_embeds[col])
    summary_similarity_matrix[row, col] = similarity
    summary_similarity_matrix[col, row] = similarity

# Draw a heatmap with the summary_similarity_matrix
plt.figure()
# Color scheme blues
plt.imshow(summary_similarity_matrix, cmap = 'Blues')
# save the figure
plt.savefig('summary_similarity_matrix.png', dpi=300)

# Run the community detection algorithm

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

  resolution = 0.85
  resolution_step = 0.01
  iterations = 40

  # Find the resolution that gives the desired number of topics
  topics_title = []
  while len(topics_title) not in [desired_num_topics, desired_num_topics + 1, desired_num_topics + 2]:
    topics_title = community.louvain_communities(title_nx_graph, weight = 'weight', resolution = resolution)
    resolution += resolution_step
  topic_sizes = [len(c) for c in topics_title]
  sizes_sd = np.std(topic_sizes)
  modularity = community.modularity(title_nx_graph, topics_title, weight = 'weight', resolution = resolution)

  lowest_sd_iteration = 0
  # Set lowest sd to inf
  lowest_sd = float('inf')

  for i in range(iterations):
    topics_title = community.louvain_communities(title_nx_graph, weight = 'weight', resolution = resolution)
    modularity = community.modularity(title_nx_graph, topics_title, weight = 'weight', resolution = resolution)
    
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

# Set num_topics to be 1/4 of the number of chunks, or 8, which ever is smaller
num_topics = min(int(num_1_chunks / 4), 8)
topics_out = get_topics(summary_similarity_matrix, num_topics = num_topics, bonus_constant = 0.2)
chunk_topics = topics_out['chunk_topics']
topics = topics_out['topics']


# Plot a heatmap of this array
plt.figure(figsize = (10, 4))
plt.imshow(np.array(chunk_topics).reshape(1, -1), cmap = 'tab20')
# Draw vertical black lines for every 1 of the x-axis 
for i in range(1, len(chunk_topics)):
  plt.axvline(x = i - 0.5, color = 'black', linewidth = 0.5)
# save the figure
plt.savefig('chunk_topics.png', dpi=300)
#exit(0)

# Query GPT-3 to get a summarized title for each topic_data
out = summarize_stage_2(stage_1_outputs, topics, summary_num_words = 250)
stage_2_outputs = out['stage_2_outputs']
stage_2_titles = [e['title'] for e in stage_2_outputs]
stage_2_summaries = [e['summary'] for e in stage_2_outputs]
final_summary = out['final_summary']


print(stage_2_outputs)
with open('stage_2_outputs.jsonl', 'w') as f:
  for e in stage_2_outputs: f.write(json.dumps(e) + '\n')

with open('final_summary.txt', 'w') as f: f.write(final_summary)