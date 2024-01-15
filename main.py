# [Reference]: https://towardsdatascience.com/summarize-podcast-transcripts-and-long-texts-better-with-nlp-and-ai-e04c89d3b2cb
import langchain
langchain.debug = True

import numpy as np
import matplotlib.pyplot as plt

from langchain.callbacks import FileCallbackHandler
from loguru import logger
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel
from scipy.spatial.distance import cosine
from typing import Dict, List

from src import utils
from src.encoder_service import encode
from src.pdf_reader import pdf_to_text
from src.summarizer import summarize_stage_1, summarize_stage_2

VERSION = 1.1
logfile = f'logs/main_v{VERSION}.log'
logger.add(logfile, colorize=True, enqueue=True)
handler = FileCallbackHandler(logfile)


class MainOut(BaseModel):
  stage_1_outputs: List[Dict[str, str]]
  stage_2_outputs: List[Dict[str, str]]
  final_summary: str
  markdown_summary: str
  summary_similarity_matrix: np.ndarray
  chunk_topics: List[int]

  class Config:
    arbitrary_types_allowed = True


def save_similarity_matrix_plot(similarity_matrix: np.ndarray, filename: str):
  # Draw a heatmap with the summary_similarity_matrix
  plt.figure()
  # Color scheme blues
  plt.imshow(similarity_matrix, cmap = 'Blues')
  # save the figure
  plt.savefig(filename, dpi=300)


def save_topics_plot(topics: List[List[int]], filename: str):
  # Plot a heatmap of this array
  plt.figure(figsize = (10, 4))
  plt.imshow(np.array(topics).reshape(1, -1), cmap = 'tab20')
  # Draw vertical black lines for every 1 of the x-axis 
  for i in range(1, len(topics)):
    plt.axvline(x = i - 0.5, color = 'black', linewidth = 0.5)
  # save the figure
  plt.savefig(filename, dpi=300)


def main(text: str) -> MainOut:
  # split into sentences
  sentences = sent_tokenize(text)
  # further split by comma
  segments = [sent.split(',') for sent in sentences]
  # flaten
  segments = [x for sent in segments for x in sent]

  sentences = utils.create_sentences(segments, MIN_WORDS=20, MAX_WORDS=80)
  chunks = utils.create_chunks(sentences, CHUNK_LENGTH=15, STRIDE=2)
  chunks_text = [chunk['text'] for chunk in chunks]

  # Run Stage 1 Summarizing
  stage_1_outputs = summarize_stage_1(chunks_text, handler=handler, verbose=True)['stage_1_outputs']
  # Split the titles and summaries
  stage_1_summaries = [e['summary'] for e in stage_1_outputs]
  stage_1_titles = [e['title'] for e in stage_1_outputs]
  num_1_chunks = len(stage_1_summaries)

  # summary and title embeddings
  summary_embeds = encode(stage_1_summaries).numpy()
  # title_embeds = encode(stage_1_titles).numpy()

  # Get similarity matrix between the embeddings of the chunk summaries
  summary_similarity_matrix = np.zeros((num_1_chunks, num_1_chunks))
  summary_similarity_matrix[:] = np.nan

  for row in range(num_1_chunks):
    for col in range(row, num_1_chunks):
      # Calculate cosine similarity between the two vectors
      similarity = 1- cosine(summary_embeds[row], summary_embeds[col])
      summary_similarity_matrix[row, col] = similarity
      summary_similarity_matrix[col, row] = similarity


  # Run the community detection algorithm
  # Set num_topics to be 1/4 of the number of chunks, or 8, which ever is smaller
  #num_topics = min(int(num_1_chunks / 4), 8)
  num_topics = num_1_chunks // 4
  topics_out = utils.get_topics(summary_similarity_matrix, num_topics=num_topics, bonus_constant=0.2)
  chunk_topics = topics_out['chunk_topics']
  topics = topics_out['topics']

  # Query GPT-3 to get a summarized title for each topic_data
  out = summarize_stage_2(stage_1_outputs, topics, summary_num_words = 250, handler=handler, verbose=True)
  markdown_summary = utils.json_to_md(out['stage_2_outputs'])

  return MainOut(
    stage_1_outputs=stage_1_outputs,
    stage_2_outputs=out['stage_2_outputs'],
    final_summary=out['final_summary'],
    markdown_summary=markdown_summary,
    summary_similarity_matrix=summary_similarity_matrix,
    chunk_topics=chunk_topics
  )



if __name__ == '__main__':
  # with open('chapter_8_biocomputing_reading.txt', 'r') as f: text = f.read()
  folder = 'outputs/lex_fridman_x_neil_gershinfeld'
  txt_fn = f'{folder}/transcription.txt'
  with open(txt_fn, 'r') as f: text = f.read()
  summary_obj = main(text)

  # save summary_obj
  import json
  with open(f'{folder}/stage_1_outputs.json', 'w') as f: json.dump(summary_obj.stage_1_outputs, f)
  with open(f'{folder}/stage_2_outputs.json', 'w') as f: json.dump(summary_obj.stage_2_outputs, f)
  with open(f'{folder}/final_summary.txt', 'w') as f: f.write(summary_obj.final_summary)
  with open(f'{folder}/markdown_summary.md', 'w') as f: f.write(summary_obj.markdown_summary)
  save_similarity_matrix_plot(summary_obj.summary_similarity_matrix, f'{folder}/summary_similarity_matrix.png')
  save_topics_plot(summary_obj.chunk_topics, f'{folder}/chunk_topics.png')

  # import pickle
  # with open(f'summary_obj_v{VERSION}.pkl', 'wb') as f: pickle.dump(summary_obj, f)
  # main(pdf_to_text('denning.pdf'))
