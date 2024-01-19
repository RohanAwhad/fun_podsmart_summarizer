# [Reference]: https://towardsdatascience.com/summarize-podcast-transcripts-and-long-texts-better-with-nlp-and-ai-e04c89d3b2cb
import langchain
langchain.debug = False
VERBOSE = False

import numpy as np

from langchain.callbacks import FileCallbackHandler
from loguru import logger
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel
from typing import Dict, List, Optional

from src import utils
from src.encoder_service import encode
from src.summarizer import summarize_stage_1, summarize_stage_2

VERSION = '1.4.2'

logfile = f'logs/main_v{VERSION}.log'
logger.add(logfile, colorize=True, enqueue=True)
handler = FileCallbackHandler(logfile)


class MainOut(BaseModel):
  stage_1_outputs: List[Dict[str, str]]
  stage_2_outputs: Optional[List[Dict[str, str]]] = None
  final_summary: Optional[str] = None
  markdown_summary: str
  summary_similarity_matrix: Optional[np.ndarray] = None
  chunk_topics: Optional[List[int]] = None

  class Config:
    arbitrary_types_allowed = True


def save_similarity_matrix_plot(similarity_matrix: np.ndarray, filename: str):
  import matplotlib.pyplot as plt
  # Draw a heatmap with the summary_similarity_matrix
  plt.figure()
  # Color scheme blues
  plt.imshow(similarity_matrix, cmap = 'Blues')
  # save the figure
  plt.savefig(filename, dpi=300)


def save_topics_plot(topics: List[List[int]], filename: str):
  import matplotlib.pyplot as plt
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
  stage_1_outputs = summarize_stage_1(chunks_text, handler=handler, verbose=VERBOSE)['stage_1_outputs']
  # Split the titles and summaries
  stage_1_summaries = [e['summary'] for e in stage_1_outputs]
  num_1_chunks = len(stage_1_summaries)

  # check if num_1_chunks < 8, if so, skip chunking, directly send to stage 2
  if num_1_chunks < 8:
    chunk_topics = [0] * num_1_chunks
    topics = [list(range(num_1_chunks))]
    summary_similarity_matrix = None
  else:
    # summary embeddings
    summary_embeds = encode(stage_1_summaries)

    # Get similarity matrix between the embeddings of the chunk summaries
    summary_similarity_matrix = utils.cosine_similarity(summary_embeds, summary_embeds)

    # Run the community detection algorithm
    # Set num_topics to be 1/4 of the number of chunks, or 8, which ever is smaller
    #num_topics = min(int(num_1_chunks / 4), 8)
    num_topics = num_1_chunks // 4
    topics_out = utils.get_topics(summary_similarity_matrix, num_topics=num_topics, bonus_constant=0.2)
    chunk_topics = topics_out['chunk_topics']
    topics = topics_out['topics']

  # Query GPT-3 to get a summarized title for each topic_data
  out = summarize_stage_2(stage_1_outputs, topics, summary_num_words = 250, handler=handler, verbose=VERBOSE)
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
  import json
  with open('tests/long_input.json', 'r') as f: text = json.load(f)['text']
  summary_obj = main(text)
