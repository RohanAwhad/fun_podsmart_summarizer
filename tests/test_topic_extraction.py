import numpy as np
from src import utils

def test_get_topics():
  # Create a random matrix of size (1000, 768)
  summary_embeds = np.load('tests/summary_embeds.npy')

  # Run the cosine similarity function
  simi_mat = utils.cosine_similarity(summary_embeds, summary_embeds)

  # Run the get_topics function
  topics_out = utils.get_topics(simi_mat, num_topics=8, bonus_constant=0.2)