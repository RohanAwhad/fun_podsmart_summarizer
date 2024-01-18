import numpy as np
import time
from scipy.spatial.distance import cosine 
from src.utils import cosine_similarity

def test_cosine_shape():
  # Create a random matrix of size (1000, 768)
  a = np.random.rand(1000, 768)
  b = np.random.rand(1000, 768)

  # Run the cosine similarity function
  actual = cosine_similarity(a, b)

  # Check that the results are the same
  assert actual.shape == (a.shape[0], b.shape[0])

def test_cosine():
  # Create a random matrix of size (1000, 768)
  n = 100
  a = np.random.rand(n, 768)
  b = np.random.rand(n, 768)

  # Run the cosine similarity function
  expected = np.zeros((n, n))
  for a_row in range(n):
    for b_row in range(a_row, n):
      simi = 1 - cosine(a[a_row], b[b_row])
      expected[a_row, b_row] = simi
      expected[b_row, a_row] = simi

  actual = cosine_similarity(a, b)

  # Check that the results are the same
  np.testing.assert_allclose(expected, actual, atol=1e-1, rtol=1)

def test_cosine_speed():
  # Create a random matrix of size (1000, 768)
  a = np.random.rand(1000, 768)
  b = np.random.rand(1000, 768)

  # Run the cosine similarity function
  start = time.time()
  cosine_similarity(a, b)
  end = time.time()

  # Check that the function is fast enough
  assert end - start < 1.0, 'Function is too slow'