import sys
sys.path.append('..')
from src import encoder_service
import json

filename = sys.argv[1]
with open(filename, 'r') as f:
  stage_1 = json.load(f)

summary_embeddings = encoder_service.encode(list(map(lambda x: x['summary'], stage_1)))

# save using pickle
import pickle
with open('nous_hermes_mixtral_stage_1_summary_embeddings.pkl', 'wb') as f:
  pickle.dump(summary_embeddings, f)