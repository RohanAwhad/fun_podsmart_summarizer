import json
import sys

import jinja2
sys.path.append('..')
from src.fun_llm_service.class_model import LLMParams, LLM
from src.fun_llm_service.service import generate

blog_url = 'https://a16z.com/emerging-architectures-for-llm-applications/'
reader_url = 'https://fun-readable-cd6ed9e43b50.herokuapp.com/convert'
import requests

res = requests.post(reader_url, json={'url': blog_url, 'is_blog': True})
res.status_code

from src import utils
from nltk.tokenize import sent_tokenize

text = res.json()['text']
# split into sentences
sentences = sent_tokenize(text)
# further split by comma
segments = [sent.split(',') for sent in sentences]
# flaten
segments = [x for sent in segments for x in sent]

sentences = utils.create_sentences(segments, MIN_WORDS=20, MAX_WORDS=80)
chunks = utils.create_chunks(sentences, CHUNK_LENGTH=5, STRIDE=1)
chunks_text = [chunk['text'] for chunk in chunks]

import os

# llm configs
together_api_url = "https://api.together.xyz/v1/completions"
together_api_key = os.environ['TOGETHER_API_KEY']

openai_api_url = "https://api.openai.com/v1/chat/completions"
openai_api_key = os.environ['OPENAI_API_KEY']

gpt_prompt = (
  'Summarize the below text in one sentence and give title.\n'
  'Respond in the following JSON format:\n'
  '{\n  "summary": <summary of the text>",\n  "title": <short title for the summary>\n}\n'
  'eg.\n{\n  "summary": "AI can make humans more productive by automating many repetitive processes.",\n  "title": "Why Artificial Intelligence is Good"\n}\n'
  '---\n'
  '\'\'\'text\n'
  '{{ text }}\n'
  '\'\'\'\n'
  '---\n'
  'Response:\n'
)
env = jinja2.Environment()
gpt_prompt = env.from_string(gpt_prompt)

gpt_4_params = LLMParams(
  max_tokens=300,
  temperature=0.5,
  top_p=0.9,
  top_k=50,
  stop=['---']
)


gpt_4 = LLM(
  url=openai_api_url,
  api_key=openai_api_key,
  model='gpt-4-turbo-preview',
  llm_params=gpt_4_params,
  prompt_template=gpt_prompt
)

def parse_title_summary_results(ai_response: str):
  start_idx = ai_response.find('{')
  end_idx = ai_response.rfind('}') + 1
  return json.loads(ai_response[start_idx:end_idx])

# generate one sample
# inputs = {'text': chunks_text[0]}
# res = gpt_4._generate(inputs, verbose=True)
# print(parse_title_summary_results(gpt_4.get_text_from_response(res)))
# exit(0)



# generate all
inputs = [[{'text': chunk_txt}] for chunk_txt in chunks_text]
llms = [gpt_4]
outputs = generate(inputs, llms, max_retries=2, total_timeout=20)


with open('gpt_4_summary_1_stage_1.txt', 'w') as f:
  for i, output in enumerate(outputs):
    f.write(f'Chunk {i+1}\n')
    f.write(f'Input: {chunks_text[i]}\n')
    f.write(f'Output: {output}\n')
    x = parse_title_summary_results(output)
    f.write(f'Title: {x["title"]}\n')
    f.write(f'Summary: {x["summary"]}\n')
    f.write('\n===\n')


with open('gpt_4_summary_1_stage_1.json', 'w') as f:
  json.dump(list(map(parse_title_summary_results, outputs)), f, indent=2)