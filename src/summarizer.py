import os

from datetime import datetime
from langchain_community.chat_models import ChatOpenAI as OpenAI
from langchain_together import Together
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

from . import together_service
from . import utils

# model_name = 'togethercomputer/llama-2-7b-chat'
# model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
model_name = 'gpt-3.5-turbo-1106'

STOP_TOKENS = ['---']
SUMMARY_STAGE_1_MAP_PROMPT = (
'[INST]\n'
'Write a concise summary for text enclosed in triple backticks (```)\n'
'- Output the summary in bullet points.\n'
'- Each point should contain no more than 2 sentences.\n'
'- Keep the summary concise.\n'
'- Points should not contain redundant information.\n'
'- Do not use pronouns. Resolve coreference and use proper nouns or common nouns.\n'
'- Summary should be grammatically correct.\n'
'\n'
'```\n'
'{text}\n'
'```\n'
'\n'
'Return your answer in the following format:\n'
'Title: <title>\n'
'Summary:\n'
'- <summary point 1>\n'
'- <summary point 2>\n'
'- <summary point n>\n'
'e.g.\n'
'Title: Why Artificial Intelligence is Good\n'
'Summary:\n'
'- AI can make humans more productive by automating many repetitive processes.\n'
'\n'
'---\n'
'TITLE AND CONCISE SUMMARY:\n'
'[/INST]\n'
)
SUMMARY_STAGE_2_MAP_PROMPT = (
  'Write a concise summary for text enclosed in triple backticks (```)\n'
  '\n'
  '```\n'
  '{text}\n'
  '```\n'
  '\n'
  '---\n'
  'Summary:\n'
)

SUMMARY_STAGE_1_MAP_LLM = Together(
  temperature=0.4,
  model='mistralai/Mixtral-8x7B-Instruct-v0.1',
  max_tokens=1024,
  top_p=0.6,
)
SUMMARY_STAGE_2_MAP_LLM = Together(
  temperature=0,
  model='mistralai/Mixtral-8x7B-Instruct-v0.1',
  max_tokens=1024,
  top_p=0.6,
)
SUMMARY_STAGE_2_TITLE_LLM = OpenAI(temperature=0, model_name=model_name)
SUMMARY_STAGE_2_REDUCE_LLM = OpenAI(temperature=0, model_name=model_name, max_tokens = 1024)


def summarize_stage_1(chunks_text, handler=None, verbose=False):
  if handler is None: handler = []
  elif not isinstance(handler, list): handler = [handler]
  
  print(f'Start time: {datetime.now()}')
  llm_chain_input = [SUMMARY_STAGE_1_MAP_PROMPT.format(text=t) for t in chunks_text]
  map_llm_chain_results = together_service.generate(
    llm_chain_input,
    'mistralai/Mixtral-8x7B-Instruct-v0.1',
    max_tokens=1024,
    top_p=0.6,
    temperature=0.4,
    stop_tokens=['---']
  )
  stage_1_outputs = utils.parse_title_summary_results(map_llm_chain_results)

  print(f'Stage 1 done time {datetime.now()}')
  return {'stage_1_outputs': stage_1_outputs}



def summarize_stage_2(stage_1_outputs, topics, summary_num_words = 250, handler=None, verbose=False):
  if handler is None: handler = []
  elif not isinstance(handler, list): handler = [handler]

  print(f'Stage 2 start time {datetime.now()}')
  
  # Prompt that passes in all the titles of a topic, and asks for an overall title of the topic
  title_prompt_template = """Write an informative title that summarizes each of the following groups of titles. Make sure that the titles capture as much information as possible, 
  and are different from each other:
  {text}
  
  Return your answer in a numbered list, with new line separating each title: 
  1. Title 1
  2. Title 2
  3. Title 3

  TITLES:
  """

  combine_prompt_template = 'Write a ' + str(summary_num_words) + """-word summary of the following, removing irrelevant information. Finish your answer:
  {text}
  """ + str(summary_num_words) + """-WORD SUMMARY:"""

  title_prompt = PromptTemplate(template=title_prompt_template, input_variables=["text"])
  map_prompt = PromptTemplate(template=SUMMARY_STAGE_2_MAP_PROMPT, input_variables=["text"])
  combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

  # === Get Titles ===
  # Get a list of each community's summaries and titles
  topics_data = []
  for c in topics:
    topic_data = {
      'summaries': [stage_1_outputs[chunk_id]['summary'] for chunk_id in c],
      'titles': [stage_1_outputs[chunk_id]['title'] for chunk_id in c]
    }
    topic_data['summaries_concat'] = ' '.join(topic_data['summaries'])
    topic_data['titles_concat'] = ', '.join(topic_data['titles'])
    topics_data.append(topic_data)
    
  # Get a list of each community's summaries (concatenated)
  topics_summary_concat = [c['summaries_concat'] for c in topics_data]
  topics_titles_concat = [c['titles_concat'] for c in topics_data]

  # Concat into one long string to do the topic title creation
  topics_titles_concat_all = ''''''
  for i, c in enumerate(topics_titles_concat):
    topics_titles_concat_all += f'''{i+1}. {c}
    '''
  
  # print('topics_titles_concat_all', topics_titles_concat_all)

  title_llm_chain = LLMChain(llm = SUMMARY_STAGE_2_TITLE_LLM, prompt = title_prompt, callbacks=handler, verbose=verbose)
  title_llm_chain_input = [{'text': topics_titles_concat_all, 'stop': STOP_TOKENS}]
  title_llm_chain_results = title_llm_chain.apply(title_llm_chain_input)
  
  
  # Split by new line
  titles = title_llm_chain_results[0]['text'].split('\n')
  # Remove any empty titles
  titles = [t for t in titles if t != '']
  # Remove spaces at start or end of each title
  titles = [t.strip() for t in titles]

  # # === Get Summaries ===
  # # Run the map-reduce chain
  # docs = [Document(page_content=t) for t in topics_summary_concat]
  # chain = load_summarize_chain(
  #   chain_type="map_reduce",
  #   map_prompt=map_prompt,
  #   combine_prompt=combine_prompt,
  #   return_intermediate_steps=True,
  #   llm=SUMMARY_STAGE_2_MAP_LLM,
  #   reduce_llm=SUMMARY_STAGE_2_REDUCE_LLM,
  #   callbacks=handler,
  #   verbose=verbose,
  #   # stop=["---"],  # TODO (rohan): find a way to add stop tokens to the chain
  # )

  # output = chain({"input_documents": docs, 'stop': STOP_TOKENS}, return_only_outputs = True)
  # final_summary = output['output_text']
  # summaries = output['intermediate_steps']
  # stage_2_outputs = [{'title': t, 'summary': s.strip()} for t, s in zip(titles, summaries)]

  # === My Implementation for stage 2 summaries ===
  input_docs = [SUMMARY_STAGE_2_MAP_PROMPT.format(text=t) for t in topics_summary_concat]
  summaries = together_service.generate(
    input_docs,
    'mistralai/Mixtral-8x7B-Instruct-v0.1',
    max_tokens=1024,
    top_p=0.6,
    temperature=0,
    stop_tokens=['---']
  )
  stage_2_outputs = [{'title': t, 'summary': s.strip()} for t, s in zip(titles, summaries)]

  # === My Implementation for final summary ===
  final_summary_prompt = (
    'Write a 250-word summary of the following, removing irrelevant information. Finish your answer:\n'
    '\n'
    '{text}\n'
    '---\n'
    '250-WORD SUMMARY:'
  )
  final_summary = SUMMARY_STAGE_2_REDUCE_LLM.invoke(final_summary_prompt).content

  # === Post Processing ===
  # if summaries in stage_2_outputs end with '---', remove it
  for i, s in enumerate(stage_2_outputs):
    if '---' in s['summary']:
      _ = s['summary'].find('---')
      stage_2_outputs[i]['summary'] = s['summary'][:_].strip()



  # Return: stage_1_outputs (title and summary), stage_2_outputs (title and summary), final_summary, chunk_allocations
  out = {'stage_2_outputs': stage_2_outputs, 'final_summary': final_summary}
  print(f'Stage 2 done time {datetime.now()}')
  return out

