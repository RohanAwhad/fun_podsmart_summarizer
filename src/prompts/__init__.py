import jinja2
env = jinja2.Environment()

with open('./src/prompts/stage_1/nous_hermes_mixtral.txt', 'r') as f:
  nous_hermes_mixtral = env.from_string(f.read())

with open('./src/prompts/stage_1/gpt_3_5.txt', 'r') as f:
  gpt_3_5 = env.from_string(f.read())

with open('./src/prompts/stage_1/gpt_4.txt', 'r') as f:
  gpt_4 = env.from_string(f.read())

stage_1 = {
  'nous_hermes_mixtral': nous_hermes_mixtral,
  'gpt_3_5': gpt_3_5,
  'gpt_4': gpt_4
}