import json

def run(jsonl_fn):
  # load jsonl
  data = []
  with open(jsonl_fn) as f:
    for line in f:
      data.append(json.loads(line))

  # create markdown with title as h3 and text as p
  ret = ''
  for d in data:
    ret += f'### {d["title"]}\n'
    ret += f'{d["summary"]}\n\n'

  return ret


if __name__ == '__main__':
  md = run('stage_2_outputs.jsonl')
  with open('chapter_8_biocomputing_reading.md', 'w') as f:
    f.write(md)

  