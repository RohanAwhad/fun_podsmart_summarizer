from openai import OpenAI
client = OpenAI()

def run(text, filename):
  """
  Converts the given text to speech and saves it to the specified file.

  Args:
    text (str): The text to be converted to speech.
    filename (str): The name of the file to save the speech to.

  Returns:
    None
  """
  response = client.audio.speech.create(
    model="tts-1",
    voice="echo",
    input=text
  )
  response.stream_to_file(filename)


if __name__ == '__main__':
  #with open('/Users/rohan/1_Project/fun_podsmart_summarizer/outputs/biocomputing_chapter_8_reading/chapter_8_biocomputing_reading.md', 'r') as f: text = f.read()
  #run(text, 'chapter_8_biocomputing_reading.mp3')

  import json
  data = []
  with open('/Users/rohan/1_Project/fun_podsmart_summarizer/outputs/biocomputing_chapter_8_reading/stage_2_outputs.jsonl', 'r') as f:
    for line in f.readlines():
      data.append(json.loads(line.strip()))

  for d in data:
    title, text = d['title'], d['summary']
    print(title)
    run(text, f'{title}.mp3')