import aiohttp
import asyncio

from loguru import logger
from tqdm import tqdm
from typing import Union, List, Optional
import numpy as np
import os

# together ai config
url = "https://api.together.xyz/v1/completions"
api_key = os.environ['TOGETHER_API_KEY']
NUM_WORKERS = int(os.environ.get('TOGETHER_NUM_WORKERS', 2))


def init(): pass

class _LLM:
  def __init__(
    self,
    url: str,
    llm: str,
    session: aiohttp.ClientSession,
    text_list: List[str],
    max_tokens: int=512,
    temperature: float=0.7,
    top_p: float=0.7,
    top_k: int=50,
    stop_tokens: Optional[List[str]]=None
  ):

    self.url = url
    self.session = session
    self.text_list = text_list
    self.num_workers = NUM_WORKERS
    self.pbar = tqdm(total=len(text_list), desc="Generating", leave=False)

    self._todo = asyncio.Queue()
    self.responses = []

    self.llm = llm
    self.max_tokens = max_tokens
    self.temperature = temperature
    self.top_p = top_p
    self.top_k = top_k
    self.stop_tokens = ['</s>', '[/INST]'] + stop_tokens if stop_tokens else ['</s>', '[/INST]']

    self.max_retries = 3

  async def _generate_batch(self) -> np.ndarray:
    outputs = []
    for txt in self.text_list: await self._todo.put((txt, 0))
    
    workers = [asyncio.create_task(self.worker()) for _ in range(self.num_workers)]
    await self._todo.join()
    for w in workers: w.cancel()
    
    for res in self.responses: outputs.append(res['choices'][0]['text'])
    return outputs

  async def worker(self):
    while True:
      try: await self.process_one()
      except asyncio.CancelledError: return

  async def process_one(self):
    prompt, _ = await self._todo.get()
    res = None
    backoff = 2
    interval = 0.5
    for attempt in range(self.max_retries):
      try:
        data = {
            "model": self.llm,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "stop": self.stop_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": 1,
            "n": 1
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": "Bearer " + api_key
        }

        async with self.session.post(self.url, json=data, headers=headers) as response:
          print('response', response)
          if response.status == 200:
            res = await response.json()
          raised_exc = None
      #except Exception as e:
        backoff_interval = interval * (backoff ** attempt)
      except aiohttp.ClientError as e:
        raise_exc = e
        logger.error(e)
        #if retry_attempts < self.max_retries:
        #  logger.warning(f"Retrying prompt ...")
        #  await self._todo.put((prompt, retry_attempts + 1))

      await asyncio.sleep(backoff_interval)

    self.pbar.update(1)
    self._todo.task_done()
    if res is None:
      raise Exception(f"Failed to generate text: {prompt}. Got text: {await response.text()}")
    else:
      self.responses.append(res)
        

async def _generate(
  text: Union[str, List[str]],
  llm: str,
  max_tokens: int=512,
  temperature: float=0.7,
  top_p: float=0.7,
  top_k: int=50,
  stop_tokens: Optional[List[str]]=None,
) -> List[str]:

  timeout = aiohttp.ClientTimeout(total=10)
  async with aiohttp.ClientSession(timeout=timeout) as session:
    _llm = _LLM(
      url,
      llm,
      session,
      text,
      max_tokens=max_tokens,
      temperature=temperature,
      top_p=top_p,
      top_k=top_k,
      stop_tokens=stop_tokens
    )
    return await _llm._generate_batch()

def generate(
  prompts: Union[str, List[str]],
  llm: str,
  max_tokens: int=512,
  temperature: float=0.7,
  top_p: float=0.7,
  top_k: int=50,
  stop_tokens: Optional[List[str]]=None
) -> List[str]:
  if isinstance(prompts, str): prompts = [prompts]
  return asyncio.run(_generate(
    prompts,
    llm=llm,
    max_tokens=max_tokens,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    stop_tokens=stop_tokens
  ))

if __name__ == '__main__':
  # testing service
  llm = 'something'
  print(generate(['<s>[INST] What is the capital of France? [/INST]'], llm))
  exit()
