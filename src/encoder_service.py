import aiohttp
import asyncio
import torch

from loguru import logger
from tqdm import tqdm
from typing import Union, List


url = "https://fun-sentence-embedder-c8f3c4818216.herokuapp.com"


def init(): pass

class _Encoder:
  def __init__(
    self,
    url,
    session: aiohttp.ClientSession,
    text_list: List[str],
    batch_size: int=8,
    num_workers: int=8
  ):

    self.url = url
    self.session = session
    self.text_list = text_list
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.pbar = tqdm(total=len(text_list) // batch_size, desc="Encoding")

    self._todo = asyncio.Queue()
    self.responses = []

  async def _encode_batch(self) -> torch.Tensor:
    for i in range(0, len(self.text_list), self.batch_size):
      text_batch = self.text_list[i : i + self.batch_size]
      await self._todo.put((text_batch))
    
    workers = [asyncio.create_task(self.worker()) for _ in range(self.num_workers)]
    await self._todo.join()
    for w in workers: w.cancel()
    return torch.cat([torch.tensor(res['embeddings']) for res in self.responses], dim=0)

  async def worker(self):
    while True:
      try: await self.process_one()
      except asyncio.CancelledError: return

  async def process_one(self):
    text_batch = await self._todo.get()
    try:
      async with self.session.post(self.url + "/embed_batch", json=text_batch) as response:
        if response.status != 200: raise Exception(f"Failed to encode text: {text_batch}. Got text: {await response.text()}")
        self.responses.append(await response.json())
    except Exception as e: logger.error(e)
    finally:
      self.pbar.update(1)
      self._todo.task_done()
        

async def _encode(text: Union[str, list[str]], batch_size: int=8) -> torch.Tensor:
  async with aiohttp.ClientSession() as session:
    _encoder = _Encoder(url, session, text, batch_size=batch_size, num_workers=8)
    return await _encoder._encode_batch()

def encode(text: Union[str, list[str]]) -> torch.Tensor:
  if isinstance(text, str): text = [text]
  return asyncio.run(_encode(text, batch_size=8))

if __name__ == '__main__':
  # testing service
  text_list = ['hello world'] * 1000
  res = encode(text_list)
  print(res.shape)

