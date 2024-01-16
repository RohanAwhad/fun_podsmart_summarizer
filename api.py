import os

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from main import main as run

app = FastAPI()
ACCESS_KEY = os.getenv('API_ACCESS_KEY')

class ReqIn(BaseModel):
  text: str
  access_key: str

class ResOut(BaseModel):
  markdown_summary: str
  error: Optional[str] = None


@app.post("/summarize")
def summarize(req: ReqIn):
  if req.access_key != ACCESS_KEY: return {'error': 'Invalid access key', 'markdown_summary': ''}
  summary_obj = run(req.text)
  return {'markdown_summary': summary_obj.markdown_summary}


if __name__ == '__main__':
  import uvicorn
  uvicorn.run('api:app', host='localhost', port=8000, reload=True)