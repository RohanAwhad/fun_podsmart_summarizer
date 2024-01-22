from main import main as run
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class LambdaOut(BaseModel):
  stages: List[Dict[str, Any]]
  final_summary: Optional[str] = None
  markdown_summary: str
  status: int = 200
  error: Optional[str] = None

def handler(event, context):
  text = event['text'].strip()
  try:
    out = run(text)
    stages = [out.stage_1_outputs, out.stage_2_outputs]
    stages = [{'outputs': s} for s in stages]
    stages[0]['chunk_topics'] = out.chunk_topics
    return LambdaOut(
      stages=stages,
      final_summary=out.final_summary,
      markdown_summary=out.markdown_summary
    ).model_dump_json()
  except Exception as e:
    return LambdaOut(
      stages=[],
      markdown_summary='',
      status=500,
      error=str(e)
    ).model_dump_json()
