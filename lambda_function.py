import boto3
import json
import os

from main import main as run
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class LambdaOut(BaseModel):
  stages: List[Dict[str, Any]]
  final_summary: Optional[str] = None
  markdown_summary: str
  status: int = 200
  error: Optional[str] = None

  # input data passing on
  url: str
  title: str
  text: str

sqs = boto3.client('sqs')
inp_sqs_url = os.environ['INP_SQS_URL']

sns = boto3.client('sns')
out_sns_topic_arn = os.environ['OUT_SNS_TOPIC_ARN']

def handler(event, context):
  if not event: return
  batch_item_failures = []

  for message in event['Records']:
    body = json.loads(message['body'])
    inp_data = json.loads(body['Message'])
    text = inp_data['text'].strip()
    receipt_handle = message['receiptHandle']
    try:
      out = run(text)
      stages = [out.stage_1_outputs, out.stage_2_outputs]
      stages = [{'outputs': s} for s in stages]
      stages[0]['chunk_topics'] = out.chunk_topics

      # write to an sns topic
      sns.publish(
        TopicArn=out_sns_topic_arn,
        Message=LambdaOut(
          stages=stages,
          final_summary=out.final_summary,
          markdown_summary=out.markdown_summary,
          url=inp_data['url'],
          title=inp_data['title'],
          text=inp_data['text']
        ).model_dump_json()
      )

      # delete message from sqs
      sqs.delete_message(
        QueueUrl=inp_sqs_url,
        ReceiptHandle=receipt_handle
      )
    except Exception as e:
      print(e)
      batch_item_failures.append({'itemIdentifier':message['messageId']})

  return {'batchItemFailures': batch_item_failures}
