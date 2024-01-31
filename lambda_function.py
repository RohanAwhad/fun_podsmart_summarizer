import boto3
import json
import os

from datetime import datetime, timezone
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from main import main as run

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

dynamodb = boto3.resource('dynamodb')

def handler(event, context):
  if not event: return
  batch_item_failures = []

  for message in event['Records']:
    body = json.loads(message['body'])
    inp_data = json.loads(body['Message'])
    url = inp_data['url']
    table_name = inp_data['table_name']

    # text = inp_data['text'].strip()
    receipt_handle = message['receiptHandle']
    try:
      # get text from dynamodb
      table = dynamodb.Table(table_name)
      response = table.get_item(Key={'url': url})
      text = response['Item']['page_text']

      # run the summarizer
      out = run(text)
      stages = [out.stage_1_outputs, out.stage_2_outputs]
      stages = [{'outputs': s} for s in stages]
      stages[0]['chunk_topics'] = out.chunk_topics

      # write to dynamodb
      updates = dict(
        updated_on=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        final_summary=out.final_summary,
        markdown_summary=out.markdown_summary,
        stages=stages
      )
      update_expression = [f'{x}=:{x}' for x in updates.keys()]  # ['title=:title', 'text=:text', ...]
      update_expression = 'SET ' + ', '.join(update_expression)  # 'SET title=:title, text=:text, ...'
      exp_attr_values = {f':{k}': v for k, v in updates.items()}  # {':title': '...', ':text': '...', ...}
      table.update_item(
        Key=dict(url=url),
        UpdateExpression=update_expression,
        ExpressionAttributeValues=exp_attr_values
      )

      # write to an sns topic
      sns_msg = {'url': url, 'table_name': table_name}
      sns.publish(
        TopicArn=out_sns_topic_arn,
        Message=json.dumps(sns_msg)
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
