import json
import pytest

from main import main as run

# starting encoder service
from src.encoder_service import encode
encode('hello world')

import os
IS_SKIP = int(os.environ.get('IS_SKIP', 1))

@pytest.mark.skipif(IS_SKIP, reason="running this costs money")
@pytest.mark.timeout(60)
def test_short_input():
  with open('tests/short_input.json', 'r') as f: text = json.load(f)['text']
  run(text)

@pytest.mark.skipif(IS_SKIP, reason="running this costs money")
@pytest.mark.timeout(60)
def test_long_input():
  with open('tests/long_input.json', 'r') as f: text = json.load(f)['text']
  run(text)