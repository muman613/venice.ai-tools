import json
import os
import requests

api_key = os.environ["VENICE_API_KEY"]

response = requests.get(
    "https://api.venice.ai/api/v1/models",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    },
    timeout=60,
)

response.raise_for_status()
data = response.json()

models = data.get("data", data)

if isinstance(models, dict):
    models = list(models.values())

for model in models:
    text = json.dumps(model, ensure_ascii=False).lower()

    if "wan" in text and any(
        version in text
        for version in ("2.7", "2-7", "2_7", "wan27")
    ):
        print(json.dumps(model, indent=2, ensure_ascii=False))