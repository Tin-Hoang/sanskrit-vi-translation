#!/usr/bin/env python3
import os
import requests
from dotenv import load_dotenv
import json


def list_groq_models():
    # Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("Error: GROQ_API_KEY not found in environment variables.")
        print("Please ensure you have a .env file with GROQ_API_KEY defined.")
        return

    url = "https://api.groq.com/openai/v1/models"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()
        models = data.get("data", [])

        # Sort models by ID for easier reading
        models.sort(key=lambda x: x["id"])

        print(f"\nFound {len(models)} available Groq models:\n")
        print(f"{'Model ID':<50} {'Owned By':<20} {'Context Window':<15}")
        print("-" * 85)

        for model in models:
            model_id = model["id"]
            owned_by = model["owned_by"]
            context = str(model.get("context_window", "N/A"))
            print(f"{model_id:<50} {owned_by:<20} {context:<15}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching models: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")


if __name__ == "__main__":
    list_groq_models()
