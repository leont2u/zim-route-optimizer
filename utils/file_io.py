import json

def save_json(data, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)
