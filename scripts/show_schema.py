import json

with open("configs/schema_v1.json") as f:
    schema = json.load(f)

for key, value in schema.items():
    print("\n", key)
    print("-" * 40)

    for item in value:
        print(item)
