# api.py
import sys
import json
from pipeline import Pipeline
import sys
sys.stdout.reconfigure(encoding='utf-8')

def process_code(input_json):
    try:
        data = json.loads(input_json)

        source_code = data.get("code", "")
        mode = data.get("mode", "both")

        pipeline = Pipeline()
        result = pipeline.run(source_code, mode=mode)

        return json.dumps({
            "success": True,
            "result": result
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


if __name__ == "__main__":
    input_data = sys.stdin.read()
    output = process_code(input_data)
    print(output)