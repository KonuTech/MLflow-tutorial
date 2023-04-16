import requests
import json

query_json = {
  "inputs": [
    [
      -0.09269547780327612,
      -0.044641636506989144,
      0.028284032228378497,
      -0.015998975220305175,
      0.03695772020942014,
      0.02499059336410222,
      0.05600337505832251,
      -0.03949338287409329,
      -0.005142189801713891,
      -0.0010776975004659671
    ]
  ]
}
query = json.dumps(query_json)

headers = {'Content-Type': 'application/json'}
request_uri = 'http://127.0.0.1:5001/invocations'

response = requests.post(request_uri, data=query, headers=headers)

print(response.content)
