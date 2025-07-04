import requests

url = "http://127.0.0.1:8000/predict/"
payload = {
    "features": [0.1, 1.2, 3.4, 0.5, 2.2, 1.1, 0.0, 0.3, 1.9, 2.5, 0.7, 0.2,
                 3.1, 4.2, 2.2, 1.8, 0.6, 2.4, 3.3, 0.9, 1.5, 2.0, 0.4, 0.1,
                 2.1, 3.7, 1.6, 0.8, 0.2, 1.3]  # exactly 30 float features
}

response = requests.post(url, json=payload)
print(response.json())
