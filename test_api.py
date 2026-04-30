import requests

# 🔁 Change this when deployed
URL = "http://127.0.0.1:8000/predict"

def test_api():
    sample_texts = [
        "Breaking news: government releases new policy",
        "Shocking truth they don't want you to know!!!",
        "Scientists publish new research in Nature journal"
    ]

    for text in sample_texts:
        response = requests.post(URL, json={"text": text})

        print("\nText:", text)
        print("Status Code:", response.status_code)
        print("Prediction:", response.json())

if __name__ == "__main__":
    test_api()