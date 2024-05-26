import requests

url = "https://hyprace-api.p.rapidapi.com/v1/circuits"

headers = {
	"X-RapidAPI-Key": "de627efef9mshde499fcfa9562cep1a1ec3jsnc7f80949ebb4",
	"X-RapidAPI-Host": "hyprace-api.p.rapidapi.com"
}

response = requests.get(url, headers=headers)

print(response.json())