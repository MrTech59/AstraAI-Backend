import requests

class HomeAssistantClient:
    def __init__(self, url: str, token: str):
        self.url = url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def get_states(self) -> list[dict]:
        endpoint = f"{self.url}/api/states"
        response = requests.get(endpoint, headers=self.headers, timeout=5)
        response.raise_for_status()
        return response.json()

    def call_service(self, domain: str, service: str, entity_id: str, data: dict = None) -> None:
        endpoint = f"{self.url}/api/services/{domain}/{service}"
        payload = {"entity_id": entity_id}
        if data:
            payload.update(data)
        
        response = requests.post(endpoint, headers=self.headers, json=payload, timeout=5)
        response.raise_for_status()
