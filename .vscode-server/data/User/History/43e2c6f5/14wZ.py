import requests
import json

def test_api():
    # Base URL for the API
    base_url = "http://localhost:8000"
    
    # Test GET request
    message = "Hello, how are you?"
    response = requests.get(f"{base_url}/chat", params={
        "message": message,
        "conversation_id": "test-conv-1"
    })
    print("\nGET Request:")
    print(f"Message: {message}")
    print(f"Response: {response.json()}")
    
    # Test POST request
    message = "What's the weather like?"
    response = requests.post(
        f"{base_url}/chat",
        json={
            "message": message,
            "conversation_id": "test-conv-2"
        }
    )
    print("\nPOST Request:")
    print(f"Message: {message}")
    print(f"Response: {response.json()}")
    
    # List conversations
    response = requests.get(f"{base_url}/conversations")
    print("\nActive Conversations:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_api()