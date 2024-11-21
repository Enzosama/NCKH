import requests
import json

def test_api(base_url="http://localhost:8000"):
    """
    Comprehensive API testing function
    
    Args:
        base_url (str): Base URL of the API, defaults to localhost:8000
    """
    try:
        # Test root endpoint
        root_response = requests.get(f"{base_url}/")
        print("\nRoot Endpoint:")
        print(json.dumps(root_response.json(), indent=2))

        # Test GET chat with default context
        get_message = "What is Generative AI?"
        get_response = requests.get(f"{base_url}/chat", params={
            "message": get_message,
            "conversation_id": "test-conv-1",
            "pdf_context": "default"  # Use default PDF context
        })
        print("\nGET Request:")
        print(f"Message: {get_message}")
        print(json.dumps(get_response.json(), indent=2))

        # Test POST chat with default context
        post_message = "Tell me about AI in cloud computing"
        post_response = requests.post(
            f"{base_url}/chat", 
            json={
                "message": post_message,
                "conversation_id": "test-conv-2"
            },
            params={"pdf_context": "default"}
        )
        print("\nPOST Request:")
        print(f"Message: {post_message}")
        print(json.dumps(post_response.json(), indent=2))

        # Optional: Test conversations endpoint if implemented
        try:
            conv_response = requests.get(f"{base_url}/conversations")
            print("\nActive Conversations:")
            print(json.dumps(conv_response.json(), indent=2))
        except Exception as conv_err:
            print(f"\nConversations endpoint not available: {conv_err}")

    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}")
    except json.JSONDecodeError as je:
        print(f"JSON Decoding Error: {je}")
    except Exception as err:
        print(f"Unexpected error: {err}")

def upload_test_pdf(base_url="http://localhost:8000"):
    """
    Test PDF upload functionality
    
    Args:
        base_url (str): Base URL of the API, defaults to localhost:8000
    """
    try:
        # Specify the path to your PDF file
        pdf_path = "651778122-The-Ultimate-Guide-to-Generative-AI-Studio-on-Google-Cloud-s-Vertex-AI.pdf"
        
        with open(pdf_path, 'rb') as pdf_file:
            files = {'file': (pdf_path, pdf_file, 'application/pdf')}
            upload_response = requests.post(f"{base_url}/upload-pdf", files=files)
        
        print("\nPDF Upload Test:")
        print(json.dumps(upload_response.json(), indent=2))
        
        # Test chat with uploaded PDF context
        if upload_response.status_code == 200:
            chat_response = requests.get(f"{base_url}/chat", params={
                "message": "What is the main topic of this document?",
                "pdf_context": pdf_path
            })
            print("\nChat with Uploaded PDF:")
            print(json.dumps(chat_response.json(), indent=2))
    
    except FileNotFoundError:
        print("PDF file not found. Please check the file path.")
    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}")

if __name__ == "__main__":

    upload_test_pdf()
