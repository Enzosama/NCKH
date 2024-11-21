import requests

BASE_URL = "http://localhost:8000"

def test_chat_endpoint():
    print("Testing /chat endpoint...")

    # Gửi yêu cầu POST tới endpoint /chat
    url = f"{BASE_URL}/chat"
    payload = {
        "message": "What is this document about?",
        "conversation_id": "test-conversation"
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Kiểm tra nếu có lỗi HTTP
        data = response.json()
        print("Response:", data)
    except requests.exceptions.RequestException as e:
        print("Error while testing /chat endpoint:", e)

def test_pdf_upload():
    print("Testing /upload_pdf endpoint...")

    # Gửi yêu cầu POST tới endpoint /upload_pdf với tệp PDF
    url = f"{BASE_URL}/upload_pdf"
    files = {"file": open("test.pdf", "rb")}

    try:
        response = requests.post(url, files=files)
        response.raise_for_status()  # Kiểm tra nếu có lỗi HTTP
        data = response.json()
        print("Response:", data)
    except requests.exceptions.RequestException as e:
        print("Error while testing /upload_pdf endpoint:", e)

def main():
    print("Starting API tests...")
    
    # Kiểm tra từng endpoint
    test_chat_endpoint()
    test_pdf_upload()

if __name__ == "__main__":
    main()
