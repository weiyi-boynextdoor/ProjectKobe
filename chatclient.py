import requests

def create_session():
    url = "http://127.0.0.1:8024/api/create_session"
    payload = {"model": "gpt-oss:120b-cloud"}

    session_id = None
    
    try:
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            response_data = response.json()
            session_id = response_data.get("session_id")
        else:
            print(f"create_session error: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("Cannot connect to server")

    return session_id

def send_message(session_id, message):
    url = f"http://127.0.0.1:8024/api/chat"
    payload = {"session_id": session_id, "message": message}

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f"Message sent successfully: {response.json()['response']}")
        else:
            print(f"send_message error: {response.status_code}")

    except requests.exceptions.ConnectionError:
        print("Cannot connect to server")

if __name__ == "__main__":
    session_id = create_session()
    print(f"Created session with ID: {session_id}")
    if session_id:
        while True:
            user_input = input("User: ")
            send_message(session_id, user_input)
