import ollama
from flask import Flask, request, jsonify

class OllamaSession:
    def __init__(self, session_id, model_name, system_prompt=""):
        self.session_id = session_id
        self.model_name = model_name

        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def chat(self, user_message):
        self.messages.append({"role": "user", "content": user_message})

        stream = ollama.chat(
            model=self.model_name,
            messages=self.messages,
            stream=True,
            options={
                "num_ctx": 8192,
                "temperature": 0.7,
            }
        )

        assistant_reply = ""
        for chunk in stream:
            content = chunk["message"]["content"]
            assistant_reply += content

        self.messages.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply

class SessionManager:
    def __init__(self):
        self.sessions: dict[int, OllamaSession] = {}
        self.next_session_id = 1

    def get_session(self, session_id) -> OllamaSession:
        return self.sessions[session_id]
    
    def create_session(self, model_name, system_prompt) -> int:
        session_id = self.next_session_id
        self.next_session_id += 1
        self.sessions[session_id] = OllamaSession(session_id, model_name, system_prompt)
        return session_id

session_manager = SessionManager()

app = Flask(__name__)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    session_id = data.get("session_id")
    user_message = data.get("message")

    session = session_manager.get_session(session_id)
    response = session.chat(user_message)
    return jsonify({"response": response})

@app.route("/api/create_session", methods=["POST"])
def api_create_session():
    data = request.json
    model_name = data.get("model", "gpt-oss:120b-cloud")
    system_prompt = data.get("system_prompt", "")
    session_id = session_manager.create_session(model_name, system_prompt)
    return jsonify({"session_id": session_id})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8024)
