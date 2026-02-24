import os
import ollama
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import tts
import globals
import time

load_dotenv()

tts_module = None

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
    print("Assistant response:", response)
    if tts_module:
        print("Generating voice...")
        start_time = time.time()
        voice_file = f"response_{session_id}.wav"
        tts_module.generate_voice_clone(response, f"audio_output/{voice_file}")
        elapsed_time = time.time() - start_time
        print(f"generate_voice_clone took {elapsed_time:.2f} seconds")
        return jsonify({"response": response, "voice_file": voice_file})
    else:
        return jsonify({"response": response})

@app.route("/api/create_session", methods=["POST"])
def api_create_session():
    session_id = session_manager.create_session(globals.config.get("llm", "model"), globals.config.get("llm", "system_prompt"))
    return jsonify({"session_id": session_id})

@app.route('/download_voice/<filename>')
def download_voice(filename):
    # as_attachment=True 会强制浏览器下载，
    # 但对于 UE5 来说，设为 False 直接流式读取内容通常更方便
    return send_from_directory("audio_output", filename, as_attachment=False)

if __name__ == "__main__":
    globals.config.read("config/config.ini")
    tts_type = globals.config.get("tts", "type")
    tts_module = tts.get_tts_module(tts_type)
    if tts_module:
        os.makedirs("audio_output", exist_ok=True)

    app.run(host=globals.config.get("host", "ip"), port=globals.config.getint("host", "port"))
