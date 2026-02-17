import ollama
from flask import Flask, request, jsonify, send_from_directory
import torch
import time
from tts import Qwen3TTS

tts = Qwen3TTS()

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
    print("Generating voice...")
    start_time = time.time()
    voice_file = f"response_{session_id}.wav"
    tts.generate_voice_clone(response, f"audio_output/{voice_file}")
    elapsed_time = time.time() - start_time
    print(f"generate_voice_clone took {elapsed_time:.2f} seconds")
    return jsonify({"response": response, "voice_file": voice_file})

@app.route("/api/create_session", methods=["POST"])
def api_create_session():
    data = request.json
    model_name = data.get("model", "gpt-oss:120b-cloud")
    system_prompt = data.get("system_prompt", "")
    session_id = session_manager.create_session(model_name, system_prompt)
    return jsonify({"session_id": session_id})

@app.route('/download_voice/<filename>')
def download_voice(filename):
    # as_attachment=True 会强制浏览器下载，
    # 但对于 UE5 来说，设为 False 直接流式读取内容通常更方便
    return send_from_directory("audio_output", filename, as_attachment=False)

def voice_clone():
    ref_audio = "./audio_input/Mamba.wav"
    ref_text  = "Man! ha ha ha ha ha ha ha. What can I say? Mamba out!"
    tts.create_voice_clone_prompt(ref_audio, ref_text)

if __name__ == "__main__":
    voice_clone()
    app.run(host="0.0.0.0", port=8024)
