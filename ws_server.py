import os
import json
import ssl
import asyncio
import websockets
import ollama
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
import uvicorn
from dotenv import load_dotenv
import globals

load_dotenv()

TTS_MODEL = "speech-2.8-hd"
TTS_FILE_FORMAT = "mp3"


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
        return self.sessions.get(session_id)

    def create_session(self, model_name, system_prompt) -> int:
        session_id = self.next_session_id
        self.next_session_id += 1
        self.sessions[session_id] = OllamaSession(session_id, model_name, system_prompt)
        return session_id


session_manager = SessionManager()


async def establish_minimax_connection(api_key):
    """Establish WebSocket connection to Minimax TTS API"""
    url = "wss://api.minimax.io/ws/v1/t2a_v2"
    headers = {"Authorization": f"Bearer {api_key}"}

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    ws = await websockets.connect(url, additional_headers=headers, ssl=ssl_context)
    connected = json.loads(await ws.recv())
    if connected.get("event") == "connected_success":
        print("Minimax TTS connected")
        return ws
    return None


async def start_tts_task(tts_ws, voice_id):
    """Send task_start to Minimax TTS"""
    start_msg = {
        "event": "task_start",
        "model": TTS_MODEL,
        "voice_setting": {
            "voice_id": voice_id,
            "speed": 1,
            "vol": 1,
            "pitch": 0,
            "english_normalization": False
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": TTS_FILE_FORMAT,
            "channel": 1
        }
    }
    await tts_ws.send(json.dumps(start_msg))
    response = json.loads(await tts_ws.recv())
    return response.get("event") == "task_started"


async def stream_tts_to_client(tts_ws, text, client_ws: WebSocket):
    """Send text to Minimax and forward audio chunks to client as hex strings"""
    await tts_ws.send(json.dumps({
        "event": "task_continue",
        "text": text
    }))

    chunk_counter = 1
    while True:
        try:
            response = json.loads(await tts_ws.recv())

            if "data" in response and "audio" in response["data"]:
                audio_hex = response["data"]["audio"]
                if audio_hex:
                    print(f"Sending audio chunk #{chunk_counter}")
                    await client_ws.send_json({
                        "event": "audio_chunk",
                        "data": audio_hex,
                        "format": TTS_FILE_FORMAT
                    })
                    chunk_counter += 1

            if response.get("is_final"):
                print(f"TTS done: {chunk_counter - 1} chunks sent")
                await client_ws.send_json({"event": "audio_done"})
                return

        except Exception as e:
            print(f"TTS streaming error: {e}")
            break

    await client_ws.send_json({"event": "audio_done"})


async def close_minimax_connection(tts_ws):
    if tts_ws:
        try:
            await tts_ws.send(json.dumps({"event": "task_finish"}))
            await tts_ws.close()
        except Exception:
            pass


app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected!")

    api_key = os.getenv("MINIMAX_API_KEY")
    try:
        voice_id = globals.config.get("tts", "voice_id")
    except Exception:
        voice_id = ""
    tts_enabled = bool(api_key and voice_id)

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            action = msg.get("action")

            if action == "create_session":
                session_id = session_manager.create_session(
                    globals.config.get("llm", "model"),
                    globals.config.get("llm", "system_prompt")
                )
                await websocket.send_json({
                    "event": "session_created",
                    "session_id": session_id
                })
                print(f"Session created: {session_id}")

            elif action == "chat":
                session_id = msg.get("session_id")
                user_message = msg.get("message")

                session = session_manager.get_session(session_id)
                if not session:
                    await websocket.send_json({"event": "error", "message": "Session not found"})
                    continue

                # Run ollama chat in thread pool to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, session.chat, user_message)
                print(f"Assistant: {response}")

                await websocket.send_json({
                    "event": "text_response",
                    "content": response
                })

                if tts_enabled:
                    tts_ws = None
                    try:
                        tts_ws = await establish_minimax_connection(api_key)
                        if tts_ws and await start_tts_task(tts_ws, voice_id):
                            await stream_tts_to_client(tts_ws, response, websocket)
                        else:
                            print("TTS task start failed")
                            await websocket.send_json({"event": "audio_done"})
                    except Exception as e:
                        print(f"TTS error: {e}")
                        await websocket.send_json({"event": "audio_done"})
                    finally:
                        await close_minimax_connection(tts_ws)
                else:
                    await websocket.send_json({"event": "audio_done"})

            else:
                await websocket.send_json({"event": "error", "message": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Connection error: {e}")


if __name__ == "__main__":
    globals.config.read("config/config.ini")
    uvicorn.run(app, host=globals.config.get("host", "ip"), port=globals.config.getint("host", "port"))
