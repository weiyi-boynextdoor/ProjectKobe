import os
import json
import ssl
import uuid
import asyncio
import subprocess
import threading
import websockets
import ollama
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
import uvicorn
from dotenv import load_dotenv
import globals

load_dotenv()

TTS_MODEL = "speech-2.8-hd"
MINIMAX_TTS_FILE_FORMAT = "mp3"


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
        self.sessions: dict[str, OllamaSession] = {}

    def get_session(self, session_id) -> OllamaSession:
        return self.sessions.get(session_id)

    def create_session(self, model_name, system_prompt) -> str:
        session_id = str(uuid.uuid4())
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
            "format": MINIMAX_TTS_FILE_FORMAT,
            "channel": 1
        }
    }
    await tts_ws.send(json.dumps(start_msg))
    response = json.loads(await tts_ws.recv())
    return response.get("event") == "task_started"


async def stream_tts_to_client(tts_ws, text, client_ws: WebSocket):
    """Send text to Minimax, convert MP3→WAV via ffmpeg, forward WAV chunks to client"""
    await tts_ws.send(json.dumps({
        "event": "task_continue",
        "text": text
    }))

    # Start ffmpeg: stdin=mp3 stream, stdout=wav stream
    ffmpeg_proc = None
    target_file_format = globals.config.get("tts", "file_format", fallback=MINIMAX_TTS_FILE_FORMAT).lower()
    if target_file_format != MINIMAX_TTS_FILE_FORMAT:
        try:
            ffmpeg_proc = subprocess.Popen(
                ["ffmpeg", "-f", MINIMAX_TTS_FILE_FORMAT, "-i", "pipe:0",
                "-f", target_file_format, "-ar", "32000", "-ac", "1", "pipe:1"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print("ffmpeg not found, falling back to raw mp3")
        except Exception as e:
            print(f"ffmpeg error: {e}, falling back to raw mp3")

    loop = asyncio.get_event_loop()
    wav_queue: asyncio.Queue = asyncio.Queue()

    def read_wav_chunks():
        """Read WAV output from ffmpeg stdout in a background thread."""
        while True:
            chunk = ffmpeg_proc.stdout.read(4096)
            if not chunk:
                break
            loop.call_soon_threadsafe(wav_queue.put_nowait, chunk)
        loop.call_soon_threadsafe(wav_queue.put_nowait, None)  # sentinel

    if ffmpeg_proc:
        reader_thread = threading.Thread(target=read_wav_chunks, daemon=True)
        reader_thread.start()

    async def forward_wav():
        """Forward WAV chunks from queue to client WebSocket."""
        while True:
            chunk = await wav_queue.get()
            if chunk is None:
                break
            await client_ws.send_json({
                "event": "audio_chunk",
                "data": chunk.hex(),
            })
        await client_ws.send_json({"event": "audio_done"})

    await client_ws.send_json({"event": "audio_start", "format": target_file_format, "sample_rate": 32000, "channel": 1, "bitrate": 128000})
    
    if ffmpeg_proc:
        forward_task = asyncio.create_task(forward_wav())

    chunk_counter = 1
    try:
        while True:
            response = json.loads(await tts_ws.recv())

            if "data" in response and "audio" in response["data"]:
                audio_hex = response["data"]["audio"]
                if audio_hex:
                    print(f"Converting chunk #{chunk_counter}")
                    if ffmpeg_proc:
                        ffmpeg_proc.stdin.write(bytes.fromhex(audio_hex))
                        ffmpeg_proc.stdin.flush()
                    else:
                        await client_ws.send_json({
                            "event": "audio_chunk",
                            "data": audio_hex,
                            "format": MINIMAX_TTS_FILE_FORMAT
                        })
                    chunk_counter += 1

            if response.get("is_final"):
                print(f"TTS done: {chunk_counter - 1} chunks received")
                break

    except Exception as e:
        print(f"TTS streaming error: {e}")
    finally:
        if ffmpeg_proc:
            try:
                ffmpeg_proc.stdin.close()
            except Exception:
                pass

    if ffmpeg_proc:
        await forward_task
    else:
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

    tts_enabled = False
    if globals.config.get("tts", "type", fallback="none").lower() != "none":
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
