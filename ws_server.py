import os
import json
import ssl
import uuid
import asyncio
import subprocess
import threading
import logging
import websockets
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
import uvicorn
from dotenv import load_dotenv
import globals
from llm.llm_session import LLMSession, create_llm_session

load_dotenv()

TTS_MODEL = "speech-2.8-hd"
MINIMAX_TTS_FILE_FORMAT = "mp3"
LOG_DIR = "log"
LOG_FILE = os.path.join(LOG_DIR, "ws_server.log")

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("ws_server")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False

class SessionManager:
    def __init__(self):
        self.sessions: dict[int, LLMSession] = {}

    def get_session(self, websocket_id: int) -> LLMSession:
        return self.sessions.get(websocket_id)

    def create_session(self, websocket_id, model_name, system_prompt) -> str:
        self.sessions[websocket_id] = create_llm_session(globals.config.get("llm", "type"), model_name, system_prompt)


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
        logger.info("Minimax TTS connected")
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
            logger.warning("ffmpeg not found, falling back to raw mp3")
        except Exception:
            logger.exception("ffmpeg error, falling back to raw mp3")

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
                    print("Converting audio chunk #%s", chunk_counter)
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
                logger.info("TTS done: %s chunks received", chunk_counter - 1)
                break

    except Exception:
        logger.exception("TTS streaming error")
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


@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    websocket_id = id(websocket)
    client_host = websocket.client.host if websocket.client else "unknown"
    client_port = websocket.client.port if websocket.client else "unknown"
    logger.info("Client connected: websocket_id=%s client=%s:%s", websocket_id, client_host, client_port)

    # create session immediately
    session_manager.create_session(
        websocket_id,
        globals.config.get("llm", "model"),
        globals.config.get("llm", "system_prompt")
    )
    await websocket.send_json({
        "event": "session_created"
    })

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

            if action == "chat":
                user_message = msg.get("message")
                logger.info("User input: websocket_id=%s message=%s", websocket_id, user_message)

                session = session_manager.get_session(websocket_id)
                if not session:
                    await websocket.send_json({"event": "error", "message": "Session not found"})
                    continue

                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, session.chat, user_message)
                logger.info("LLM reply: websocket_id=%s response=%s", websocket_id, response)

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
                            logger.warning("TTS task start failed")
                            await websocket.send_json({"event": "audio_done"})
                    except Exception:
                        logger.exception("TTS error")
                        await websocket.send_json({"event": "audio_done"})
                    finally:
                        await close_minimax_connection(tts_ws)
                else:
                    await websocket.send_json({"event": "audio_done"})

            else:
                await websocket.send_json({"event": "error", "message": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        logger.info("Client disconnected: websocket_id=%s client=%s:%s", websocket_id, client_host, client_port)
    except Exception:
        logger.exception("Connection error: websocket_id=%s client=%s:%s", websocket_id, client_host, client_port)


if __name__ == "__main__":
    globals.config.read("config/config.ini")
    uvicorn.run(app, host=globals.config.get("host", "ip"), port=globals.config.getint("host", "port"))
