import asyncio
import websockets
import json
import ssl
import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

model = "speech-2.8-hd"
file_format = "mp3"

class StreamAudioPlayer:
    def __init__(self):
        self.mpv_process = None
        self.ffmpeg_process = None

    def start_processes(self):
        """同时启动 FFmpeg 转换器和 MPV 播放器"""
        try:
            # 1. 启动 FFmpeg: 输入 mp3 (pipe:0), 输出 wav (pipe:1)
            # -f mp3: 指定输入格式
            # -f wav: 指定输出格式
            # -ar 32000: 采样率与你的 API 设置保持一致
            ffmpeg_command = [
                "ffmpeg", "-i", "pipe:0", 
                "-f", "wav", "-ar", "32000", "-ac", "1", 
                "pipe:1"
            ]
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, # 捕获输出给 mpv
                stderr=subprocess.DEVNULL
            )

            # 2. 启动 MPV: 输入接收 FFmpeg 的输出
            mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
            self.mpv_process = subprocess.Popen(
                mpv_command,
                stdin=self.ffmpeg_process.stdout, # 直接对接 ffmpeg 的输出
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            
            print("FFmpeg 转换器与 MPV 播放器已就绪")
            return True
        except FileNotFoundError:
            print("错误: 未找到 ffmpeg 或 mpv，请确保已安装")
            return False

    def play_audio_chunk(self, hex_audio):
        """将 MP3 原始字节喂给 FFmpeg"""
        try:
            if self.ffmpeg_process and self.ffmpeg_process.stdin:
                audio_bytes = bytes.fromhex(hex_audio)
                # 写入 ffmpeg 的 stdin 进行转码
                self.ffmpeg_process.stdin.write(audio_bytes)
                self.ffmpeg_process.stdin.flush()
                return True
        except Exception as e:
            print(f"转码/播放失败: {e}")
            return False
        return False

    def stop(self):
        """停止所有进程"""
        for p in [self.ffmpeg_process, self.mpv_process]:
            if p:
                if p.stdin: p.stdin.close()
                p.terminate()

async def establish_connection(api_key):
    """Establish WebSocket connection"""
    url = "wss://api.minimax.io/ws/v1/t2a_v2"
    headers = {"Authorization": f"Bearer {api_key}"}

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    try:
        ws = await websockets.connect(url, additional_headers=headers, ssl=ssl_context)
        connected = json.loads(await ws.recv())
        if connected.get("event") == "connected_success":
            print("Connection successful")
            return ws
        return None
    except Exception as e:
        print(f"Connection failed: {e}")
        return None

async def start_task(websocket):
    """Send task start request"""
    start_msg = {
        "event": "task_start",
        "model": model,
        "voice_setting": {
            "voice_id": "moss_audio_0251081c-f530-11f0-8583-3ae0c9a1b09a",
            "speed": 1,
            "vol": 1,
            "pitch": 0,
            "english_normalization": False
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": file_format,
            "channel": 1
        }
    }
    await websocket.send(json.dumps(start_msg))
    response = json.loads(await websocket.recv())
    return response.get("event") == "task_started"

async def continue_task_with_stream_play(websocket, text, player):
    """Send continue request and stream play audio"""
    await websocket.send(json.dumps({
        "event": "task_continue",
        "text": text
    }))

    chunk_counter = 1
    total_audio_size = 0
    audio_data = b""

    while True:
        try:
            response = json.loads(await websocket.recv())

            if "data" in response and "audio" in response["data"]:
                audio = response["data"]["audio"]
                if audio:
                    print(f"Playing chunk #{chunk_counter}")
                    audio_bytes = bytes.fromhex(audio)
                    if player.play_audio_chunk(audio):
                        total_audio_size += len(audio_bytes)
                        audio_data += audio_bytes
                        chunk_counter += 1

            if response.get("is_final"):
                print(f"Audio synthesis completed: {chunk_counter-1} chunks")
                if player.mpv_process and player.mpv_process.stdin:
                    player.mpv_process.stdin.close()

                # Save audio to file
                with open(f"output.{file_format}", "wb") as f:
                    f.write(audio_data)
                print(f"Audio saved as output.{file_format}")

                estimated_duration = total_audio_size * 0.0625 / 1000
                wait_time = max(estimated_duration + 5, 10)
                return wait_time

        except Exception as e:
            print(f"Error: {e}")
            break

    return 10

async def close_connection(websocket):
    """Close connection"""
    if websocket:
        try:
            await websocket.send(json.dumps({"event": "task_finish"}))
            await websocket.close()
        except Exception:
            pass

async def main():
    API_KEY = os.getenv("MINIMAX_API_KEY")
    TEXT = """Radiance Field methods have recently revolutionized novel-view synthesis of scenes captured with multiple photos or videos. However, achieving high visual quality still requires neural networks that are costly to train and render, while recent faster methods inevitably trade off speed for quality. For unbounded and complete scenes (rather than isolated objects) and 1080p resolution rendering, no current method can achieve real-time display rates."""

    player = StreamAudioPlayer()

    try:
        if not player.start_processes():
            return

        ws = await establish_connection(API_KEY)
        if not ws:
            return

        if not await start_task(ws):
            print("Task startup failed")
            return

        wait_time = await continue_task_with_stream_play(ws, TEXT, player)
        await asyncio.sleep(wait_time)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        player.stop()
        if 'ws' in locals():
            await close_connection(ws)

if __name__ == "__main__":
    asyncio.run(main())