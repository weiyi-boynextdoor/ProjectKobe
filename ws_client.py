import asyncio
import json
import subprocess
import websockets

WS_URL = "ws://127.0.0.1:8024/ws"


class StreamAudioPlayer:
    def __init__(self):
        self.mpv_process = None

    def start(self):
        try:
            self.mpv_process = subprocess.Popen(
                ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except FileNotFoundError:
            print("[Error] mpv not found, please install mpv")
            return False

    def feed(self, hex_audio: str):
        if self.mpv_process and self.mpv_process.stdin:
            try:
                self.mpv_process.stdin.write(bytes.fromhex(hex_audio))
                self.mpv_process.stdin.flush()
            except Exception as e:
                print(f"[Audio feed error] {e}")

    def finish(self):
        """Close stdin so mpv knows the stream ended, then wait for playback."""
        if self.mpv_process:
            try:
                if self.mpv_process.stdin:
                    self.mpv_process.stdin.close()
                self.mpv_process.wait()
            except Exception as e:
                print(f"[Audio finish error] {e}")
            self.mpv_process = None

    def stop(self):
        if self.mpv_process:
            try:
                if self.mpv_process.stdin:
                    self.mpv_process.stdin.close()
                self.mpv_process.terminate()
            except Exception:
                pass
            self.mpv_process = None


async def main():
    print(f"Connecting to {WS_URL} ...")
    try:
        async with websockets.connect(WS_URL) as ws:
            await ws.send(json.dumps({"action": "create_session"}))
            response = json.loads(await ws.recv())

            if response.get("event") != "session_created":
                print(f"Failed to create session: {response}")
                return

            session_id = response["session_id"]
            print(f"Session created (id={session_id}). Type your message (Ctrl+C to quit).\n")

            loop = asyncio.get_event_loop()

            while True:
                try:
                    user_input = await loop.run_in_executor(None, input, "User: ")
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                if not user_input.strip():
                    continue

                await ws.send(json.dumps({
                    "action": "chat",
                    "session_id": session_id,
                    "message": user_input
                }))

                player = StreamAudioPlayer()
                player_started = False

                while True:
                    msg = json.loads(await ws.recv())
                    event = msg.get("event")

                    if event == "text_response":
                        print(f"Assistant: {msg['content']}\n")

                    elif event == "audio_chunk":
                        audio_hex = msg.get("data", "")
                        if audio_hex:
                            if not player_started:
                                player_started = player.start()
                            if player_started:
                                player.feed(audio_hex)

                    elif event == "audio_done":
                        if player_started:
                            # finish() blocks until mpv is done playing
                            await loop.run_in_executor(None, player.finish)
                        break

                    elif event == "error":
                        print(f"[Error] {msg.get('message')}")
                        player.stop()
                        break

    except (websockets.exceptions.ConnectionRefusedError, OSError):
        print(f"Cannot connect to server at {WS_URL}")


if __name__ == "__main__":
    asyncio.run(main())
