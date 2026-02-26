import asyncio
import json
import websockets
import io
import pygame

WS_URL = "ws://127.0.0.1:8024/ws"


def play_audio(chunks: list[bytes], fmt: str):
    """Play audio chunks using pygame, or save to file as fallback."""
    audio_data = b"".join(chunks)
    try:
        pygame.mixer.music.load(io.BytesIO(audio_data))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
    except Exception as e:
        print(f"Error playing audio: {e}")
        filename = f"response.{fmt}"
        with open(filename, "wb") as f:
            f.write(audio_data)
        print(f"[Audio saved to {filename}]")


async def main():
    print(f"Connecting to {WS_URL} ...")
    try:
        async with websockets.connect(WS_URL) as ws:
            # Create session
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

                audio_chunks: list[bytes] = []
                audio_format = "mp3"

                while True:
                    msg = json.loads(await ws.recv())
                    event = msg.get("event")

                    if event == "text_response":
                        print(f"Assistant: {msg['content']}\n")

                    elif event == "audio_chunk":
                        audio_hex = msg.get("data", "")
                        audio_format = msg.get("format", "mp3")
                        if audio_hex:
                            audio_chunks.append(bytes.fromhex(audio_hex))

                    elif event == "audio_done":
                        if audio_chunks:
                            play_audio(audio_chunks, audio_format)
                        break

                    elif event == "error":
                        print(f"[Error] {msg.get('message')}")
                        break

    except (websockets.exceptions.ConnectionRefusedError, OSError):
        print(f"Cannot connect to server at {WS_URL}")


if __name__ == "__main__":
    pygame.mixer.init()
    asyncio.run(main())
