# https://platform.minimax.io/docs/api-reference/speech-t2a-http

import requests
import os

class MinimaxTTS:
    def __init__(self):
        self.api_key = os.getenv("MINIMAX_API_KEY")
        self.url = "https://api.minimax.io/v1/t2a_v2"
        self.voice_id = "moss_audio_0251081c-f530-11f0-8583-3ae0c9a1b09a"

    def create_voice_clone_prompt(self, ref_audio, ref_text):
        pass

    def generate_voice_clone(self, text, output_path, ref_audio=None, ref_text=None, voice_clone_prompt=None):
        payload = {
            "text": text,
            "model": "speech-2.8-hd",
            "voice_setting": {
                "voice_id": self.voice_id
            },
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "wav",
                "channel": 1
            }
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(self.url, headers=headers, json=payload)

        if response.status_code == 200:
            with open(f"{output_path}", "wb") as f:
                audio_bytes = bytes.fromhex(response.json()['data']['audio'])
                f.write(audio_bytes)
            print(f"Audio saved as {output_path}")
