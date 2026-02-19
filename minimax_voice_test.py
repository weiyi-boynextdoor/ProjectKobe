# https://platform.minimax.io/docs/api-reference/speech-t2a-http

import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MINIMAX_API_KEY")
url = "https://api.minimax.io/v1/t2a_v2"

payload = {
    "text": "Man! What can I say? Mamba out!",
    "model": "speech-2.8-hd",
    "voice_setting": {
        "voice_id": "moss_audio_0251081c-f530-11f0-8583-3ae0c9a1b09a"
    },
    "audio_setting": {
        "sample_rate": 32000,
        "bitrate": 128000,
        "format": "wav",
        "channel": 1
    }
}

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    with open("audio_output/output.wav", "wb") as f:
        audio_bytes = bytes.fromhex(response.json()['data']['audio'])
        f.write(audio_bytes)
    print("Audio saved as output.wav")
