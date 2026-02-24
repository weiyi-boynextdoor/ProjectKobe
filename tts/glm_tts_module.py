# https://docs.bigmodel.cn/cn/guide/models/sound-and-video/glm-tts#python

from zai import ZhipuAiClient
import os

class GlmTTS:
    def __init__(self):
        api_key = os.getenv("ZHIPUAI_API_KEY", "")
        self.client = ZhipuAiClient(api_key=api_key)

    def generate_voice_clone(self, text, output_path):
        response = self.client.audio.speech(
            model="glm-tts",
            input=text,
            voice="female",
            response_format="wav",
            speed=1.0,
            volume=1.0
        )
        response.stream_to_file(output_path)
        print(f"generate_voice_clone file save to {output_path}")
