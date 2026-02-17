# https://docs.bigmodel.cn/cn/guide/models/sound-and-video/glm-tts#python

from zai import ZhipuAiClient

client = ZhipuAiClient(api_key="")
speech_file_path = ""
response = client.audio.speech(
    model="glm-tts",
    input="Man! What can I say?",
    voice="female",
    response_format="wav",
    speed=1.0,
    volume=1.0
)
response.stream_to_file(speech_file_path)