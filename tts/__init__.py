
def get_tts_module(name):
    if name == "qwen3":
        from .qwen3_tts_module import Qwen3TTS
        return Qwen3TTS()
    elif name == "glm":
        from .glm_tts_module import client
        return client
    else:
        raise ValueError(f"Unsupported TTS module: {name}")
