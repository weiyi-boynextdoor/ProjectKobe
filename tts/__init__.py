
def get_tts_module(name):
    if name == "qwen3":
        from .qwen3_tts_module import Qwen3TTS
        return Qwen3TTS()
    elif name == "glm":
        from .glm_tts_module import GlmTTS
        return GlmTTS()
    elif name == "minimax":
        from .mninimax_tts_module import MinimaxTTS
        return MinimaxTTS()
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unsupported TTS module: {name}")
