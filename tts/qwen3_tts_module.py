from typing import Any, Dict, List, Optional, Tuple, Union
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
import soundfile as sf
import torch
import numpy as np
import globals

AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

class Qwen3TTS:
    def __init__(self):
        self.tts_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        ref_audio = globals.config.get("tts", "ref_audio")
        ref_text = globals.config.get("tts", "ref_text")
        self.tts_prompt = self.tts_model.create_voice_clone_prompt(ref_audio, ref_text)
    
    def generate_voice_clone(
        self,
        text: str,
        output_path: str,
    ):
        if not voice_clone_prompt:
            voice_clone_prompt = self.tts_prompt
        wavs, sr = self.tts_model.generate_voice_clone(text, voice_clone_prompt=self.tts_prompt)
        sf.write(output_path, wavs[0], sr)
        print(f"generate_voice_clone file save to {output_path}")
