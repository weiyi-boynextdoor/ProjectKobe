from typing import Any, Dict, List, Optional, Tuple, Union
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
import soundfile as sf
import torch
import numpy as np

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
        self.tts_prompt : List[VoiceClonePromptItem] = []

    def create_voice_clone_prompt(
        self,
        ref_audio: Union[AudioLike, List[AudioLike]],
        ref_text: Optional[Union[str, List[Optional[str]]]] = None
    ) -> List[VoiceClonePromptItem]:
        # Placeholder for TTS synthesis logic
        self.tts_prompt = self.tts_model.create_voice_clone_prompt(ref_audio, ref_text)
        return self.tts_prompt
    
    def generate_voice_clone(
        self,
        text: str,
        output_path: str,
        ref_audio: Optional[Union[AudioLike, List[AudioLike]]] = None,
        ref_text: Optional[Union[str, List[Optional[str]]]] = None,
        voice_clone_prompt: Optional[Union[Dict[str, Any], List[VoiceClonePromptItem]]] = None,
    ):
        if not voice_clone_prompt:
            voice_clone_prompt = self.tts_prompt
        wavs, sr = self.tts_model.generate_voice_clone(text, ref_audio=ref_audio, ref_text=ref_text, voice_clone_prompt=voice_clone_prompt)
        sf.write(output_path, wavs[0], sr)
        print(f"generate_voice_clone file save to {output_path}")
