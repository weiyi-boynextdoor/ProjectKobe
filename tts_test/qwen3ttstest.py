import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
)

ref_audio = "../audio_input/Mamba.wav"
ref_text  = "Man! ha ha ha ha ha ha ha. What can I say? Mamba out!"

wavs, sr = model.generate_voice_clone(
    text="Hey buddy you get in the wrong door, the leather club is 2 blocks down.",
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)
sf.write("../audio_output/output_voice_clone.wav", wavs[0], sr)