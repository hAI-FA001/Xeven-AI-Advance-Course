import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from transformers import AutoProcessor, AutoModelForTextToSpectrogram, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf


import numpy as np

custom = np.load(r"C:\Users\PMLS\Desktop\code\py\Free AI Advance Course\Streamlit Practice\custom-speaker-embeddings\output\cmu_us_awb_arctic-0.90-release\cmu_us_awb_arctic\wav\arctic_a0001.npy")



# embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# speaker_embeddings = embeddings_dataset[7306]["xvector"]
# speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)
speaker_embeddings = torch.tensor(custom).unsqueeze(0)

processor = AutoProcessor.from_pretrained("TheUpperCaseGuy/Guy-Urdu-TTS")
model = AutoModelForTextToSpectrogram.from_pretrained("TheUpperCaseGuy/Guy-Urdu-TTS")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")


text = "Testing this thing"
inputs = processor(text=text, return_tensors="pt")
with torch.no_grad():
    outs = model.generate_speech(inputs['input_ids'], speaker_embeddings, vocoder=vocoder)
sf.write("tts.wav", outs.numpy(), samplerate=16000)
