from dotenv import load_dotenv
load_dotenv()

import os
os.system('chcp 65001')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
from transformers import pipeline, AutoProcessor, AutoModelForTextToSpectrogram, SpeechT5HifiGan
from datasets import load_dataset

import torch
import numpy as np
from scipy.io import wavfile

import google.generativeai as genai




# embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# speaker_embeddings = embeddings_dataset[7306]["xvector"]
# load custom Indian voice embedding
custom = np.load("speaker-embedding/arctic_a0001.npy")
speaker_embeddings = torch.tensor(custom).unsqueeze(0)

stt_pipe = pipeline("automatic-speech-recognition", model="Harveenchadha/vakyansh-wav2vec2-urdu-urm-60")
tts_processor = AutoProcessor.from_pretrained("TheUpperCaseGuy/Guy-Urdu-TTS")
tts_model = AutoModelForTextToSpectrogram.from_pretrained("TheUpperCaseGuy/Guy-Urdu-TTS")
tts_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
llm = genai.GenerativeModel('gemini-1.5-flash')

def text_to_speech(text):
    inputs = tts_processor(text=text, return_tensors="pt")
    with torch.no_grad():
        outs = tts_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=tts_vocoder)
    
    st.session_state['audio'] = outs.numpy()


st.set_page_config(page_title="URDU Chatbot")
st.title("URDU Chatbot")
st.session_state['audio'] = None


audio_input = st.audio_input("Say Something")
if audio_input or st.button("Test Input"):
    if audio_input:
        with open("tmp.wav", "wb") as f:
            f.write(audio_input.getvalue())
        audio = wavfile.read("./tmp.wav")
        if os.path.exists('./tmp.wav'):
            os.remove("./tmp.wav")
    else:
        audio = wavfile.read("./test-a7/test2.wav")

    audio = np.array(audio[1], dtype=np.float32)
    text = stt_pipe(audio)
    text = text['text'].replace('<s>', ' ')
    st.write(f"Transcription: {text}")
    
    # for now, do this
    resp = llm.generate_content(f"""This is a combination of Urdu letters.
                                After combining them, what does this word mean?
                                Make sure your response is in Urdu.
                                {text}""")
    st.write(f"Model's Response: {resp.text}")
    
    text_to_speech(resp.text)

if st.session_state['audio'] is not None:
    st.header("Response")
    st.audio(st.session_state['audio'], sample_rate=16000)  # sample rate from SpeechT5
