from transformers import VitsModel, AutoTokenizer
import torch
import scipy
import numpy as np


# must be "script_arabic"
model = VitsModel.from_pretrained("facebook/mms-tts-urd-script_arabic")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-urd-script_arabic")

text = "آپ کیسے ہیں"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

# fix for "torch.dtype has no attribute kind" error
output = output.numpy()
output = np.squeeze(output)

scipy.io.wavfile.write("test-tts.wav", rate=model.config.sampling_rate, data=output)
