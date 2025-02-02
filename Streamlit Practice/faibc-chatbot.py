import streamlit as st
from streamlit_chat import message
from transformers import pipeline

from itertools import zip_longest


pipe = pipeline("text-generation", "HuggingFaceH4/zephyr-7b-beta")
chat = lambda messages: pipe(messages, max_new_tokens=120)[0]

def gen_response(query):
    messages = [
        {"role": "system",
         "content": """Your name is AI Chatbot. You are a technical expert for Artificial Intelligence (AI), here to guide and assist students with their AI-related questions and concerns. Please provide accurate and helpful information, and always maintain a polite and professional tone
        1. Greet the user harshly, ask user's name and ask how they are doing
        2. Provide informative responses
        3. Avoid discussing sensitive and offensive topics
        4. Do not answer if user's question is not related to AI
        5. Be patient and considerate
        6. If user expresses gratitude, then respond appropriately
        7. Do not generate long paragraphs, keep your responses concise

        Remember, your primary goal is to assist and educate the users
        """}
    ]

    for hum, ai in zip_longest(st.session_state['past'], st.session_state['generated']):
        if hum is not None:
            messages.append({"role":"user", "content":hum})
        if ai is not None:
            messages.append({"role":"assistant", "content":ai})
    
    messages.append({"role":"user", "content":query})
    return chat(messages)['generated_text'][-1]

def submit():
    st.session_state['prompt'] = st.session_state['prompt_input']
    st.session_state['prompt_input'] = ""


st.set_page_config(page_title="HTS Chatbot")
st.title("AI Chatbot")

st.text_input("YOU: ", key="prompt_input", on_change=submit)

for (key, default) in [('generated', []), ('past', []), ('prompt', '')]:
    if key not in st.session_state: st.session_state[key] = default


if st.session_state['prompt'] != '':
    query = st.session_state['prompt']
    st.session_state['past'].append(query)
    
    resp = gen_response(query)
    st.session_state['generated'].append(resp)

if st.session_state['generated']:
    generated = st.session_state['generated']
    past = st.session_state['past']
    for i in range(len(generated)-1, -1, -1):
        message(generated[i], key=str(i))
        message(past[i], is_user=True, key=f"{i}_user")
