import streamlit as st
from transformers import pipeline

# Initialize the chatbot model
st.set_page_config(page_title="Basic Chatbot", page_icon="🤖")

@st.cache_resource
def load_text_generator():
    text_generator = pipeline("text-generation", model="gpt2")
    text_generator.tokenizer.pad_token = text_generator.tokenizer.eos_token
    return text_generator


SYSTEM_PROMPT = (
    "You are a helpful AI assistant for Software Engineers. "
    "Answer the questions as accurately as possible. "
    "Keep your answers concise and to the point. "
    "Answer in markdown format."
)

#build conversation prompt
def build_conversation_prompt(chat_history, user_question):
    formatted_conversation = []
    for previous_question, previous_answer in chat_history:
        formatted_conversation.append(f"User: {previous_question}\nAI: {previous_answer}\n")

    formatted_conversation.append(f"User: {user_question}\nAI:")
    return SYSTEM_PROMPT + "\n" + "\n".join(formatted_conversation)

st.title("🤖 Basic Chatbot for Software Engineers")
st.caption("Ask me anything related to software engineering!")

# ensure chat history exists
st.session_state.setdefault('chat_history', [])

#sidebar for config
with st.sidebar:
     st.header("Model controls/config")
     max_new_tokens = st.slider("Max new tokens", min_value=50, max_value=1000, value=200, step=50)
     temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

     if st.button("Clear chat"):
         st.session_state['chat_history'] = []
         st.success("Chat history cleared!")

#Display chat history
for user_message, bot_response in st.session_state.get('chat_history', []):
    st.markdown(f"**User:** {user_message}")
    st.markdown(f"**AI:** {bot_response}")
     

#User input
user_input = st.chat_input("Nass replyy...")
if user_input:
    st.chat_message("user").markdown(user_input)

    with st.spinner('thinking...'):
        try:
            text_generator = load_text_generator()
            prompt = build_conversation_prompt(st.session_state.get('chat_history', []), user_input)
            
            generation_output = text_generator(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=text_generator.tokenizer.eos_token_id,
                eos_token_id=text_generator.tokenizer.eos_token_id,
                do_sample=True,
                num_return_sequences=1,
            )

            # Extracting Model answer from generation text
            generated_answer = ""
            if generation_output and isinstance(generation_output, list):
                generated_text = generation_output[0].get('generated_text', '')
                if "AI:" in generated_text:
                    generated_answer = generated_text.split("AI:")[-1].strip()
                else:
                    # fallback: attempt to remove the prompt part
                    generated_answer = generated_text.replace(prompt, "").strip()
            if not generated_answer:
                generated_answer = "Sorry, I couldn't generate a response."

        except Exception as e:
            st.error(f"Generation error: {e}")
            generated_answer = "Sorry, an error occurred while generating the response."


    # Displaying and storing the generated answer
    st.chat_message("assistant").markdown(generated_answer)    
    st.session_state['chat_history'].append((user_input, generated_answer))
