import pandas as pd
import numpy as np
import streamlit as st
from streamlit_chat import message
import whisper
import yt_dlp
from yt_dlp import Youtube
import openai
from openai.embeddings_utils import distances_from_embeddings
import os

# whisper
model = whisper.load_model("base")
output = ""
data = []
data_transcription = []

# ... (other global variables)

# Sidebar# ... (previous code remains the same)

# Sidebar
with st.sidebar:
    user_secret = st.text_input(
        label=":blue[OpenAI API key]",
        value="",
        placeholder="Paste your OpenAI API key, sk-",
        type="password",
        key="Gaurav",  # Unique key for this text input
    )
    youtube_link = st.text_input(
        label=":red[Youtube link]",
        value="https://youtu.be/rQeXGvFAJDQ",
        placeholder="",
        key="Gaurav",  # Unique key for this text input
    )
    if youtube_link and user_secret:
        try:
            youtube_video = Youtube(youtube_link)
            video_id = youtube_video.video_id

            # Check if 'streamingData' key is present
            if "streamingData" not in youtube_video.vid_info:
                st.error("No streaming data available for this video.")
            else:
                streams = youtube_video.streams.filter(only_audio=True)
                stream = streams.first()
                if st.button("Start Analysis"):
                    if os.path.exists("word_embeddings.csv"):
                        os.remove("word_embeddings.csv")

                    with st.spinner("Running process..."):
                        mp4_video = stream.download(filename="youtube_video.mp4")
                        audio_file = open(mp4_video, "rb")
                        st.write(youtube_video.title)
                        st.video(youtube_link)

                        # Placeholder for the actual transcription code
                        output = {"text": "Sample transcription", "segments": []}

                        transcription = {
                            "title": youtube_video.title.strip(),
                            "transcription": output["text"],
                        }
                        data_transcription.append(transcription)
                        pd.DataFrame(data_transcription).to_csv("transcription.csv")
                        segments = output["segments"]

                        # Placeholder for the actual embedding code
                        for segment in segments:
                            openai.api_key = user_secret
                            response = openai.Embedding.create(
                                input=segment["text"].strip(),
                                model="text-embedding-ada-002",
                            )
                            embeddings = response["data"][0]["embedding"]
                            meta = {
                                "text": segment["text"].strip(),
                                "start": segment["start"],
                                "end": segment["end"],
                                "embedding": embeddings,
                            }
                            data.append(meta)
                        pd.DataFrame(data).to_csv("word_embeddings.csv")
                        os.remove("youtube_video.mp4")
                        st.success("Analysis completed")
        except yt_dlp.utils.DownloadError as e:
            st.error(f"DownloadError: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# ... (the rest of your code remains the same)

# ... (the rest of your code remains the same)

st.markdown(
    "<h1>Youtube GPT 🤖<small> by <a href='https://codegpt.co'>Code GPT</a></small></h1>"
   , unsafe_allow_html=True,
)


# ... (the rest of your code remains the same)


st.write(
    "Start a chat with this video of Microsoft CEO Satya Nadella's interview. You just need to add your OpenAI API Key and paste it in the 'Chat with the video' tab."
)

DEFAULT_WIDTH = 80
VIDEO_DATA = "https://youtu.be/bsFXgfbj8Bc"

width = 40

width = max(width, 0.01)
side = max((100 - width) / 2, 0.01)

_, container, _ = st.columns([side, 47, side])
container.video(data=VIDEO_DATA)
tab2, tab3, tab4 = st.tabs(
    ["Transcription", "Embedding", "Chat with the Video"]
)

with tab2:
    st.header("Transcription:")
    if os.path.exists("youtube_video.mp4"):
        audio_file = open("youtube_video.mp4", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/ogg")
    if os.path.exists("transcription.csv"):
        df = pd.read_csv("transcription.csv")
        st.write(df)
with tab3:
    st.header("Embedding:")
    if os.path.exists("word_embeddings.csv"):
        df = pd.read_csv("word_embeddings.csv")
        st.write(df)
with tab4:
    user_secret = st.text_input(
        label=":blue[OpenAI API key]",
        placeholder="Paste your openAI API key, sk-",
        type="password",
    )
    st.write(
        "To obtain an API Key you must create an OpenAI account at the following link: https://openai.com/api/"
    )
    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    def get_text():
        if user_secret:
            st.header("Ask me something about the video:")
            input_text = st.text_input("You: ", "", key="input")
            return input_text

    user_input = get_text()

    def get_embedding_text(api_key, prompt):
        openai.api_key = user_secret
        response = openai.Embedding.create(
            input=prompt.strip(), model="text-embedding-ada-002"
        )
        q_embedding = response["data"][0]["embedding"]
        df = pd.read_csv("word_embeddings.csv", index_col=0)
        df["embedding"] = df["embedding"].apply(eval).apply(np.array)

        df["distances"] = distances_from_embeddings(
            q_embedding, df["embedding"].values, distance_metric="cosine"
        )
        returns = []

        # Sort by distance with 2 hints
        for i, row in df.sort_values("distances", ascending=True).head(4).iterrows():
            # Else add it to the text that is being returned
            returns.append(row["text"])

        # Return the context
        return "\n\n###\n\n".join(returns)

    def generate_response(api_key, prompt):
        one_shot_prompt = (
            """I am YoutubeGPT, a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer.
        Q: What is human life expectancy in the United States?
        A: Human life expectancy in the United States is 78 years.
        Q: """
            + prompt
            + """
        A: """
        )
        completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt=one_shot_prompt,
            max_tokens=1024,
            n=1,
            stop=["Q:"],
            temperature=0.2,
        )
        message = completions.choices[0].text
        return message

    if user_input:
        text_embedding = get_embedding_text(user_secret, user_input)
        title = pd.read_csv("transcription.csv")["title"]
        string_title = "\n\n###\n\n".join(title)
        user_input_embedding = (
            'Using this context: "'
            + string_title
            + ". "
            + text_embedding
            + '", answer the following question. \n'
            + user_input
        )
        # st.write(user_input_embedding)
        output = generate_response(user_secret, user_input_embedding)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
