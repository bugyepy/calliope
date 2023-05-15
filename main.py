import streamlit
from functions import sentiment_analysis

streamlit.title("Calliope - LLM Sandbox")

input_text = streamlit.text_area("下記の文章をもとに感情を判定します。", height=100)

if streamlit.button("判定する"):
    streamlit.write(sentiment_analysis.sentiment_analyzer(input_text))