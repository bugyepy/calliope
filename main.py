import streamlit
from functions import sentiment_analysis, albert_japanese, open_calm

streamlit.title("Calliope - LLM Sandbox")

tab1, tab2, tab3 = streamlit.tabs(["感情分析", "ALBERT_TEST", "OpenCALM_TEST"])

with tab1:
    input_text = streamlit.text_area("下記の文章をもとに感情を判定します。", height=100, key=1)

    if streamlit.button("判定する", key=2):
        streamlit.write(sentiment_analysis.sentiment_analyzer(input_text))

with tab2:
    input_text = streamlit.text_area("下記の文章内の[MASK]という文字列が何か予測します。", height=100, key=3)
    if streamlit.button("判定する", key=4):
        if "[MASK]" in input_text:
            streamlit.write(albert_japanese.albert(input_text))
        else:
            streamlit.write("[MASK]を含んだ文章を入力してください。")

with tab3:
    input_text = streamlit.text_area("下記の文章に続く文章を予測します。", height=100, key=5)
    if streamlit.button("判定する", key=6):
        streamlit.write(open_calm.calm(input_text))
