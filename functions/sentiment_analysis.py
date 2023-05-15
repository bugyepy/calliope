from transformers import pipeline

classifier_pipeline = pipeline(
    'sentiment-analysis',
    model='jarvisx17/japanese-sentiment-analysis',
    tokenizer='jarvisx17/japanese-sentiment-analysis'
)

def sentiment_analyzer(
    input_text
):
    output_text = classifier_pipeline(input_text)[0]

    print(f'{input_text} :判定 {output_text["label"]} , score: {output_text["score"]}')
    return True if output_text["label"] == "positive" else False