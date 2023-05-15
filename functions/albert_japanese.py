from transformers import (
    AutoModelForMaskedLM, AutoTokenizer
)

tokenizer = AutoTokenizer.from_pretrained(
    "ken11/albert-base-japanese-v1-with-japanese-tokenizer")
model = AutoModelForMaskedLM.from_pretrained(
    "ken11/albert-base-japanese-v1-with-japanese-tokenizer")

text = "名前は[MASK]です。"
tokens = tokenizer(text, return_tensors="pt")
mask_index = tokens["input_ids"][0].tolist().index(tokenizer.mask_token_id)
predict = model(**tokens)[0]
_, result = predict[0, mask_index].topk(5)

print(tokenizer.convert_ids_to_tokens(result.tolist()))
