from typing import Union
from fastapi import FastAPI

from transformers import BertTokenizer, BertForQuestionAnswering
import torch

app = FastAPI()

tokenizer = BertTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")

@app.get("/")
def predict(question: str, text: str):

    inputs = tokenizer(question, text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    return tokenizer.decode(predict_answer_tokens)