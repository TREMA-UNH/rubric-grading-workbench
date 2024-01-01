import torch
from transformers import(
  AutoModelForQuestionAnswering,
  AutoTokenizer,
  pipeline
)
model_name = "sjrhuschlee/flan-t5-large-squad2"

# a) Using pipelines
nlp = pipeline(
  'question-answering',
  model=model_name,
  tokenizer=model_name,
  # trust_remote_code=True, # Do not use if version transformers>=4.31.0
)
qa_input = {
'question': f'{nlp.tokenizer.cls_token}Where do I live?',  # '<cls>Where do I live?'
'context': 'My name is Sarah and I live in London'
}
res = nlp(qa_input)
# {'score': 0.984, 'start': 30, 'end': 37, 'answer': ' London'}

# b) Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(
  model_name,
  # trust_remote_code=True # Do not use if version transformers>=4.31.0
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

question = f'{tokenizer.cls_token}Where do I live?'  # '<cls>Where do I live?'
context = 'My name is Sarah and I live in London'

print(f'question: {question}, context: {context}')
encoding = tokenizer(question, context, return_tensors="pt")
output = model(
  encoding["input_ids"],
  attention_mask=encoding["attention_mask"]
)

all_tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())
answer_tokens = all_tokens[torch.argmax(output["start_logits"]):torch.argmax(output["end_logits"]) + 1]
answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))

print(f'answer={answer}')
# 'London'
