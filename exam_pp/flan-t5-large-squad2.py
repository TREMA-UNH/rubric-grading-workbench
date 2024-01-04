import torch
from transformers import(
  AutoModelForQuestionAnswering,
  AutoTokenizer,
  pipeline
)
model_name = "sjrhuschlee/flan-t5-large-squad2"

device = None
BATCH_SIZE=1
MAX_TOKEN_LEN=512

print('\n\n Laura\'s approach')

# Laura's approach
# tokenizer = T5TokenizerFast.from_pretrained(self.modelName)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(
  model_name,
  # trust_remote_code=True # Do not use if version transformers>=4.31.0
)
print(f"T5 model config: { model.config}")
# self.promptGenerator = promptGenerator

# Create a Hugging Face pipeline
t5_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer, device=device, batch_size=BATCH_SIZE)

# def answer_multiple_choice_question(self, qpc:QuestionPromptWithChoices, promptGenerator:PromptGenerator)->str:
#     prompt = promptGenerator(qpc)

prompt1 = {
'question': f'{tokenizer.cls_token}Earth science is the study of',  # '<cls>Where do I live?'
'context': '''The geology of Mars is differentiable from that of Earth by, among other things, its extremely 
        large volcanoes and lack of crust movement. A goal of the MEP is to understand these differences 
        from Earth along with the way that wind, water, volcanoes, tectonics, cratering and other processes 
        have shaped the surface of Mars. Rocks can help scientists describe the sequence of events in Mars\' 
        history, tell whether there was an abundance of water on the planet through identifying minerals that
          are formed only in water, and tell if Mars once had a magnetic field (which would point toward Mars
            at one point being a dynamic Earth-like planet)
            '''
}
prompt2 = {
'question': f'{tokenizer.cls_token}What does earth science is the study?',  # '<cls>Where do I live?'
'context': '''The geology of Mars is differentiable from that of Earth by, among other things, its extremely 
        large volcanoes and lack of crust movement. A goal of the MEP is to understand these differences 
        from Earth along with the way that wind, water, volcanoes, tectonics, cratering and other processes 
        have shaped the surface of Mars. Rocks can help scientists describe the sequence of events in Mars\' 
        history, tell whether there was an abundance of water on the planet through identifying minerals that
          are formed only in water, and tell if Mars once had a magnetic field (which would point toward Mars
            at one point being a dynamic Earth-like planet)
            '''
}
prompt3 = {
'question': f'{tokenizer.cls_token}Where is Frankfurt located?',  # '<cls>Where do I live?'
'context': '''The geology of Mars is differentiable from that of Earth by, among other things, its extremely 
        large volcanoes and lack of crust movement. A goal of the MEP is to understand these differences 
        from Earth along with the way that wind, water, volcanoes, tectonics, cratering and other processes 
        have shaped the surface of Mars. Rocks can help scientists describe the sequence of events in Mars\' 
        history, tell whether there was an abundance of water on the planet through identifying minerals that
          are formed only in water, and tell if Mars once had a magnetic field (which would point toward Mars
            at one point being a dynamic Earth-like planet)
            '''
}
    # print("prompt",prompt)
outputs = t5_pipeline([prompt1,prompt2,prompt3], max_length=MAX_TOKEN_LEN, num_beams=5, early_stopping=True)
print(outputs)
# return outputs[0]['generated_text']
  # {'score': 0.984, 'start': 30, 'end': 37, 'answer': ' London'}





def a():
  print('\n\n a) using pipelines')

  # a) Using pipelines
  nlp = pipeline(
    'question-answering',
    model=model_name,
    tokenizer=model_name,
    device=None, batch_size=1, use_fast=True
    # trust_remote_code=True, # Do not use if version transformers>=4.31.0
  )
  print(nlp.model.config)
  qa_input = {
  'question': f'{nlp.tokenizer.cls_token}Where do I live?',  # '<cls>Where do I live?'
  'context': 'My name is Sarah and I live in London'
  }
  res = nlp(qa_input)
  # {'score': 0.984, 'start': 30, 'end': 37, 'answer': ' London'}


def b():
  print('\n\n b) load model and tokenizer')

  # b) Load model & tokenizer
  model = AutoModelForQuestionAnswering.from_pretrained(
    model_name,
    # trust_remote_code=True # Do not use if version transformers>=4.31.0
  )
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  print(model.config)

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


# a()

# b()