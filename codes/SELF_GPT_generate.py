import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

def get_model_response(message):
    response = client.chat.completions.create(
        # model="gpt-4-turbo",
        # model="gpt-3.5-turbo-16k-0613",
        model="gpt-4",
        messages=message,
        temperature=0.8)
    return response.choices[0].message.content

# response generation function
def clf_response(model, tokenizer,question, answer):
    # make category dict
    category_dict = {0: 'yes', 1: 'unknown', 2: 'no'}
    input_dict = tokenizer(answer, question, padding='max_length', truncation=True, return_tensors='pt')
    # set device to cuda
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    # make response from finetuned bert
    batch = {k: v.to(device) for k,v in input_dict.items()}
    
    with torch.no_grad():
        outputs = model(**batch)
    
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1)
    
    return category_dict[preds[0].cpu().item()]

def get_dataset(file_path):
    answer=[]
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(tqdm(file)):
            json_data = json.loads(line)
            answer.append(json_data['subject'])
    return answer

def init_prompt():
    prompt = """You are the questioner.\n
            The questioner should carefully select questions to narrow down the possibilities based on the answerer's responses.\n
            You must ask questions that can be answered with "yes," "no," or "unknown." \n
            The questioner has up to 20 questions to guess what the answerer is thinking of.\n\n

            Example : \n
            questioner : Is it alive?\n
            answer : No.\n
            questioner : Is it related to technology?\n
            answer : Yes.\n
            questioner : Can it be carried by hand?\n
            answer : Yes.\n
            questioner : Is it primarily used for communication?\n
            answer : Yes.\n
            questioner : Is it a smartphone?\n
            answer : Yes, that is correct!\n\n

            Generate a question and infer the correct answer to get correct answers like Example\n
            questioner :
            """ 
    return [{"role": "user", "content": f"{prompt}"}]
def refine_prompt(prompt,gpt_response, clf_answer):
    prompt.append({"role": "assistant", "content": f"{gpt_response}"})
    tmp_prompt = clf_answer + "\n questioner :"
    prompt.append({"role": "user", "content": f"{tmp_prompt}"})
    return prompt

def generation(answer_list):
    model_path = './results/model-2-roberta-large'
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=3)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    dialogue_history=[]
    avg=0.0
    for ref_answer in tqdm(answer_list[:1000]):
        cnt=0
        question_history=[]
        answer_history=[]
        flag=False
        prompt = init_prompt()
        while cnt<20:
            question = get_model_response(prompt)
            question_history.append(question)
            if ref_answer.lower() in question.lower() :
                flag=True
                avg+=1
                break
            else:
                clf_answer = clf_response(model, tokenizer,question,ref_answer)
                answer_history.append(clf_answer)
                prompt = refine_prompt(prompt, question, clf_answer)
            cnt+=1
        dialogue_history.append({'question_log':question_history, 'answer_log':answer_history,'answer':ref_answer, 'find':flag})
    # import pdb;pdb.set_trace();
    print("Accuracy : ", avg/1000)
    with open("./results/gpt-4_1000.json", 'w', encoding='utf-8') as file:
        json.dump(dialogue_history, file, ensure_ascii=False, indent=4)
           

if __name__=="__main__":
    dataset_path = './results/self_data/twentyquestions-test-preprocess-2.jsonl'
    answer_list = get_dataset(dataset_path)
    print("### 20hills Answer Data", len(answer_list), " ###")
    generation(answer_list)
    