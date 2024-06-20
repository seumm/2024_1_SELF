import os
import json
from random import randrange
from functools import partial
import torch
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          LlamaForCausalLM,
                          BitsAndBytesConfig,
                          HfArgumentParser,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForLanguageModeling,
                          EarlyStoppingCallback,
                          pipeline,
                          logging,
                          set_seed)
from transformers import pipeline, Conversation
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, AutoPeftModelForCausalLM
from trl import SFTTrainer
from datamodule import LlamaDataset
from tqdm import tqdm

def create_bnb_config(load_in_4bit, bnb_4bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype):
    """
    Configures model quantization method using bitsandbytes to speed up training and inference

    :param load_in_4bit: Load model in 4-bit precision mode
    :param bnb_4bit_use_double_quant: Nested quantization for 4-bit model
    :param bnb_4bit_quant_type: Quantization data type for 4-bit model
    :param bnb_4bit_compute_dtype: Computation data type for 4-bit model
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = load_in_4bit,
        bnb_4bit_use_double_quant = bnb_4bit_use_double_quant,
        bnb_4bit_quant_type = bnb_4bit_quant_type,
        bnb_4bit_compute_dtype = bnb_4bit_compute_dtype,
    )

    return bnb_config
def load_model(model_name, bnb_config):
    """
    Loads model and model tokenizer

    :param model_name: Hugging Face model name
    :param bnb_config: Bitsandbytes configuration
    """

    # Get number of GPU device and set maximum memory
    n_gpus = torch.cuda.device_count()
    max_memory = f'{40960}MB'
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map = "auto", # dispatch the model efficiently on the available resources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )

    # Load model tokenizer with the user authentication token
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token = True)

    # Set padding token as EOS token
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
################################################################################
# transformers parameters
################################################################################

# The pre-trained model from the Hugging Face Hub to load and fine-tune

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
load_in_4bit = True

# Activate nested quantization for 4-bit base models (double quantization)
bnb_4bit_use_double_quant = True

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Compute data type for 4-bit base models
bnb_4bit_compute_dtype = torch.bfloat16
# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

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
    return prompt
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

def test(model, tokenizer, dataloader):
    clf_model_path = './results/model-2-roberta-large'
    clf_model = RobertaForSequenceClassification.from_pretrained(clf_model_path, num_labels=3)
    clf_tokenizer = RobertaTokenizer.from_pretrained(clf_model_path)
    pipe = pipeline(task='conversational', model=model, tokenizer=tokenizer)
    avg=0
    dialogue_history=[]
    for ref_answer in tqdm(dataloader[:100]):
        conv = Conversation(init_prompt())
        question_history=[]
        answer_history=[]
        flag=False
        cnt=0
        while cnt<20:
            question = pipe(conv)[-1]['content']
            if question.find("question:\n\n") != -1:
                question = question[question.find("question:\n\n")+len("question:\n\n"):]
                conv.messages[-1]["content"]=question
            elif question.find("question: ") != -1:
                question = question[question.find("question: ")+len("question: "):]
                conv.messages[-1]["content"]=question
            question_history.append(question)
            if ref_answer.lower() in question.lower() :
                flag=True
                avg+=1
                break
            else:
                clf_answer = clf_response(clf_model, clf_tokenizer,question,ref_answer)
                answer_history.append(clf_answer)
                cont_prompt = clf_answer + "\n questioner :"
            conv.add_message({"role":"user","content":cont_prompt})
            cnt+=1
        dialogue_history.append({'question_log':question_history, 'answer_log':answer_history,'answer':ref_answer, 'find':flag})
        
    print(avg/100)
    with open('my_list.json', 'w') as f:
        json.dump(dialogue_history, f)
def get_dataset(file_path):
    answer=[]
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(tqdm(file)):
            json_data = json.loads(line)
            answer.append(json_data['subject'])
    return answer

if __name__ == "__main__":
    dataset_path = './results/self_data/twentyquestions-test-preprocess-2.jsonl'
    answer_list = get_dataset(dataset_path)
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    # model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map = "auto", torch_dtype = torch.bfloat16)
    # Load model from Hugging Face Hub with model name and bitsandbytes configuration

    bnb_config = create_bnb_config(load_in_4bit, bnb_4bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype)
    model, tokenizer = load_model(model_name, bnb_config)
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token = True)
    tokenizer.pad_token = tokenizer.eos_token
    # datamodule = LlamaDataset(tokenizer, 4086)
    # test_loader = datamodule.test_dataloader()
    test(model,tokenizer,answer_list)
    