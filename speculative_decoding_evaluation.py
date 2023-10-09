from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from tqdm import tqdm
from rich import pretty,print
pretty.install()

def load_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer

def load_models(model_id,assistant_checkpoint,device,peft_model_id=None):
    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True)
    # comment out AttributeError: 'LlamaForCausalLM' object has no attribute 'load_adapter'
    if peft_model_id:
        model.load_adapter(peft_model_id)
    print("Large model loaded")

    model.config.use_cache = True
    assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint).half().to(device)  
    assistant_model.config.use_cache = True
    print("Small model loaded")
    return model, assistant_model

def inference_comparison(prompt,tokenizer,model,assistant_model,device):
    print("---"*50)    
    formatted_prompt = f"### Human: {prompt}### Assistant:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)    
    print("### Large-Model Native Decoding Starts...\n")
    start = time.time()
    outputs = model.generate(**inputs, assistant_model=None, max_new_tokens=512)
    end = time.time()
    result1=tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(result1[0])
    result1_time = end - start
    print("Time took: ", result1_time)

    print("### Tiny Assisted Model Decoding Starts...\n")
    start = time.time()
    outputs = model.generate(**inputs, assistant_model=assistant_model,max_new_tokens=512)
    end = time.time()
    result2=tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(result2[0])
    # print time in seconds
    result2_time = end - start    
    print("Time took: ", result2_time)
    return result1_time, result2_time

def assisted_inference(prompt,tokenizer,model,assistant_model,device):
    print("---"*50)    
    formatted_prompt = f"### Human: {prompt}### Assistant:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)    
    print("### Tiny Assisted Model Decoding Starts...\n")
    start = time.time()
    outputs = model.generate(**inputs, assistant_model=assistant_model,max_new_tokens=512)
    end = time.time()
    result2=tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(result2[0])
    # print time in seconds
    result2_time = end - start    
    print("Time took: ", result2_time)
    return result2_time

def run_tests():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "meta-llama/Llama-2-13b-chat-hf"
    #model_id = "huggyllama/llama-13b"
    #peft_model_id = "timdettmers/guanaco-13b"
    assistant_checkpoint = "PY007/TinyLlama-1.1B-Chat-v0.1"

    tokenizer=load_tokenizer(model_id)
    model, assistant_model = load_models(model_id,assistant_checkpoint,device)

    print("Running warmup...\n")
    warmup_prompt_list = [
        "Give me detailed info about Justin Trudeau.",         
        "Name planets in our solar system",       
        ]    
    ## warmup prompt
    for prompt in tqdm(warmup_prompt_list):
        inference_comparison(prompt,tokenizer,model,assistant_model,device)

    ## test prompts 
    comparison_result=[]
    native_result=[]    
    print("Running test prompts...\n")    
    test_prompt_list = [
        "Name planets in our solar system",        
        "Give me detailed info about Justin Trudeau.",                 
        "Generate a few good titles for a draft of a post [type of transformer models]",
        "Write a 5-line poem that describes a cat in a creative and original way. Use the following words in your poem: cat, fur, tail, independent, and clean.",
        "Given the text below, generate Q&A from the text provide: [rewrite of minGPT that prioritizes teeth over education.]",
        "Write a highly detailed discussion response, in the structure of an essay, responding to the following prompt: Explain the causes of the Inflation and whether expansion played a role in the economic recession. Include evidence to support your argument.",
        "Make a coding challenge for Python with 3 questions and answers"
        ]
    for prompt in tqdm(test_prompt_list):
        result1_time, result2_time = inference_comparison(prompt,tokenizer,model,assistant_model,device)
        comparison_result.append(str(result1_time)+","+str(result2_time))
        native_result.append(str(result1_time))

    print(50*"*****")    
    ## assisted inference test 
    assisted_result=[]
    print("Assisted Inference Only")
    for prompt in tqdm(test_prompt_list):
        result2_time = assisted_inference(prompt,tokenizer,model,assistant_model,device)  
        assisted_result.append(str(result2_time))  

    print(50*"===")
    print("native vs assisted comparison result:\n",comparison_result) 
    print("native only:\n",native_result)     
    print("assisted only:\n",assisted_result)     
    print(50*"===")    

if __name__ == '__main__':
    run_tests()
