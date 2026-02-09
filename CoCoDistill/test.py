import os
import torch
import json
import random
from pathlib import Path
from tqdm import tqdm
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from load_data import load_root_data

# Set random seed for reproducibility
random.seed(42)

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Define device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model
merged_model = AutoModelForCausalLM.from_pretrained(
    "/data/models/Phi-3.5-mini-instruct/", torch_dtype=torch.bfloat16, trust_remote_code=True
).to(device)

# Load LoRA weights and clean adapter configurations
lora_root = "./MentalManip_con_fine_tuning_res"

def clean_adapter_config(adapter_dir: str):
    """
    Clean up unnecessary fields in the adapter config file.
    """
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if not cfg_path.exists():
        print(f"[WARN] {cfg_path} does not exist. Skipping cleanup.")
        return
    
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    unnecessary_keys = ["corda_config", "qalora_group_size", "use_qalora", "eva_config", "loftq_config", 
                        "trainable_token_indices", "exclude_modules", "lora_bias"]
    
    for key in unnecessary_keys:
        if key in cfg:
            cfg.pop(key)
    
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Cleaned {cfg_path}")

# Clean adapter configurations for all epochs
for epoch in range(3):
    clean_adapter_config(os.path.join(lora_root, f"lora_weights_epoch_{epoch}"))

# Merge LoRA weights for the latest epoch
lora_model = PeftModel.from_pretrained(
    merged_model, os.path.join(lora_root, "lora_weights_epoch_2")
).to(device)

# Prepare tokenizer
merged_tokenizer = AutoTokenizer.from_pretrained("/data/models/Qwen2.5-3B-Instruct/", padding_side='left')
merged_model.eval()

# Functions for extracting responses
def extract_label_from_response(response):
    return 1 if "yes" in response else 0

def extract_rationale_from_response(response):
    return response.split("Rationale:")[1].strip()

def get_student_response(dataset, output_path):
    """
    Generate student responses and store them in a JSONLines file.
    """
    for data in tqdm(dataset):

        system_prompt = "I will provide you with a dialogue. Please determine if it contains elements of mental manipulation. Answer Yes or No and give a rationale."
        user_prompt = data["dialogue"]
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt = merged_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = merged_tokenizer([prompt], return_tensors="pt")
        model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}

        with torch.no_grad():
            generated_ids = merged_model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
                pad_token_id=merged_tokenizer.eos_token_id
            )
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)]
        response = merged_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        all_data = {"id": data["id"], "manipulative": data["manipulative"], "answer": response}

        with jsonlines.open(output_path, mode='a') as writer:
            writer.write(all_data)

def get_IAP_response(dataset, output_path):
    """
    Generate student IAP responses and store them in a JSONLines file.
    """
    system_prompt_1 = """
    I will provide you with a dialogue. Please summarize the intent of the statement made by Person1 in one sentence. \n"""
    system_prompt_2 = """
    I will provide you with a dialogue. Please summarize the intent of the statement made by Person2 in one sentence. \n"""
    system_prompt = """
    I will provide you with a dialogue and intent of person1, and intent of person2.
    Please carefully analyze the dialogue and intents, and determine if it contains elements of mental manipulation.
    Just answer with 'Yes' or 'No', and don't add anything else. \n
    """
    
    for data in tqdm(dataset):
        # Generate intent for Person1
        messages = [{"role": "system", "content": system_prompt_1}, {"role": "user", "content": data["dialogue"]}]
        text = merged_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = merged_tokenizer([text], return_tensors="pt").to("cuda")
        generated_ids = merged_model.generate(**model_inputs, max_new_tokens=512, do_sample=False, pad_token_id=merged_tokenizer.eos_token_id)
        intent_p1 = merged_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Generate intent for Person2
        messages = [{"role": "system", "content": system_prompt_2}, {"role": "user", "content": data["dialogue"]}]
        text = merged_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = merged_tokenizer([text], return_tensors="pt").to("cuda")
        generated_ids = merged_model.generate(**model_inputs, max_new_tokens=512, do_sample=False, pad_token_id=merged_tokenizer.eos_token_id)
        intent_p2 = merged_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Combine intents and generate final answer
        user_input = f'{data["dialogue"]} {intent_p1} {intent_p2}'
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
        text = merged_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = merged_tokenizer([text], return_tensors="pt").to("cuda")
        generated_ids = merged_model.generate(**model_inputs, max_new_tokens=1, do_sample=False, pad_token_id=merged_tokenizer.eos_token_id)
        answer = merged_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        new_data = {"id": data["id"], "manipulative": data["manipulative"], "intent1": intent_p1, "intent2": intent_p2, "answer": answer}

        with jsonlines.open(output_path, mode='a') as writer:
            writer.write(new_data)

def get_self_percept_response(dataset, output_path):
    """
    Reproduction of the SELF-PERCEPT method (two-stage prompting) for binary detection.
    """
    system_prompt_stage1 = """
    Stage 1 – Observation of Behavior
    Context: Provide a brief overview of the conversation or situation,
    including the participants and the main topic being discussed.
    Instructions: You are to observe and list the specific behaviors and
    statements made by the characters involved in the conversation.
    Pay attention to both verbal and non-verbal cues.
    Rules:
    - Note all actions and words that might indicate the characters’ attitudes or beliefs.
    - Focus on inconsistencies between what is said and the behavior displayed.
    - Identify any persuasive techniques or manipulations used in the conversation.
    Story/Conversation:
    {dialogue}
    Question: What behaviors and statements indicate the attitudes or beliefs of each character?
    List them clearly.
    """

    system_prompt_stage2 = """
    Stage 2 – Self-Inference
    Instructions: Based on the observed behaviors and statements from Stage 1 ({stage1_output}),
    answer the following question:
    Question: Based on the behaviors you observed,
    is there any manipulation detected in the conversation?
    Answer with 'Yes' or 'No' only.
    """

    for data in tqdm(dataset):
        # Stage 1 generation
        dialogue_text = system_prompt_stage1.format(dialogue=data["dialogue"])
        messages = [{"role": "system", "content": dialogue_text}]
        text = merged_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = merged_tokenizer([text], return_tensors="pt").to("cuda")
        generated_ids = merged_model.generate(**model_inputs, max_new_tokens=512, do_sample=False, pad_token_id=merged_tokenizer.eos_token_id)
        stage1_output = merged_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        # Stage 2 generation
        stage2_text = system_prompt_stage2.format(stage1_output=stage1_output)
        messages = [{"role": "system", "content": stage2_text}]
        text = merged_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = merged_tokenizer([text], return_tensors="pt").to("cuda")
        generated_ids = merged_model.generate(**model_inputs, max_new_tokens=1, do_sample=False, pad_token_id=merged_tokenizer.eos_token_id)
        answer = merged_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        new_data = {"id": data["id"], "manipulative": data["manipulative"], "stage1_output": stage1_output, "answer": answer}

        with jsonlines.open(output_path, mode='a') as writer:
            writer.write(new_data)

if __name__ == "__main__":
    input_path = "./dataset/MentalManip_con_test.json"
    output_path = "./results/student_test.json"

    with open(input_path, mode='r', encoding='utf-8') as json_file:
        data = [json.loads(f) for f in json_file]
    
    # Call the functions as needed
    get_student_response(data, output_path)
    # get_self_percept_response(data, output_path)
    # get_IAP_response(data, output_path)
