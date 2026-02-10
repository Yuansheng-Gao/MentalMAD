import os
import json
import random
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from bitsandbytes.optim import Lion
import peft

from trainer import train,validate
from load_data import preprocess_train_data, preprocess_valid_data


class CustomDataset(Dataset):
    def __init__(self, binary_input, binary_target, rationale_input, rationale_target, feedback_input, feedback_target, tokenizer, max_length):
        self.binary_input = binary_input
        self.binary_target = binary_target
        self.rationale_input = rationale_input
        self.rationale_target = rationale_target
        self.feedback_input = feedback_input
        self.feedback_target = feedback_target
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __len__(self):
        return len(self.rationale_target)

    def encode_sample(self, prompt, target, system_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False) + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            attention_mask = [1] * self.max_length
        else:
            pad_len = self.max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            attention_mask = [1] * (len(input_ids) - pad_len) + [0] * pad_len

        return (
            torch.tensor(input_ids),
            torch.tensor(attention_mask),
            torch.tensor(labels)
        )

    def __getitem__(self, idx):
        binary = self.encode_sample(
            self.binary_input[idx], self.binary_target[idx],
            "I will provide you with a dialogue. Please determine if it contains elements of mental manipulation. Just answer with 'Yes' or 'No', and don't add anything else.\n"
        ) # This item is only used for SFT, not for CoCoDistill.
        rationale = self.encode_sample(
            self.rationale_input[idx], self.rationale_target[idx],
            "I will provide you with a dialogue. Please determine if it contains elements of mental manipulation. Answer Yes or No and give a rationale."
        )
        feedback = self.encode_sample(
            self.feedback_input[idx], self.feedback_target[idx],
            "In the task of detecting whether the dialogue contains elements of mental manipulation, one student gave an answer. Please give the correct answer and point out any mistakes (if any) in the student's response."
        )
        return {
            "binary_input_ids": binary[0].long(),
            "binary_attention_mask": binary[1].long(),
            "binary_labels": binary[2].long(),
            "rationale_input_ids": rationale[0].long(),
            "rationale_attention_mask": rationale[1].long(),
            "rationale_labels": rationale[2].long(),
            "feedback_input_ids": feedback[0].long(),
            "feedback_attention_mask": feedback[1].long(),
            "feedback_labels": feedback[2].long()
        }

class ValidDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_length,
                 system_prompt="I will provide you with a dialogue. Please determine if it contains elements of mental manipulation. Answer Yes or No and give a rationale."):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __len__(self):
        return len(self.inputs)

    def encode_sample(self, prompt, target, system_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False) + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            attention_mask = [1] * self.max_length
        else:
            pad_len = self.max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            attention_mask = [1] * (len(input_ids) - pad_len) + [0] * pad_len

        return (
            torch.tensor(input_ids).long(),
            torch.tensor(attention_mask).long(),
            torch.tensor(labels).long()
        )

    def __getitem__(self, idx):
        input_ids, attention_mask, labels = self.encode_sample(
            self.inputs[idx], self.targets[idx], self.system_prompt
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def phase_training(phase,
                   model,
                   tokenizer,
                   learning_rate,
                   weight_decay,
                   num_epochs_per_phase,
                   gradient_accumulation_steps,
                   accelerator,
                   train_loader,
                   valid_loader,
                   save_path,
                   return_model,
                   valid_mark = True):
    optimizer = Lion(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in tqdm(range(num_epochs_per_phase)):

        model, optimizer = accelerator.prepare(model, optimizer)

        train(phase, epoch, model, train_loader, optimizer, accelerator, gradient_accumulation_steps)

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        save_path = os.path.join(save_path, f"lora_weights_phase_{phase}_epoch_{epoch}")
        unwrapped_model.save_pretrained(save_path)

        if valid_mark:
            report = validate(tokenizer, model, valid_loader, accelerator)
            print(report)
            with open(os.path.join(save_path, "training_logs.txt"), "a") as f:
                f.write(f"Phase {phase}, Epoch {epoch}: {report}\n")

    if return_model:
        return unwrapped_model


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    random.seed(args.seed)
    torch.cuda.empty_cache()

    accelerator = Accelerator(device_placement=True, mixed_precision="bf16")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    lora_config = peft.LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        task_type=args.task_type
    )
    model = peft.get_peft_model(model, lora_config)

    valid_input, valid_answer = preprocess_valid_data(args.valid_data_path)
    train_data = preprocess_train_data(
        row_data_path=args.train_data_path,
        correct_data_path=args.correct_data_path,
        incorrect_data_path=args.incorrect_data_path,
        feedback_data_path=args.feedback_data_path,
        EvoSA_data_path=args.EvoSA_train_data_path,
        EvoSA_correct_data_path=args.EvoSA_correct_data_path,
        EvoSA_incorrect_data_path=args.EvoSA_incorrect_data_path,
        EvoSA_feedback_data_path=args.EvoSA_feedback_data_path
    )

    train_dataset = CustomDataset(
        binary_input=train_data["binary_input"],
        binary_target=train_data["binary_target"],
        rationale_input=train_data["rationale_input"],
        rationale_target=train_data["rationale_target"],
        feedback_input=train_data["feedback_input"],
        feedback_target=train_data["feedback_target"],
        tokenizer=tokenizer,
        max_length=args.train_max_length
    )

    valid_dataset = ValidDataset(valid_input, valid_answer, tokenizer, max_length=args.valid_max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=True)
    train_loader = accelerator.prepare(train_loader)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True)
    model = phase_training(phase=1,
                           model=model,
                           tokenizer = tokenizer,
                           learning_rate=args.learning_rate,
                           weight_decay=args.weight_decay,
                           num_epochs_per_phase=args.num_epochs_per_phase,
                           gradient_accumulation_steps=args.gradient_accumulation_steps,
                           accelerator=accelerator,
                           train_loader=train_loader,
                           valid_loader=valid_loader,
                           save_path=args.save_path,
                           return_model = True)
    model = phase_training(phase=2,
                           model=model,
                           tokenizer = tokenizer,
                           learning_rate=args.learning_rate,
                           weight_decay=args.weight_decay,
                           num_epochs_per_phase=args.num_epochs_per_phase,
                           gradient_accumulation_steps=args.gradient_accumulation_steps,
                           accelerator=accelerator,
                           train_loader=train_loader,
                           valid_loader=valid_loader,
                           save_path=args.save_path,
                           return_model = True)
    phase_training(phase=3,
                   model=model,
                   tokenizer = tokenizer,
                   learning_rate=args.learning_rate,
                   weight_decay=args.weight_decay,
                   num_epochs_per_phase=args.num_epochs_per_phase,
                   gradient_accumulation_steps=args.gradient_accumulation_steps,
                   accelerator=accelerator,
                   train_loader=train_loader,
                   valid_loader=valid_loader,
                   save_path=args.save_path,
                   return_model = False)

    print("\n--------------------Training Finished--------------------\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda_visible_devices", type=str, default="0,1")
    
    parser.add_argument("--model_path", type=str, default="/data/models/Qwen2.5-3B-Instruct/")
    parser.add_argument("--tokenizer_path", type=str, default="/data/models/Qwen2.5-3B-Instruct/")
    parser.add_argument("--save_path", type=str, default="./MentalManip_con_fine_tuning_res")

    parser.add_argument("--train_data_path", type=str, default="./dataset/MentalManip_con_train.json")
    parser.add_argument("--valid_data_path", type=str, default="./dataset/MentalManip_con_valid.json")
    parser.add_argument("--correct_data_path", type=str, default="./data_for_KD/MentalManip_con_correct_data.json")
    parser.add_argument("--incorrect_data_path", type=str, default="./data_for_KD/MentalManip_con_incorrect_data.json")
    parser.add_argument("--feedback_data_path", type=str, default="./data_for_KD/MentalManip_con_feedback_data.json")

    parser.add_argument("--EvoSA_train_data_path", type=str, default="./data_for_KD/MentalManip_con_EvoSA_data.json")
    parser.add_argument("--EvoSA_correct_data_path", type=str, default="./data_for_KD/MentalManip_con_EvoSA_correct_data.json")
    parser.add_argument("--EvoSA_incorrect_data_path", type=str, default="./data_for_KD/MentalManip_con_EvoSA_incorrect_data.json")
    parser.add_argument("--EvoSA_feedback_data_path", type=str, default="./data_for_KD/MentalManip_con_EvoSA_feedback_data.json")

    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=list, default=["q_proj","v_proj"])
    parser.add_argument("--lora_dropout", type=int, default=0.05)
    parser.add_argument("--bias", type=str, default="none")
    parser.add_argument("--task_type", type=str, default="CAUSAL_LM")
    
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=int, default=1e-5)
    parser.add_argument("--weight_decay", type=int, default=0.01)
    parser.add_argument("--valid_batch_size", type=int, default=16)
    parser.add_argument("--train_max_length", type=int, default=1500)
    parser.add_argument("--valid_max_length", type=int, default=1000)
    parser.add_argument("--num_epochs_per_phase", type=int, default=1)
    
    args = parser.parse_args()

    main(args)
