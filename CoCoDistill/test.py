import os
import torch
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def clean_adapter_config(adapter_dir: str):
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if not cfg_path.exists():
        print(f"[WARN] {cfg_path} does not exist. Skipping cleanup.")
        return

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    unnecessary_keys = [
        "corda_config", "qalora_group_size", "use_qalora",
        "eva_config", "loftq_config",
        "trainable_token_indices", "exclude_modules", "lora_bias"
    ]
    for key in unnecessary_keys:
        cfg.pop(key, None)

    cfg_path.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"[INFO] Cleaned {cfg_path}")


def get_student_response(dataset, output_path, model, tokenizer, device):
    system_prompt = (
        "I will provide you with a dialogue. Please determine if it contains "
        "elements of mental manipulation. Answer Yes or No and give a rationale."
    )

    for data in tqdm(dataset):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data["dialogue"]}
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([prompt], return_tensors="pt")
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_ids = generated_ids[0][model_inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        with jsonlines.open(output_path, "a") as writer:
            writer.write({
                "id": data["id"],
                "manipulative": data["manipulative"],
                "answer": response
            })


def get_IAP_response(dataset, output_path, model, tokenizer, device):
    system_prompt_1 = (
        "I will provide you with a dialogue. "
        "Please summarize the intent of the statement made by Person1 in one sentence."
    )
    system_prompt_2 = (
        "I will provide you with a dialogue. "
        "Please summarize the intent of the statement made by Person2 in one sentence."
    )
    system_prompt = (
        "I will provide you with a dialogue and intent of person1, and intent of person2. "
        "Please carefully analyze the dialogue and intents, and determine if it contains "
        "elements of mental manipulation. Just answer with 'Yes' or 'No'."
    )

    for data in tqdm(dataset):
        # intent p1
        messages = [{"role": "system", "content": system_prompt_1},
                    {"role": "user", "content": data["dialogue"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        intent_p1 = tokenizer.decode(
            model.generate(**model_inputs, max_new_tokens=512)[0],
            skip_special_tokens=True
        )

        # intent p2
        messages = [{"role": "system", "content": system_prompt_2},
                    {"role": "user", "content": data["dialogue"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        intent_p2 = tokenizer.decode(
            model.generate(**model_inputs, max_new_tokens=512)[0],
            skip_special_tokens=True
        )

        # final
        user_input = f"{data['dialogue']} {intent_p1} {intent_p2}"
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        answer = tokenizer.decode(
            model.generate(**model_inputs, max_new_tokens=1)[0],
            skip_special_tokens=True
        )

        with jsonlines.open(output_path, "a") as writer:
            writer.write({
                "id": data["id"],
                "manipulative": data["manipulative"],
                "intent1": intent_p1,
                "intent2": intent_p2,
                "answer": answer
            })


def get_self_percept_response(dataset, output_path, model, tokenizer, device):
    system_prompt_stage1 = """
    Stage 1 – Observation of Behavior
    Story/Conversation:
    {dialogue}
    Question: What behaviors and statements indicate the attitudes or beliefs of each character?
    """

    system_prompt_stage2 = """
    Stage 2 – Self-Inference
    Based on the observations below:
    {stage1_output}
    Is there any manipulation detected? Answer Yes or No.
    """

    for data in tqdm(dataset):
        text1 = system_prompt_stage1.format(dialogue=data["dialogue"])
        messages = [{"role": "system", "content": text1}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        stage1_output = tokenizer.decode(
            model.generate(**model_inputs, max_new_tokens=512)[0],
            skip_special_tokens=True
        )

        text2 = system_prompt_stage2.format(stage1_output=stage1_output)
        messages = [{"role": "system", "content": text2}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        answer = tokenizer.decode(
            model.generate(**model_inputs, max_new_tokens=1)[0],
            skip_special_tokens=True
        )

        with jsonlines.open(output_path, "a") as writer:
            writer.write({
                "id": data["id"],
                "manipulative": data["manipulative"],
                "stage1_output": stage1_output,
                "answer": answer
            })


def main(args):
    random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.test_data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        padding_side="left"
    )

    for e in range(args.num_lora_epochs):
        clean_adapter_config(f"{args.lora_root}/lora_weights_epoch_{e}")

    model = PeftModel.from_pretrained(
        model,
        f"{args.lora_root}/lora_weights_epoch_{args.use_lora_epoch}"
    ).to(device)

    if args.mode == "ours":
        get_student_response(data, args.output_path, model, tokenizer, device)
    elif args.mode == "iap":
        get_IAP_response(data, args.output_path, model, tokenizer, device)
    elif args.mode == "sp":
        get_self_percept_response(data, args.output_path, model, tokenizer, device)
    else:
        raise ValueError("mode must be one of: ours | iap | sp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda_visible_devices", type=str, default="1")
    parser.add_argument("--dataset_root_path", type=str,
                        default="./dataset/mentalmanip_con.json")
    parser.add_argument("--test_data_path", type=str,
                        default="./dataset/MentalManip_con_test.json")
    parser.add_argument("--model_path", type=str,
                        default="/data/models/Qwen2.5-3B-Instruct/")
    parser.add_argument("--tokenizer_path", type=str,
                        default="/data/models/Qwen2.5-3B-Instruct/")
    parser.add_argument("--lora_root", type=str,
                        default="./MentalManip_con_fine_tuning_res")
    parser.add_argument("--num_lora_epochs", type=int, default=3)
    parser.add_argument("--use_lora_epoch", type=int, default=2)
    parser.add_argument("--output_path", type=str,
                        default="./results/student_test.json")
    parser.add_argument("--mode", type=str,
                        choices=["ours", "iap", "sp"],
                        default="ours")

    args = parser.parse_args()
    main(args)
