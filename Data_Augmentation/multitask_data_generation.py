from llm_response import send_request
from tqdm import tqdm
import jsonlines
import json
import random
random.seed(42)

model_name = "llama3-70b-it"

def correct_response(dialogue, label):
    system_prompt = f'''You are an advanced dialogue analysis agent. Using your knowledge of dark psychology based on the definition of mental manipulation, please explain why this dialogue {"does not contain" if label == "0" else "contains"} elements of mental manipulation. Let's think step by step.
    
    ### Definition of Mental Manipulation:
    Mental manipulation is using language to influence, alter, or control an individual's psychological state or perception for the manipulator's benefit.

    ### Output Format:
    Rationale: [Provide only strong evidence using direct dialogue quotes. Clearly explain how the language used aligns—or does not align—with known manipulation tactics.]
    '''

    user_prompt = f'''### Dialogue:
    {dialogue}'''

    message = [{"role": "system","content":system_prompt}, {"role": "user","content": user_prompt}]
    
    model_config = {
        "temperature": 0,
        "max_tokens": 512
    }
    
    response = send_request(message, model_name, model_config)

    return response


def incorrect_response(dialogue, label):
    system_prompt = f'''You are an advanced dialogue analysis agent. Using your knowledge of dark psychology based on the definition of mental manipulation, please explain why this dialogue {"does not contain" if label == "1" else "contains"} elements of mental manipulation. Let's think step by step.
    
    ### Definition of Mental Manipulation:
    Mental manipulation is using language to influence, alter, or control an individual's psychological state or perception for the manipulator's benefit.
    
    ### Output Format:
    Rationale: [Provide only strong evidence using direct dialogue quotes. Clearly explain how the language used aligns—or does not align—with known manipulation tactics.]'''

    user_prompt = f'''### Dialogue:
    {dialogue}'''

    message = [{"role": "system","content":system_prompt}, {"role": "user","content": user_prompt}]
    
    model_config = {
        "temperature": 0,
        "max_tokens": 512
    }
    
    response = send_request(message, model_name, model_config)

    return response


def feedback_response(dialogue, incorrect, label):
    system_prompt = f'''You are an advanced dialogue analysis teacher. In the task of detecting whether the dialogue contained elements of mental manipulation, students gave incorrect answers. You should point out the mistakes in the student's answer using knowledge of dark psychology based on the definition of mental manipulation. Let's think step by step.
    
    ### Definition of Mental Manipulation:
    Mental manipulation is using language to influence, alter, or control an individual's psychological state or perception for the manipulator's benefit.
    
    ### Hint:
    This dialogue {"does not contain" if label == "0" else "contains"} elements of mental manipulation.
    
    ### Output Format:
    Feedback: [Provide student's mistakes.]'''

    user_prompt = f'''### Dialogue:
    {dialogue}
    
    ### Student's Answer:
    {incorrect}'''
    
    message = [{"role": "system","content":system_prompt}, {"role": "user","content": user_prompt}]
    
    model_config = {
        "temperature": 0,
        "max_tokens": 512
    }
    
    response = send_request(message, model_name, model_config)

    return response

def correct_or_incorrect_data_generation(dataset, data_type, output_path):
    for data in tqdm(dataset):
        if data_type.lower() == "correct":
            res = correct_response(data["dialogue"],data["manipulative"])
        else:
            res = incorrect_response(data["dialogue"],data["manipulative"])

        new_data={
            "id1": data["id1"],
            "id2": data["id2"],
            "true_label": data["manipulative"],
            "response": res
        }

        with jsonlines.open(output_path, mode='a') as writer:
            writer.write(new_data)


def feedback_data_generation(dataset, incorrect, output_path):

    for i in tqdm(range(len(dataset))):
        res = feedback_response(dataset[i]["dialogue"],incorrect[i]["response"],dataset[i]["manipulative"])
        new_data={
            "id1": dataset[i]["id1"],
            "id2": dataset[i]["id2"],
            "true_label": dataset[i]["manipulative"],
            "response": res
        }

        with jsonlines.open(output_path, mode='a') as writer:
            writer.write(new_data)


if __name__ == "__main__":

    with open("./dataset/MentalManip_con_train.json", mode='r', encoding='utf-8') as json_file:
        data = [json.loads(file) for file in json_file]
    correct_or_incorrect_data_generation(data, "correct", "./data_for_CoCoDistill/MentalManip_con_correct_data.json")
    correct_or_incorrect_data_generation(data, "incorrect", "./data_for_CoCoDistill/MentalManip_con_incorrect_data.json")

    with open("./data_for_CoCoDistill/MentalManip_con_incorrect.json", mode='r', encoding='utf-8') as json_file:
        incorrect_data = [json.loads(file) for file in json_file]
    feedback_data_generation(data, incorrect_data, "./data_for_CoCoDistill/MentalManip_con_feedback_data.json")
