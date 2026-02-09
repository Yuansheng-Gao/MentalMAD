from llm_response import send_request
from tqdm import tqdm
import jsonlines
import json
import random
random.seed(42)

model_name = "llama3-70b-it"

def generate_unique_pairs(n, min_val, max_val):
    pairs = set()
    while len(pairs) < n:
        a, b = random.randint(min_val, max_val), random.randint(min_val, max_val)
        pair = tuple(sorted((a, b)))
        if a != b:
            pairs.add(pair)
    
    return list(pairs)
    
def EvoSA_get_child_dialogue(dataset,label,max_k,output_path):
    parent_data = []

    for i in tqdm(range(len(dataset))):
        if dataset[i]["manipulative"] == label: parent_data.append(dataset[i])
    min_val = 0
    max_val = len(parent_data)-1
    maxn = (max_val-min_val+1)*(max_val-min_val)//2
    n = maxn
    pairs = generate_unique_pairs(n, min_val, max_val)

    k = 0
    for i,j in tqdm(pairs):
        k += 1
        if k > max_k:break
        system_prompt = f'''You are an advanced dialogue analysis agent. Your task is to generate a new dialogue that is different from Dialogue1 and Dialogue2.
        
        ### Generation process:
        Step 1: Choose specific lines from two dialogues (Dialogue1 and Dialogue2). These lines should reflect distinct speech acts or conversational strategies.
        Step 2: Combine the selected lines from Dialogue1 and Dialogue2 to form a new dialogue.
        Step 3: Introduce greater changes in the new dialogue, such as altering the tone, wording, roles, themes, and context.
        Step 4: Polish the newly generated dialogue to ensure logical coherence and natural flow. Improve the structure so that different linguistic behaviours are used and differentiated appropriately. Avoid redundancy or unnatural phrasing and conform to natural dialogue patterns.
        Step 5: It is known that both Dialogue1 and Dialogue2 {"contain" if label == '1' else "do not contain"} elements of mental manipulation. Please analyse why Dialogue1 and Dialogue2 {"contain" if label == '1' else "do not contain"} elements of mental manipulation.
        Step 6: Based on the analysis in Step 5, ensure that the newly generated dialogue {"contains" if label == '1' else "does not contain"} elements of mental manipulation, as in the case of Dialogue1 and Dialogue2.
        Step 7: If necessary, substantially modify the new dialogue to achieve Step 4 and Step 6, while ensuring the length of the new dialogue remains between the lengths of Dialogue1 and Dialogue2.
        
        ### Output format:
        Here is the new dialogue: [Only output the new dialogue, no other extraneous content is allowed]'''

        user_prompt = f'''### Dialogue1:
        {parent_data[i]["dialogue"]}

        ### Dialogue2:
        {parent_data[j]["dialogue"]}'''

        message = [{"role": "system","content": system_prompt},{"role": "user","content": user_prompt}]
        model_config = {
            "temperature": 0,
            "max_tokens": 1024
        }
        response = send_request(message,model_name,model_config)
        response = response.replace("Here is the new dialogue:\n\n","")

        child_data={
            "id1": parent_data[i]["id"],
            "id2": parent_data[j]["id"],
            "manipulative": label,
            "dialogue": response
        }

        with jsonlines.open(output_path, mode='a') as writer:
            writer.write(child_data)


def naive_get_child_dialogue(dataset,label,max_k,output_path):
    parent_data = []

    for i in tqdm(range(len(dataset))):
        if dataset[i]["manipulative"] == label: parent_data.append(dataset[i])
    min_val = 0
    max_val = len(parent_data)-1
    maxn = (max_val-min_val+1)*(max_val-min_val)//2
    n = maxn
    pairs = generate_unique_pairs(n, min_val, max_val)

    k = 0
    for i,j in tqdm(pairs):
        k += 1
        if k > max_k:break
        system_prompt = f'''You are an advanced dialogue analysis agent. Your task is to generate a new dialogue that is different from Dialogue1 and Dialogue2.
        
        ### Generation process:
        Step 1: Read Dialogue1 and Dialogue2 carefully.
        Step 2: Generate a new dialogue that is coherent and natural, and clearly different from both Dialogue1 and Dialogue2.
        Step 3: Both Dialogue1 and Dialogue2 {"contain" if label == '1' else "do not contain"} elements of mental manipulation. Ensure that the newly generated dialogue also {"contain" if label == '1' else "do not contain"} elements of mental manipulation.
        Step 4: Polish the new dialogue to make it logically coherent and natural.

        ### Output format:
        Here is the new dialogue: [Only output the new dialogue, no other extraneous content is allowed]'''

        user_prompt = f'''### Dialogue1:
        {parent_data[i]["dialogue"]}

        ### Dialogue2:
        {parent_data[j]["dialogue"]}'''

        message = [{"role": "system","content": system_prompt},{"role": "user","content": user_prompt}]
        model_config = {
            "temperature": 0,
            "max_tokens": 1024
        }
        response = send_request(message,model_name,model_config)
        response = response.replace("Here is the new dialogue:\n\n","")

        child_data={
            "id1": parent_data[i]["id"],
            "id2": parent_data[j]["id"],
            "manipulative": label,
            "dialogue": response
        }

        with jsonlines.open(output_path, mode='a') as writer:
            writer.write(child_data)

if __name__ == "__main__":
    input_path = "./dataset/MentalManip_con_train.json"
    output_path = "./data_for_CoCoDistill/MentalManip_con_EvoSA_data.json"
    yes_child_num = 300
    no_child_num = 189

    with open(input_path, mode='r', encoding='utf-8') as json_file:
        train_data = [json.loads(file) for file in json_file]

    EvoSA_get_child_dialogue(train_data,"1", yes_child_num, output_path)
    EvoSA_get_child_dialogue(train_data,"0", no_child_num, output_path)

    # Comparator
    # naive_get_child_dialogue(train_data, "1", yes_num,output_path)
    # naive_get_child_dialogue(train_data, "0", no_num,output_path)
