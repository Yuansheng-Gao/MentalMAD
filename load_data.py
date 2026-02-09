import json
import random
import jsonlines
from collections import Counter
random.seed(42)

# After dividing the data using this code, 
# we saved it locally and will not use this code again in the future.
def load_root_data(dataset_path, train_n=0.6, valid_n=0.2):
    with open(dataset_path, mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    toxic_data = []
    benign_data = []
    for d in data:
        if d["manipulative"] == "1":
            toxic_data.append(d)
        else:
            benign_data.append(d)

    random.shuffle(toxic_data)
    random.shuffle(benign_data)

    toxic_data_num = len(toxic_data)
    benign_data_num = len(benign_data)

    train_toxic_num = round(train_n * toxic_data_num)
    train_benign_num = round(train_n * benign_data_num)
    valid_toxic_num = round(valid_n * toxic_data_num)
    valid_benign_num = round(valid_n * benign_data_num)

    train_data = toxic_data[:train_toxic_num] + benign_data[:train_benign_num]
    valid_data = (
        toxic_data[train_toxic_num : train_toxic_num + valid_toxic_num]
        + benign_data[train_benign_num : train_benign_num + valid_benign_num]
    )
    test_data = (
        toxic_data[train_toxic_num + valid_toxic_num :]
        + benign_data[train_benign_num + valid_benign_num :]
    )

    return train_data, valid_data, test_data


def get_rationale(text):
    text = text.replace("Rationale:","")
    return text.strip()

def get_feedback(text):
    text = text.replace("Feedback:","")
    return text.strip()


def oversample_text_data(train_data, resample_yes_num=300):
    # Extract Category: Assume the target format is "Yes. Rationale: ..." or "No. Rationale: ..."
    categories = [t[0].strip() for t in train_data["binary_target"]]

    # Count the number of each category
    label_counts = Counter(categories)
    yes_count = label_counts.get("Y", 0)
    no_count = label_counts.get("N", 0)

    # -------------------------
    # 1) First, resample 300 from the "Yes" category
    # -------------------------
    yes_indices = [i for i, c in enumerate(categories) if c == "Y"]

    for _ in range(resample_yes_num):
        idx = random.choice(yes_indices)
        for key in train_data:
            train_data[key].append(train_data[key][idx])

    # Update the current count
    new_yes_count = yes_count + resample_yes_num

    # -------------------------
    # 2) Resample "No" based on the original ratio
    #    Original ratio = no_count / yes_count
    # -------------------------
    if yes_count == 0:
        return train_data  # Extreme case: If there is no "Yes," return directly.

    original_ratio = no_count / yes_count
    target_no_count = int(new_yes_count * original_ratio)

    add_no_num = max(0, target_no_count - no_count)

    no_indices = [i for i, c in enumerate(categories) if c == "N"]

    for _ in range(add_no_num):
        idx = random.choice(no_indices)
        for key in train_data:
            train_data[key].append(train_data[key][idx])

    return train_data


def preprocess_train_data(row_data_path, correct_data_path, incorrect_data_path, feedback_data_path, 
                          EvoSA_data_path, EvoSA_correct_data_path, EvoSA_incorrect_data_path, EvoSA_feedback_data_path):
    
    try:
        with open(row_data_path, 'r', encoding='utf-8') as f:
            row_data = [json.loads(line) for line in f]
    except:
        with open(row_data_path, mode='r', encoding='utf-8') as json_file:
            row_data = json.load(json_file)
    
    correct_data_list = []
    with jsonlines.open(correct_data_path) as reader:
        for obj in reader:
            correct_data_list.append(obj)

    incorrect_data_list = []
    with jsonlines.open(incorrect_data_path) as reader:
        for obj in reader:
            incorrect_data_list.append(obj)

    feedback_data_list = []
    with jsonlines.open(feedback_data_path) as reader:
        for obj in reader:
            feedback_data_list.append(obj)

    rationale_input = []
    rationale_target = []
    binary_input = []
    binary_target = []

    for i in range(len(row_data)):
        binary_input.append(row_data[i]["dialogue"])
        rationale_input_s = row_data[i]["dialogue"]
        rationale_input.append(rationale_input_s)

        if row_data[i]["manipulative"] == "0":
            binary_target.append("No")
            rationale_target_s = "No. Rationale: "+get_rationale(correct_data_list[i]["response"])
        else:
            binary_target.append("Yes")
            rationale_target_s = "Yes. Rationale: "+get_rationale(correct_data_list[i]["response"])
        rationale_target.append(rationale_target_s)

    EvoSA_row_data = []
    with jsonlines.open(EvoSA_data_path) as reader:
        for obj in reader:
            EvoSA_row_data.append(obj)

    EvoSA_correct_data = []
    with jsonlines.open(EvoSA_correct_data_path) as reader:
        for obj in reader:
            EvoSA_correct_data.append(obj)

    EvoSA_incorrect_data = []
    with jsonlines.open(EvoSA_incorrect_data_path) as reader:
        for obj in reader:
            EvoSA_incorrect_data.append(obj)

    EvoSA_feedback_data = []
    with jsonlines.open(EvoSA_feedback_data_path) as reader:
        for obj in reader:
            EvoSA_feedback_data.append(obj)


    for i in range(len(EvoSA_row_data)):
        binary_input.append(EvoSA_row_data[i]["dialogue"])
        rationale_input_s = EvoSA_row_data[i]["dialogue"]
        rationale_input.append(rationale_input_s)

        if EvoSA_row_data[i]["manipulative"] == "0":
            binary_target.append("No")
        else:
            binary_target.append("Yes")

        if EvoSA_correct_data[i]["true_label"] == "0":
            rationale_target_s = "No. Rationale: "+get_rationale(EvoSA_correct_data[i]["response"])
        else:
            rationale_target_s = "Yes. Rationale: "+get_rationale(EvoSA_correct_data[i]["response"])
        rationale_target.append(rationale_target_s)

    feedback_input = []
    feedback_target = []

    for i in range(len(row_data)):    
        feedback_input_s = f'''### Dialogue:
        {row_data[i]["dialogue"]}
        
        ### Student's Response:
        {incorrect_data_list[i]["response"]}'''
        feedback_input.append(feedback_input_s)
        if row_data[i]["manipulative"] == "0":
            feedback_target_s = "Correct answer: No. "+get_feedback(feedback_data_list[i]["response"])
        else:
            feedback_target_s = "Correct answer: Yes. "+get_feedback(feedback_data_list[i]["response"])

        feedback_target.append(feedback_target_s)

    for i in range(len(EvoSA_row_data)):

        feedback_input_s = f'''### Dialogue:
        {EvoSA_row_data[i]["dialogue"]}
        
        ### Student's Response:
        {EvoSA_incorrect_data[i]["response"]}'''
        feedback_input.append(feedback_input_s)
        if EvoSA_correct_data[i]["true_label"] == "0":
            feedback_target_s = "Correct answer: No. "+get_feedback(EvoSA_feedback_data[i]["response"])
        else:
            feedback_target_s = "Correct answer: Yes. "+get_feedback(EvoSA_feedback_data[i]["response"])

        feedback_target.append(feedback_target_s)

    train_data = {
        "binary_input": binary_input,
        "binary_target": binary_target,
        "rationale_input": rationale_input,
        "rationale_target": rationale_target,
        "feedback_input": feedback_input,
        "feedback_target": feedback_target
    }
    # train_data = oversample_text_data(train_data)
    
    return train_data


def preprocess_valid_data(valid_data_path):
    input = []
    answer = []
    try:
        with open(valid_data_path, 'r', encoding='utf-8') as f:
            valid_data = [json.loads(line) for line in f]
    except:
        with open(valid_data_path, mode='r', encoding='utf-8') as json_file:
            valid_data = json.load(json_file)

    for vd in valid_data:

        input_s = vd["dialogue"]
        input.append(input_s)
        answer.append('Yes' if vd["manipulative"] == '1' else 'No')

    return input, answer