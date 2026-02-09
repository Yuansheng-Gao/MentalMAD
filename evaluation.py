import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def extract_label_from_response(response):
    response = response.lower()
    if "yes" in response or "1" in response:
        return 1
    else:
        return 0


# Read the data.
with open("./data_for_KD/student_test__.json", "r", encoding="utf-8") as f:
    data = [json.loads(line.strip()) for line in f]

# Extract the predicted labels and true labels.
pre_label = []
true_label = []
for i in range(len(data)):
    if extract_label_from_response(data[i]["answer"])!=-1:
        pre_label.append(extract_label_from_response(data[i]["answer"]))
        true_label.append(extract_label_from_response(data[i]["manipulative"]))

# Calculate the confusion matrix (for binary classification 0/1).
tn, fp, fn, tp = confusion_matrix(true_label, pre_label).ravel()

# Calculate the metrics.
acc = accuracy_score(true_label, pre_label)
pre = precision_score(true_label, pre_label, average='binary')
rec = recall_score(true_label, pre_label, average='binary')
f1_mac = f1_score(true_label, pre_label, average='macro')
f1_weight = f1_score(true_label, pre_label, average='weighted')

# Print the results, keeping three decimal places.
print(f"TP: {tp}")
print(f"FN: {fn}")
print(f"FP: {fp}")
print(f"TN: {tn}\n")

print(f"Accuracy   : {acc:.3f}")
print(f"Precision  : {pre:.3f}")
print(f"Recall     : {rec:.3f}")
print(f"F1-macro   : {f1_mac:.3f}")
print(f"F1-weighted: {f1_weight:.3f}")
