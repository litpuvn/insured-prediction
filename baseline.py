import csv
from sklearn.metrics import confusion_matrix

provider_names = {}
regions = {}
diagnosis_codes = {}
diagnosis_names = {}

# read all training files and form rules
with open('data/archive/Diagnosis_Code_Approve.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        diagnosis_codes[row[0]] = 1

with open('data/archive/Diagnosis_Code_Deny.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        diagnosis_codes[row[0]] = 0

with open('data/archive/Diagnosis_Name_Approve.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        diagnosis_names[row[0]] = 1

with open('data/archive/Diagnosis_Name_Deny.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        diagnosis_names[row[0]] = 0

with open('data/archive/provier_approve.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        provider_names[row[0]] = 1

with open('data/archive/provier_deny.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        provider_names[row[0]] = 0

with open('data/archive/Region_Approve.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        regions[row[0]] = 1

with open('data/archive/Region_Deny.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        regions[row[0]] = 0

predicted = []
labeled = []

# read test file
with open('data/test.csv', newline='') as testfile:
    test_reader = csv.DictReader(testfile, delimiter=',')
    check_columns = ["feed_id_full_path", "Diagnosis_Code", "Diagnosis_Name", "Requesting_Provider", "Overall_Auth_Status","Line_Item_Status","Status_Reason"]
    header = True
    for row in test_reader:
        if header:
            header = False
            continue

        current_label = 'APPROVE' == row['Overall_Auth_Status']
        labeled.append(current_label)

        provider_name = row['Requesting_Provider']
        region = row['feed_id_full_path']
        diagnosis_code = row['Diagnosis_Code']
        diagnosis_name = row['Diagnosis_Name']

        if provider_name in provider_names:
            predicted.append(provider_names[provider_name])
        elif region in regions:
            predicted.append(regions[region])
        elif diagnosis_code in diagnosis_codes:
            predicted.append(diagnosis_codes[diagnosis_code])
        elif diagnosis_name in diagnosis_names:
            predicted.append(diagnosis_names[diagnosis_name])
        else:
            # try to make incorrect prediction
            predicted.append(not current_label)

CM = confusion_matrix(labeled, predicted)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

precision = TP / (TP + FP)
recall = TP / (TP + FN)
fscore = (2*TP) / (2*TP + FP + FN)
accuracy = (TP + FN) / (TP + TN + FN + FP)

print('precision:', precision, ';recall:', recall, ';f-score:', fscore, ';accuracy:', accuracy)

