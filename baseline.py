import csv
from sklearn.metrics import confusion_matrix

provider_names = {}
regions = {}
diagnosis_codes = {}
diagnosis_names = {}

# read all training files and form rules
with open('data/training/Diagnosis_Code_Approve.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        diagnosis_codes[row[0]] = 1

with open('data/training/Diagnosis_Code_Deny.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        diagnosis_codes[row[0]] = 0

with open('data/training/Diagnosis_Name_Approve.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        diagnosis_names[row[0]] = 1

with open('data/training/Diagnosis_Name_Deny.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        diagnosis_names[row[0]] = 0

with open('data/training/provier_approve.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        provider_names[row[0]] = 1

with open('data/training/provier_deny.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        provider_names[row[0]] = 0

with open('data/training/Region_Approve.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        regions[row[0]] = 1

with open('data/training/Region_Deny.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        regions[row[0]] = 0

# read 100% approved and denied in the test set
test_provider_names = {}
test_regions = {}
test_diagnosis_codes = {}
test_diagnosis_names = {}

with open('data/test/Diagnosis_Code_Approve.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        test_diagnosis_codes[row[0]] = 1

with open('data/test/Diagnosis_Code_Deny.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        test_diagnosis_codes[row[0]] = 0

with open('data/test/Diagnosis_Name_Approve.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        test_diagnosis_names[row[0]] = 1

with open('data/test/Diagnosis_Name_Deny.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        test_diagnosis_names[row[0]] = 0

with open('data/test/Provider_Approve.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        test_provider_names[row[0]] = 1

with open('data/test/Provider_Deny.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        test_provider_names[row[0]] = 0

with open('data/test/Region_Approve.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        test_regions[row[0]] = 1

with open('data/test/Region_Deny.csv', newline='') as csvfile:
    diagnosis_reader = csv.reader(csvfile, delimiter=',')
    header = True
    for row in diagnosis_reader:
        if header:
            header = False
            continue
        test_regions[row[0]] = 0
# ****************************************

predicted = []
labeled = []

total_correct_prediction = 0
total_100_in_train_and_test = 0
total_100_correct_prediction = 0
total_below_100_or_100_in_test_not_train = 0
total_below_100_or_100_in_test_not_train_correct_prediction = 0
# read test file
with open('data/test.csv', newline='') as testfile:
    test_reader = csv.DictReader(testfile, delimiter=',')
    check_columns = ["feed_id_full_path", "Diagnosis_Code", "Diagnosis_Name", "Requesting_Provider", "Overall_Auth_Status","Line_Item_Status","Status_Reason"]
    header = True
    for row in test_reader:
        if header:
            header = False
            continue

        if 'APPROVE' == row['Overall_Auth_Status']:
            current_label = 1
        else:
            current_label = 0

        labeled.append(current_label)

        provider_name = row['Requesting_Provider']
        region = row['feed_id_full_path']
        diagnosis_code = row['Diagnosis_Code']
        diagnosis_name = row['Diagnosis_Name']

        # try to make incorrect prediction xor 1
        predicted_label = current_label ^ 1

        if provider_name in provider_names:
            predicted_label = provider_names[provider_name]
            predicted.append(predicted_label)
            total_100_in_train_and_test = total_100_in_train_and_test + 1
        elif region in regions:
            predicted_label = regions[region]
            predicted.append(predicted_label)
            total_100_in_train_and_test = total_100_in_train_and_test + 1
        elif diagnosis_code in diagnosis_codes:
            predicted_label = diagnosis_codes[diagnosis_code]
            predicted.append(predicted_label)
            total_100_in_train_and_test = total_100_in_train_and_test + 1
        elif diagnosis_name in diagnosis_names:
            predicted_label = diagnosis_names[diagnosis_name]
            predicted.append(diagnosis_names[diagnosis_name])
            total_100_in_train_and_test = total_100_in_train_and_test + 1
        else:
            # try to make incorrect prediction
            predicted_label = 1 # predict true by default
            predicted.append(predicted_label)

        if predicted_label == current_label:
            total_correct_prediction = total_correct_prediction + 1
        # count 100% correct prediction
        if provider_name in test_provider_names and provider_name in provider_names and current_label == provider_names[provider_name]:
                total_100_correct_prediction = total_100_correct_prediction + 1
        elif region in test_regions and region in regions and current_label == regions[region]:
                total_100_correct_prediction = total_100_correct_prediction + 1
        elif diagnosis_code in test_diagnosis_codes and diagnosis_code in diagnosis_codes and current_label == diagnosis_codes[diagnosis_code]:
                total_100_correct_prediction = total_100_correct_prediction + 1
        elif diagnosis_name in test_diagnosis_names and diagnosis_name in diagnosis_names and current_label == diagnosis_names[diagnosis_name]:
                total_100_correct_prediction = total_100_correct_prediction + 1

        # count < 100% accuracy
        if (provider_name not in provider_names) and \
            (region not in regions) and \
            (diagnosis_code not in diagnosis_codes) and \
            (diagnosis_name not in diagnosis_names):

            total_below_100_or_100_in_test_not_train = total_below_100_or_100_in_test_not_train + 1
            if predicted_label == current_label:
                total_below_100_or_100_in_test_not_train_correct_prediction = total_below_100_or_100_in_test_not_train_correct_prediction + 1


CM = confusion_matrix(labeled, predicted, labels=[1, 0])
TN, FP, FN, TP = CM.ravel()
# TN = CM[0][0]
# FN = CM[1][0]
# TP = CM[1][1]
# FP = CM[0][1]
count_true = 0
for i in range(len(labeled)):
    if labeled[i] == predicted[i]:
        count_true = count_true + 1

precision = TP / (TP + FP)
recall = TP / (TP + FN)
fscore = (2*TP) / (2*TP + FP + FN)
accuracy = (TP + TN) / (TP + TN + FN + FP)

print('total test set:', len(predicted))
print('total test set -correct prediction:', total_correct_prediction)

print()
print('total 100% accuracy:', total_100_in_train_and_test)
print('total 100% accuracy correct prediction:', total_100_correct_prediction)
print('100% accuracy prediction: ', (total_100_correct_prediction / total_100_in_train_and_test))

print()
print('total below or 100% accuracy in test not in train:', total_below_100_or_100_in_test_not_train)
print('total below or 100% accuracy in test not in train - correct prediction:', total_below_100_or_100_in_test_not_train_correct_prediction)
print('below or 100% accuracy in test not in train -  accuracy prediction: ', (total_below_100_or_100_in_test_not_train_correct_prediction / total_below_100_or_100_in_test_not_train))

print()
print('Overall')
print('precision:', precision, ';recall:', recall)
print('f-score:', fscore, ';accuracy:', accuracy)

