import json, re

with open('/Users/roya/Desktop/Thesis/Thesis_Project/ML/ML_Fraud_Detection_2.ipynb') as f:
    nb = json.load(f)

# Check remaining prefit calls
actual_prefit = []
for cell in nb['cells']:
    if cell['cell_type'] != 'code': continue
    for line in cell['source']:
        if 'CalibratedClassifierCV(' in line and 'prefit' in line and 'Drop-in' not in line:
            actual_prefit.append(repr(line[:120]))

print("Remaining actual prefit calls:", len(actual_prefit))
for l in actual_prefit:
    print(l)

# Also show PrefitCalibratedClassifier usages
pcc = []
for cell in nb['cells']:
    if cell['cell_type'] != 'code': continue
    for line in cell['source']:
        if '_PrefitCalibratedClassifier(' in line:
            pcc.append(repr(line[:120]))
print("\n_PrefitCalibratedClassifier usages:", len(pcc))
for l in pcc:
    print(l)
