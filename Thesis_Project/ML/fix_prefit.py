import json, re

with open("ML_Fraud_Detection_2.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

def fix_source(lines):
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Single-line isotonic prefit
        if re.search(r'CalibratedClassifierCV\(\w+,\s*cv="prefit",\s*method="isotonic"\)', line):
            m = re.search(r'(\w+)\s*=\s*CalibratedClassifierCV\((\w+)', line)
            lhs = m.group(1) if m else 'cal'
            base = m.group(2) if m else 'base'
            indent = re.match(r'^(\s*)', line).group(1)
            new_lines.append(f'{indent}{lhs} = _PrefitCalibratedClassifier({base}, method="isotonic")\n')
            i += 1
            continue

        # Single-line sigmoid prefit
        if re.search(r'CalibratedClassifierCV\(\w+,\s*cv="prefit",\s*method="sigmoid"\)', line):
            m = re.search(r'(\w+)\s*=\s*CalibratedClassifierCV\((\w+)', line)
            lhs = m.group(1) if m else 'cal'
            base = m.group(2) if m else 'base'
            indent = re.match(r'^(\s*)', line).group(1)
            new_lines.append(f'{indent}{lhs} = _PrefitCalibratedClassifier({base}, method="sigmoid")\n')
            i += 1
            continue

        # Multi-line: line ends with CalibratedClassifierCV(
        stripped = line.rstrip()
        # also handle "    calibrated_rf_final = CalibratedClassifierCV(\n"
        if re.search(r'CalibratedClassifierCV\($', stripped) or re.search(r'=\s*CalibratedClassifierCV\(\s*$', stripped):
            m_lhs = re.search(r'(\w+)\s*=\s*CalibratedClassifierCV', line)
            lhs = m_lhs.group(1) if m_lhs else 'cal'
            indent = re.match(r'^(\s*)', line).group(1)
            j = i + 1
            base = None
            method = "isotonic"
            # scan continuation lines until closing )
            while j < len(lines):
                seg = lines[j]
                seg_stripped = seg.strip().rstrip(',')
                if base is None:
                    m_b = re.search(r'^\s*([\w_]+)\s*,', seg)
                    if m_b:
                        base = m_b.group(1)
                if 'sigmoid' in seg:
                    method = "sigmoid"
                if seg.rstrip().endswith(')'):
                    j += 1
                    break
                j += 1
            if base is None:
                m2 = re.search(r'CalibratedClassifierCV\((\w+)', line)
                if m2:
                    base = m2.group(1)
            new_lines.append(f'{indent}{lhs} = _PrefitCalibratedClassifier({base}, method="{method}")\n')
            i = j
            continue

        new_lines.append(line)
        i += 1
    return new_lines

changed = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = cell['source']
    new_src = fix_source(src)
    if new_src != src:
        cell['source'] = new_src
        changed += 1

print(f"Fixed {changed} cells")

with open("ML_Fraud_Detection_2.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Saved.")
