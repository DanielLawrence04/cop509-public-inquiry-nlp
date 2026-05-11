"""Classify Grenfell rows of interest."""
from src.pdf_loader import extract_pages
from src.response_units import extract_response_units
from src.classification import classify_with_confidence

pages = extract_pages('data/raw/Grenfell-Phase2-Response.pdf')
units = extract_response_units(pages)
print(f"Total units: {len(units)}")
print()
flat = {}
for u in units:
    for lbl in u.get('recommendation_labels') or [u.get('recommendation_label')]:
        if lbl:
            flat[lbl] = u
WANT = ['14','50','51','54','55','58']
for lbl in WANT:
    u = flat.get(lbl)
    if u is None:
        print(f'{lbl}: NOT FOUND')
        continue
    rt = (u.get('response_text') or '').strip()
    label, conf = classify_with_confidence(rt)
    head = rt[:240].replace('\n', ' ')
    print(f"{lbl:5s}  {label:20s}  conf={conf:.2f}  {head}")
