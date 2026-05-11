"""Run classifier across a doc's response units and show per-row labels."""
import sys
from src.pdf_loader import extract_pages
from src.response_units import extract_response_units
from src.classification import classify_with_confidence

DOCS = {
    'bc':     'data/raw/Behaviour-Change-Response.pdf',
    'po':     'data/raw/PostOfficeHorizon-IT-Inquiry-Response.pdf',
    'space':  'data/raw/TheSpaceEconomyResponse.pdf',
}
which = sys.argv[1] if len(sys.argv) > 1 else 'bc'
path = DOCS[which]
pages = extract_pages(path)
units = extract_response_units(pages)
for unit in units:
    labels = unit.get('recommendation_labels') or [unit.get('recommendation_label')]
    rt = (unit.get('response_text') or '').strip()
    lbl, conf = classify_with_confidence(rt)
    head = rt[:90].replace("\n", " ")
    print(f"{','.join(l for l in labels if l):8s}  {lbl:20s}  conf={conf:.2f}  {head}...")
