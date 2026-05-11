"""Final classification audit across all 8 response documents."""
from collections import Counter
from src.pdf_loader import extract_pages
from src.response_units import extract_response_units
from src.classification import classify_response

DOCS = [
    ('behaviour_change', 'data/raw/Behaviour-Change-Response.pdf'),
    ('post_office',      'data/raw/PostOfficeHorizon-IT-Inquiry-Response.pdf'),
    ('space_economy',    'data/raw/TheSpaceEconomyResponse.pdf'),
    ('covid_1',          'data/raw/UK-Covid-19_Inquiry_Module_1_Response.pdf'),
    ('blood',            'data/raw/Volume_1-Blood-Inquiry-Response.pdf'),
    ('grenfell',         'data/raw/Grenfell-Phase2-Response.pdf'),
    ('covid_2',          'data/raw/UK-Covid-19_Inquiry_Module_2_Response.pdf'),
    ('summer_2024',      'data/raw/Summer2024-Disorder-Response.pdf'),
]
totals = Counter()
for name, path in DOCS:
    pages = extract_pages(path)
    units = extract_response_units(pages)
    counts = Counter()
    for u in units:
        rt = (u.get('response_text') or '').strip()
        lbl = classify_response(rt) if rt else 'not_addressed'
        labels = u.get('recommendation_labels') or [u.get('recommendation_label')]
        for L in labels:
            if L:
                counts[lbl] += 1
                totals[lbl] += 1
    used_chunks = "STRUCTURED" if len(units) >= 2 else "chunk_fallback"
    print(f"{name:20s}  units={len(units):3d}  path={used_chunks:18s}  {dict(counts)}")
print(f"{'TOTAL (structured only)':<20s}  {dict(totals)}")
