"""Print snippets of each problem-doc response to see heading formats."""
import re
from src.pdf_loader import extract_pages
from src.response_units import _INLINE_SPLIT, _RESPONSE_OPENER

DOCS = [
    ('covid_1',     'data/raw/UK-Covid-19_Inquiry_Module_1_Response.pdf'),
    ('blood',       'data/raw/Volume_1-Blood-Inquiry-Response.pdf'),
    ('grenfell',    'data/raw/Grenfell-Phase2-Response.pdf'),
    ('covid_2',     'data/raw/UK-Covid-19_Inquiry_Module_2_Response.pdf'),
    ('summer_2024', 'data/raw/Summer2024-Disorder-Response.pdf'),
]
for name, path in DOCS:
    pages = extract_pages(path)
    all_text = "\n".join((p.get("text") or "") for p in pages)
    print("="*70)
    print(name, "total chars:", len(all_text), "pages:", len(pages))
    # Search for any "Recommendation N" patterns
    matches = list(re.finditer(r"\b(?:recommendation|response)s?\b[^\n]{0,80}", all_text, re.IGNORECASE))
    print(f"  'recommendation/response' tokens: {len(matches)}")
    for m in matches[:8]:
        snippet = m.group(0).strip()[:90]
        print(f"     • {snippet}")
    # Check inline-split pattern matches
    splits = list(_INLINE_SPLIT.finditer(all_text))
    print(f"  _INLINE_SPLIT matches: {len(splits)}")
    for s in splits[:5]:
        ctx = all_text[max(0,s.start()-30):s.end()+40].replace("\n"," ")
        print(f"     • [{s.group('label')}] ...{ctx}...")
    # Look for numbered recommendation patterns "Recommendation 1", "Recommendation 1:" etc
    numbered = list(re.finditer(r"\bRecommendation\s+\d+[a-z]?[:.]?", all_text))
    print(f"  'Recommendation N' (literal) hits: {len(numbered)}")
    for n in numbered[:5]:
        print(f"     • {n.group(0)}  @char {n.start()}")
    print()
