"""One-off probe of response unit extraction across all 8 documents."""
from src.pdf_loader import extract_pages
from src.response_units import extract_response_units

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

for name, path in DOCS:
    try:
        pages = extract_pages(path)
        units = extract_response_units(pages)
        flat = []
        for u in units:
            labels = u.get('recommendation_labels') or [u.get('recommendation_label')]
            for l in labels:
                if l:
                    flat.append(l)
        print(f"{name:20s}  units={len(units):3d}  unique_labels={len(set(flat)):3d}  sample={flat[:15]}")
    except Exception as e:
        print(f"{name:20s}  ERROR: {e}")
