"""Utility to convert Chinese disease names to English equivalents."""
from __future__ import annotations

import re
from typing import Dict

_BASE_MAP: Dict[str, str] = {
    "鼠疫": "Plague",
    "霍乱": "Cholera",
    "狂犬病": "Rabies",
    "人感染高致病性禽流感": "Highly Pathogenic Avian Influenza",
    "传染性非典型肺炎": "SARS",
    "艾滋病": "AIDS",
    "病毒性肝炎": "Viral Hepatitis",
    "乙型肝炎": "Hepatitis B",
    "丙型肝炎": "Hepatitis C",
    "丁型肝炎": "Hepatitis D",
    "戊型肝炎": "Hepatitis E",
    "未分型肝炎": "Unspecified Hepatitis",
    "肺结核": "Tuberculosis",
    "流行性出血热": "Hemorrhagic Fever",
    "人感染禽流感": "Avian Influenza",
    "脊髓灰质炎": "Poliomyelitis",
    "白喉": "Diphtheria",
    "新生儿破伤风": "Neonatal Tetanus",
    "炭疽": "Anthrax",
    "流行性脑脊髓膜炎": "Meningococcal Meningitis",
    "登革热": "Dengue Fever",
    "麻疹": "Measles",
    "百日咳": "Pertussis",
    "梅毒": "Syphilis",
    "伤寒和副伤寒": "Typhoid & Paratyphoid",
    "细菌性和阿米巴性痢疾": "Bacillary/Amoebic Dysentery",
    "流行性乙型脑炎": "Japanese Encephalitis",
    "钩端螺旋体病": "Leptospirosis",
    "血吸虫病": "Schistosomiasis",
    "布鲁氏菌病": "Brucellosis",
    "淋病": "Gonorrhea",
    "猴痘": "Monkeypox",
    "黑热病": "Visceral Leishmaniasis",
    "丝虫病": "Lymphatic Filariasis",
    "包虫病": "Echinococcosis",
    "流行性感冒": "Influenza",
    "手足口病": "Hand-Foot-Mouth Disease",
    # "其他感染性腹泻病": "Other Infectious Diarrheal Diseases",
    "其他感染性腹泻病": "Infectious Diarrheal",
    "急性出血性结膜炎": "Acute Hemorrhagic Conjunctivitis",
    "猩红热": "Scarlet Fever",
    "流行性腮腺炎": "Mumps",
    "风疹": "Rubella",
    "麻风病": "Leprosy",
    "疟疾": "Malaria",
    "斑疹伤寒": "Typhus",
    "新型冠状病毒肺炎": "COVID-19",
    "新型冠状病毒感染": "COVID-19",
    "阿米巴痢疾": "Amoebic Dysentery",
}


def translate(name: str) -> str:
    """Return English name for a Chinese disease label."""

    cleaned = re.sub(r"[^\u4e00-\u9fffA-Za-z]", "", name)
    return _BASE_MAP.get(cleaned, name)
