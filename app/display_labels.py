from __future__ import annotations

from typing import Any


SIMULATION_INTERPRETATION_NOTE = (
    "Hasil ini merupakan interpretasi berbasis simulasi musculoskeletal, bukan diagnosis klinis."
)

LABEL_DISPLAY_MAP = {
    "within_reference": "Sesuai Pola Acuan",
    "deviating": "Pola Aktivitas Berbeda",
    "prob_deviating": "Probabilitas Pola Berbeda",
    "probability_deviating": "Probabilitas Pola Berbeda",
    "deviation_score": "Skor Perbedaan Pola",
    "average_deviation_score": "Skor Perbedaan Pola",
    "relational_risk": "Skor Relasi Antarotot",
    "affected_muscle": "Otot Terkait Secara Relasional",
    "affected_muscles": "Otot Terkait Secara Relasional",
    "sesuai_acuan": "Sesuai Pola Acuan",
    "terindikasi_menyimpang": "Pola Aktivitas Berbeda",
    "berpotensi_terdampak": "Otot Terkait Secara Relasional",
    "normal": "Data Acuan",
    "abnormal": "Data Uji Pola Berbeda",
    "testing": "Data Testing",
}

TRIAL_LABEL_DISPLAY_TO_INTERNAL = {
    "Data Acuan": "normal",
    "Data Uji Pola Berbeda": "abnormal",
    "Data Testing": "testing",
    "normal": "normal",
    "abnormal": "abnormal",
    "testing": "testing",
}

TRIAL_LABEL_DISPLAY_VALUES = ("Data Acuan", "Data Uji Pola Berbeda", "Data Testing")


def display_label(value: Any, *, default: str = "-") -> str:
    key = str(value or "").strip()
    if not key:
        return str(default)
    return LABEL_DISPLAY_MAP.get(key, key)


def display_status(value: Any, *, default: str = "-") -> str:
    return display_label(value, default=default)


def internal_trial_label(value: Any) -> str:
    key = str(value or "").strip()
    return TRIAL_LABEL_DISPLAY_TO_INTERNAL.get(key, "normal")


def display_trial_label(value: Any) -> str:
    return display_label(internal_trial_label(value))


def status_description(value: Any) -> str:
    key = str(value or "").strip()
    if key == "terindikasi_menyimpang" or key == "deviating":
        return "Otot dengan pola aktivitas berbeda terhadap acuan simulasi."
    if key == "berpotensi_terdampak" or key == "affected_muscle":
        return "Otot terkait secara relasional dengan otot yang memiliki pola aktivitas berbeda."
    if key == "sesuai_acuan" or key == "within_reference":
        return "Otot berada dalam pola aktivitas acuan simulasi."
    return display_label(key)
