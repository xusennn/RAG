# src/text_processor.py

from typing import List, Dict

def build_embedding_text(data: List[Dict]) -> List[str]:
    texts = []
    for item in data:
        parts = []
        if item.get("attraction"):
            parts.append(item["attraction"])
        if item.get("country"):
            parts.append(f"in {item['country']}.")
        if item.get("best_time_to_visit"):
            parts.append(f"Best time to visit: {item['best_time_to_visit']}.")
        if item.get("tourism_type"):
            parts.append(f"Tourism types: {', '.join(item['tourism_type'])}.")
        if item.get("highlight"):
            parts.append(f"Highlights: {item['highlight']}")
        full_text = " ".join(parts)
        texts.append(full_text)
    return texts
