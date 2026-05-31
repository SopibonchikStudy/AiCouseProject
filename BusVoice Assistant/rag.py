"""
RAG через текстовый поиск по ключевым словам с приоритетом заголовков.
"""
import os
import re

_manual = None
_chunks = None
_headers = None

def load_manual():
    """Загружает инструкцию и разбивает на чанки."""
    global _manual, _chunks, _headers
    
    if _chunks is not None:
        return _chunks
    
    with open("data/manual.txt", "r", encoding="utf-8") as f:
        _manual = f.read()
    
    # Разбиваем по заголовкам (строки, заканчивающиеся на :)
    raw_chunks = re.split(r'\n(?=[А-ЯЁ][А-ЯЁ\s]+:)', _manual)
    
    _chunks = []
    _headers = []
    
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        
        # Извлекаем заголовок (первая строка до :)
        header_match = re.match(r'^([А-ЯЁ][А-ЯЁ\s]+):', chunk)
        if header_match:
            _headers.append(header_match.group(1).lower())
        else:
            _headers.append("")
        
        _chunks.append(chunk)
    
    return _chunks


def _clean_text(text: str) -> str:
    """Очистка текста от знаков препинания."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text


def query_manual(query: str) -> str:
    """
    Ищет релевантный раздел инструкции.
    Приоритет: совпадение с заголовком > ключевые слова в тексте.
    """
    chunks = load_manual()
    query_clean = _clean_text(query)
    query_words = query_clean.split()
    
    if not query_words:
        return chunks[-1] if chunks else "Инструкция не найдена."
    
    # Ключевые слова-маркеры для систем автобуса
    system_keywords = {
        "тормоз": "ТОРМОЗНАЯ СИСТЕМА",
        "тормоза": "ТОРМОЗНАЯ СИСТЕМА",
        "тормозной": "ТОРМОЗНАЯ СИСТЕМА",
        "двигатель": "ДВИГАТЕЛЬ",
        "двигателя": "ДВИГАТЕЛЬ",
        "мотор": "ДВИГАТЕЛЬ",
        "перегрев": "ДВИГАТЕЛЬ",
        "стук": "ДВИГАТЕЛЬ",
        "фар": "ОСВЕЩЕНИЕ",
        "фары": "ОСВЕЩЕНИЕ",
        "освещение": "ОСВЕЩЕНИЕ",
        "свет": "ОСВЕЩЕНИЕ",
        "двер": "ДВЕРИ",
        "двери": "ДВЕРИ",
        "дверь": "ДВЕРИ",
        "пневм": "ПНЕВМОСИСТЕМА",
        "давление": "ПНЕВМОСИСТЕМА",
    }
    
    # Шаг 1: Ищем точное совпадение с заголовком по маркерам
    target_header = None
    for word in query_words:
        if word in system_keywords:
            target_header = system_keywords[word]
            break
    
    # Шаг 2: Если нашли заголовок — возвращаем соответствующий чанк
    if target_header:
        for i, chunk in enumerate(chunks):
            if chunk.startswith(target_header):
                return chunk
    
    # Шаг 3: Иначе — поиск по ключевым словам с приоритетом заголовка
    scored_chunks = []
    for i, chunk in enumerate(chunks):
        score = 0
        chunk_lower = chunk.lower()
        header = _headers[i] if i < len(_headers) else ""
        
        # Бонус за совпадение с заголовком
        for word in query_words:
            if word in header:
                score += 10  # Большой бонус
                break
        
        # Очки за совпадения в тексте
        for word in query_words:
            if word in chunk_lower:
                score += 1
        
        if score > 0:
            scored_chunks.append((chunk, score))
    
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    if scored_chunks:
        return scored_chunks[0][0]
    
    # Общее правило
    for chunk in chunks:
        if "ОБЩЕЕ ПРАВИЛО" in chunk:
            return chunk
    
    return "Следуйте общим правилам безопасности."


if __name__ == "__main__":
    tests = [
        "отказали тормоза",
        "дверь заклинило",
        "двигатель перегрелся",
        "фары не работают",
        "давление упало",
    ]
    for t in tests:
        print(f"\n{'='*50}")
        print(f"Запрос: {t}")
        print(f"Найдено:\n{query_manual(t)[:200]}")