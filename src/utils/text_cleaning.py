import re

def clean_text(text):
    # Удаление HTML-тегов вроде <br />
    text = re.sub(r"<.*?>", " ", text)

    # Удаление спецсимволов типа &amp;
    text = re.sub(r"&[a-z]+;", " ", text)

    # Удаление лишних пробелов
    text = re.sub(r"\s+", " ", text)

    # Обрезаем пробелы по краям
    return text.strip()
