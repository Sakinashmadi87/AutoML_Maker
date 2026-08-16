import re

def get_heuristic_caption(latex_str):
    """
    Übersetzt LaTeX-Symbole in semantische Wörter für den Embedder.
    """
    translations = {
        # Operatoren & Kalkül
        r'\int': ' integral ', r'\sum': ' summation ', r'\nabla': ' gradient ',
        r'\partial': ' partial derivative ', r'\Delta': ' laplacian ',
        
        # Logik & Mengen
        r'\forall': ' for all ', r'\exists': ' exists ', r'\in': ' element of ',
        r'\subset': ' subset of ',
        
        # Griechische Buchstaben
        r'\alpha': ' alpha ', r'\beta': ' beta ', r'\gamma': ' gamma ',
        r'\theta': ' theta ', r'\mu': ' mu ', r'\sigma': ' sigma ',
        r'\omega': ' omega ', r'\pi': ' pi ', r'\lambda': ' lambda ',
        
        # Relationen & Funktionen
        r'\approx': ' approximately equal ', r'\neq': ' not equal ',
        r'\le': ' less than or equal ', r'\ge': ' greater than or equal ',
        r'\infty': ' infinity ', r'\exp': ' exponential ', r'\log': ' logarithm '
    }
    
    caption = latex_str.strip()
    for cmd, word in translations.items():
        caption = caption.replace(cmd, word)
    
    # Entferne LaTeX-Strukturzeichen ($, \, {, }, ^, _)
    caption = re.sub(r'[\$\{\}\^_\\]', ' ', caption)
    return f"[MATH: {re.sub(r'\s+', ' ', caption).strip()}]"

def clean_scientific_markdown(text, replace_math=True):
    """
    Bereinigt Markdown-Text von wissenschaftlichem Rauschen (Referenzen, URLs, LaTeX).
    """
    if not text:
        return ""

    # 1. Referenzen entfernen (alles nach dem Literaturverzeichnis)
    text = re.split(r'\n#+ (?:References|Bibliography)', text, flags=re.IGNORECASE)[0]
    
    # 2. URLs und arXiv-Wasserzeichen entfernen
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'arXiv:\d{4}\.\d{4,5}(?:v\d+)?\s\[.*\]', '', text)
    
    # 3. Mathematisches Rauschen behandeln
    if replace_math:
        # Ersetze alle mathematischen Ausdrücke ($...$ oder $$...$$) durch die Heuristik
        text = re.sub(r'\$\$?.*?\$\$?', lambda m: get_heuristic_caption(m.group(0)), text)

    # 4. Cleanup: Mehrfache Leerzeichen und Zeilenumbrüche glätten
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()
