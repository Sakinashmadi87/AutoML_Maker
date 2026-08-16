import re

def get_heuristic_caption(latex_str):
    # Mapping von LaTeX-Befehlen zu semantischen Begriffen
    translations = {
        # Operatoren & Kalkül
        r'\int': ' integral ',
        r'\sum': ' summation ',
        r'\prod': ' product ',
        r'\nabla': ' gradient ',
        r'\partial': ' partial derivative ',
        r'\delta': ' delta ',
        r'\Delta': ' laplacian ',
        r'\diff': ' differential ',
        
        # Logik & Mengen
        r'\forall': ' for all ',
        r'\exists': ' exists ',
        r'\in': ' element of ',
        r'\subset': ' subset of ',
        r'\cap': ' intersection ',
        r'\cup': ' union ',
        
        # Griechische Buchstaben (häufige Variablen)
        r'\alpha': ' alpha ',
        r'\beta': ' beta ',
        r'\gamma': ' gamma ',
        r'\theta': ' theta ',
        r'\mu': ' mu ',
        r'\sigma': ' sigma ',
        r'\omega': ' omega ',
        r'\pi': ' pi ',
        r'\lambda': ' lambda ',
        r'\epsilon': ' epsilon ',
        r'\tau': ' tau ',
        
        # Relationen & Funktionen
        r'\approx': ' approximately equal ',
        r'\neq': ' not equal ',
        r'\le': ' less than or equal ',
        r'\ge': ' greater than or equal ',
        r'\rightarrow': ' approaches ',
        r'\infty': ' infinity ',
        r'\exp': ' exponential ',
        r'\log': ' logarithm ',
        r'\sin': ' sine ',
        r'\cos': ' cosine ',
        
        # Strukturen
        r'\frac': ' ratio ',
        r'\sqrt': ' square root ',
        r'\mathbf': ' vector ',
        r'\mathcal': ' set '
    }
    
    # 1. Bereinigung: LaTeX-Strings säubern
    caption = latex_str.strip()
    
    # 2. Übersetzen der Symbole
    for cmd, word in translations.items():
        # Nutze Regex für Wortgrenzen, um Teil-Matches zu vermeiden (z.B. \sin vs \sinh)
        caption = caption.replace(cmd, word)
    
    # 3. Mathematische Strukturzeichen entfernen
    # Entfernt $, \, {, }, ^, _, [, ]
    caption = re.sub(r'[\$\{\}\^_\\]', ' ', caption)
    
    # 4. Mehrfache Leerzeichen glätten
    caption = re.sub(r'\s+', ' ', caption).strip()
    
    return f"[MATH: {caption}]"

# Beispiel-Test:
# Input: "$\int_0^\infty e^{-x^2} dx$"
# Output: "[MATH: integral 0 infinity e -x 2 dx]"
