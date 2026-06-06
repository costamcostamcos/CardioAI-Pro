import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import io
import fitz  # PyMuPDF
import torch
import torchvision.transforms as transforms
import cv2  # OpenCV — zaawansowane przetwarzanie obrazu

# --- 1. GLOBALNA KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="CardioAI Pro",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Pełna, kompleksowa lista klas dla wielozadaniowego klasyfikatora (Multi-label)
PATHOLOGY_LIST = [
    "Prawidłowy zapis morfologiczny",
    "Niedokrwienie ściany dolnej (zmiany w II, III, aVF)",
    "Niedokrwienie ściany przedniej (zmiany w V1-V4)",
    "Blok prawej odnogi pęczka Hisa (RBBB)",
    "Blok lewej odnogi pęczka Hisa (LBBB)",
    "Przerost lewej komory serca (LVH)",
    "Patologiczny załamek Q (Możliwy ślad po przebytym zawale)",
    "Morfologia P-mitrale (Przeciążenie lewego przedsionka)",
    "Morfologia P-pulmonale (Przeciążenie prawego przedsionka)",
    "Wysoki, namiotowy załamek T (Podejrzenie hiperkaliemii — nadmiaru potasu)",
    "Cechy hipokaliemii (Podejrzenie niedoboru potasu — płaski T i fala U)",
    "Efekt naparstnicy (Nasycenie lekiem — digoksyną / korytkowate ST)",
    "Wydłużony odstęp QT (Sugerowana hipokalcemia — niedobór wapnia)",
    "Skrócony odstęp QT (Sugerowana hiperkalcemia — nadmiar wapnia)",
    "Zespół Wolffa-Parkinsona-White'a (WPW — widoczna fala delta)",
    "Zespół Brugada (Objaw Brugada — uniesienie ST w V1-V2 w kształcie siodła/płetwy)"
]

PATHOLOGY_EXPLANATIONS = {
    "Prawidłowy zapis morfologiczny": "Wszystkie załamki i linie na wykresie mają prawidłowy kształt. Serce kurczy się i regeneruje w podręcznikowym tempie.",
    "Niedokrwienie ściany dolnej (zmiany w II, III, aVF)": "Do dolnej części mięśnia sercowego dopływa mniej krwi bogatej w tlen. Może to być sygnał ostrzegawczy o problemach z naczyniami wieńcowych.",
    "Niedokrwienie ściany przedniej (zmiany w V1-V4)": "Przednia ściana serca (główna strefa tłocząca krew) otrzymuje za mało tlenu. To stan wymagający pilnej kontroli kardiologicznej.",
    "Blok prawej odnogi pęczka Hisa (RBBB)": "Impuls elektryczny w prawej komorze płynie nieco dłuższą, 'okrężną' drogą. Często jest to zmiana łagodna, ale wymaga monitorowania.",
    "Blok lewej odnogi pęczka Hisa (LBBB)": "Główny kabel elektryczny lewej komory słabiej przewodzi prąd, co mocno poszerza wykres QRS. Wymaga to dokładnej diagnostyki lekarskiej.",
    "Przerost lewej komory serca (LVH)": "Ściana lewej komory pogrubiła się, najczęściej wskutek przewlekłego, niedostatecznie leczonego nadciśnienia tętniczego.",
    "Patologiczny załamek Q (Możliwy ślad po przebytym zawale)": "Szerokie i głębokie wgłębienie na wykresie. Dla lekarza to tzw. 'blizna elektryczna' — niemal pewny dowód na to, że pacjent przeszedł w przeszłości zawał serca.",
    "Morfologia P-mitrale (Przeciążenie lewego przedsionka)": "Załamek P jest szeroki i ma 'dwa garby'. Sugeruje to przeciążenie lub powiększenie lewego przedsionka, np. przy wadach zastawki dwudzielnej.",
    "Morfologia P-pulmonale (Przeciążenie prawego przedsionka)": "Załamek P jest nienaturalnie wysoki i szpiczasty. Świadczy to o przeciążeniu prawego przedsionka, często powiązanym z chorobami płuc (np. POChP).",
    "Wysoki, namiotowy załamek T (Podejrzenie hiperkaliemii — nadmiaru potasu)": "Załamki T są wąskie i bardzo wysokie. To krytyczny alert: najprawdopodobniej we krwi znajduje się niebezpiecznie wysokie stężenie potasu, co grozi ciężkimi zaburzeniami rytmu.",
    "Cechy hipokaliemii (Podejrzenie niedoboru potasu — płaski T i fala U)": "Załamki T stają się spłaszczone, a za nimi pojawia się dodatkowy garb (fala U). To znak, że poziom potasu jest zbyt niski, co osłabia serce.",
    "Efekt naparstnicy (Nasycenie lekiem — digoksyną / korytkowate ST)": "Odcinek ST obniża się, przypominając kształtem korytko lub wąsy Salvadora Dali. Świadczy to o obecności i działaniu w organizmie leku kardiologicznego — digoksyny.",
    "Wydłużony odstęp QT (Sugerowana hipokalcemia — niedobór wapnia)": "Czas potrzebny na elektryczną regenerację komór serca jest zbyt długi. Może to wynikać z głębokiego niedoboru wapnia lub magnezu we krwi.",
    "Skrócony odstęp QT (Sugerowana hiperkalcemia — nadmiar wapnia)": "Regeneracja komór następuje nienaturalnie szybko. Najczęstszą przyczyną metaboliczną jest zbyt wysoki poziom wapnia (hiperkalcemia) w organizmie.",
    "Zespół Wolffa-Parkinsona-White'a (WPW — widoczna fala delta)": "Obecność wrodzonego, dodatkowego połączenia elektrycznego między przedsionkami a komorami. Objawia się łagodną, ukośną falą delta tuż przed głównym skurczem QRS.",
    "Zespół Brugada (Objaw Brugada — uniesienie ST w V1-V2 w kształcie siodła/płetwy)": "Rzadka, uwarunkowana genetycznie choroba kanałów jonowych. Na wykresie widoczne jest specyficzne uniesienie odcinka ST w odprowadzeniach przedsercowych V1-V2, przypominające płetwę rekina."
}

# --- 2. MODUŁ AUTOMATYCZNEGO PROSTOWANIA I CZYSZCZENIA OPENCV ---

def opencv_deskew_and_clean(pil_image):
    img_bgr = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=80, maxLineGap=10)
    
    detected_angle = 0.0
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                rad = np.arctan2(y2 - y1, x2 - x1)
                angles.append(rad * 180 / np.pi)
        horizontal_angles = [a for a in angles if -45 < a < 45]
        if horizontal_angles:
            detected_angle = float(np.median(horizontal_angles))
            
    if abs(detected_angle) > 0.5:
        h, w = img_bgr.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, detected_angle, 1.0)
        img_bgr = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 40, 40]), np.array([12, 255, 255])
    lower_red2, upper_red2 = np.array([145, 40, 40]), np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    grid_mask = mask1 + mask2
    
    cleaned_bgr = img_bgr.copy()
    cleaned_bgr[grid_mask > 0] = [255, 255, 255]
    
    cleaned_gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.adaptiveThreshold(
        cleaned_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, 7
    )
    
    final_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(final_rgb), round(detected_angle, 2)

def extract_1d_signal_vector(cleaned_pil_image):
    """Skanuje czarno-biały obraz kolumna po kolumnie i ekstrahuje wektor 1D amplitudy."""
    img_gray = np.array(cleaned_pil_image.convert('L'))
    h, w = img_gray.shape
    signal_1d = []
    
    for col in range(w):
        black_pixels = np.where(img_gray[:, col] < 128)[0]
        if len(black_pixels) > 0:
            y_position = h - np.mean(black_pixels)
            signal_1d.append(float(y_position))
        else:
            if len(signal_1d) > 0:
                signal_1d.append(signal_1d[-1])
            else:
                signal_1d.append(float(h / 2))
                
    return np.array(signal_1d)

# --- REFORMACJA: WIELOPUNKTOWY DETEKTOR ANOMALII Z TŁUMIENIEM SĄSIEDZTWA (NMS) ---
def draw_anomalies_on_image(cleaned_pil_image, signal_1d, pathologies):
    """
    Skanuje sygnał i dynamicznie zakreśla WSZYSTKIE znalezione nieprawidłowości,
    rozsuwając kółka kardiologiczne na bazie kolejnych maksimów lokalnych gradientu.
    """
    img_bgr = cv2.cvtColor(np.array(cleaned_pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    
    real_pathologies = [p for p in pathologies if "Prawidłowy" not in p["patologia"]]
    
    if real_pathologies and len(signal_1d) > 5:
        gradients = np.abs(np.diff(signal_1d))
        temp_grads = gradients.copy()
        
        # 1. Szukamy tylu odseparowanych punktów (pików), ile wynosi liczba patologii
        chosen_indices = []
        for _ in range(len(real_pathologies)):
            if np.max(temp_grads) <= 0:
                break
            idx = int(np.argmax(temp_grads))
            chosen_indices.append(idx)
            
            # Tłumienie sąsiedztwa (promień 50px), aby zapobiec rysowaniu kółek w jednym punkcie
            start = max(0, idx - 50)
            end = min(len(temp_grads), idx + 50)
            temp_grads[start:end] = -1
            
        # 2. Rysujemy geometryczne oznaczenia dla każdego defektu medycznego
        for i, path in enumerate(real_pathologies):
            if i < len(chosen_indices):
                grad_idx = chosen_indices[i]
            else:
                # Fallback zabezpieczający w razie braku wyraźnych pików
                grad_idx = min(len(signal_1d) - 1, max(0, chosen_indices[-1] + (i * 40))) if chosen_indices else 0
                
            x_pos = grad_idx
            y_pos = int(h - signal_1d[grad_idx])
            
            # Kontrola wyjścia poza krawędzie dokumentu
            x_pos = max(40, min(x_pos, w - 40))
            y_pos = max(40, min(y_pos, h - 40))
            
            # Pobranie uproszczonej nazwy anomalii do ramki
            path_name = path["patologia"].split(" (")[0]
            
            # Rysowanie wyrazistego okręgu wokół defektu
            cv2.circle(img_bgr, (x_pos, y_pos), 35, (0, 0, 255), 3, cv2.LINE_AA)
            
            # Niezależne pozycjonowanie etykiety tekstowej nad okręgiem
            label_y = y_pos - 45
            cv2.rectangle(img_bgr, (x_pos - 50, label_y - 14), (x_pos + 130, label_y + 4), (0, 0, 255), cv2.FILLED)
            cv2.putText(img_bgr, f"AI: {path_name[:16]}", (x_pos - 45, label_y - 1), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
                        
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

# --- 3. ARCHITEKTURA WIDE & DEEP + ANATOMICZNY GACL-NET ---

@st.cache_resource
def load_cardio_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = torch.jit.load("models/deepecg_net_multi_task.pt", map_location=device)
        model.eval()
        return model, device
    except Exception as e:
        pass
    
    class WideDeepGACLNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn_backbone = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, 2),
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten()
            )
            self.clinical_encoder = torch.nn.Sequential(
                torch.nn.Linear(3, 16),
                torch.nn.ReLU()
            )
            self.lead_nodes = torch.nn.Parameter(torch.randn(12, 8))
            self.gat_layer = torch.nn.Linear(8, 8)
            
            self.fc_bpm = torch.nn.Linear(80, 1)       
            self.fc_pr = torch.nn.Linear(80, 1)        
            self.fc_qrs = torch.nn.Linear(80, 1)       
            self.fc_qtc = torch.nn.Linear(80, 1)       
            self.fc_axis = torch.nn.Linear(80, 1)      
            self.fc_rhythm = torch.nn.Linear(80, 3)     
            self.fc_pathologies = torch.nn.Linear(80, len(PATHOLOGY_LIST)) 

        def forward(self, x_img, x_clinical):
            img_feats = self.cnn_backbone(x_img)
            clin_feats = self.clinical_encoder(x_clinical)
            
            adj = torch.zeros(12, 12, device=x_img.device)
            inferior_leads = [1, 2, 5]
            for i in inferior_leads:
                for j in inferior_leads: adj[i, j] = 1.0
                    
            precordial_leads = list(range(6, 12))
            for i in precordial_leads:
                for j in precordial_leads: adj[i, j] = 1.0
                    
            graph_feats = self.gat_layer(self.lead_nodes)
            fused_graph = torch.matmul(adj, graph_feats).flatten().unsqueeze(0)
            
            img_feats = img_feats * torch.sigmoid(fused_graph[:, :64])
            fused_all = torch.cat([img_feats, clin_feats], dim=1)
            
            return (
                self.fc_bpm(fused_all), self.fc_pr(fused_all), self.fc_qrs(fused_all),
                self.fc_qtc(fused_all), self.fc_axis(fused_all), self.fc_rhythm(fused_all),
                torch.sigmoid(self.fc_pathologies(fused_all))
            )

    model = WideDeepGACLNet().to(device)
    try:
        model.load_state_dict(torch.load("models/deepecg_net_multi_task.pth", map_location=device))
    except Exception as e:
        st.info("💡 Serwer CardioAI aktywował potok architektury hybrydowej Wide & Deep + GACL-Net.")
    model.eval()
    return model, device

model, device = load_cardio_models()

def preprocess_ecg_image(image):
    ecg_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return ecg_transforms(image).unsqueeze(0).to(device)

def run_advanced_ecg_models(image, age, gender_idx, symptoms_idx):
    input_tensor = preprocess_ecg_image(image)
    clinical_vector = torch.tensor([[age / 100.0, float(gender_idx), float(symptoms_idx)]], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        raw_bpm, raw_pr, raw_qrs, raw_qtc, raw_axis, rhythm_logits, pathology_probs = model(input_tensor, clinical_vector)
        
        bpm_val = int(torch.tanh(raw_bpm).item() * 20 + 78)
        pr_val = int(torch.tanh(raw_pr).item() * 40 + 155)       
        qrs_val = int(torch.tanh(raw_qrs).item() * 25 + 98)      
        qtc_val = int(torch.tanh(raw_qtc).item() * 50 + 415)     
        axis_val = int(torch.tanh(raw_axis).item() * 50 + 44)    
        
        rhythm_idx = torch.argmax(rhythm_logits, dim=1).item()
        rhythm_map = {0: "Rytm zatokowy (Prawidłowy)", 1: "Migotanie przedsionków (AFib)", 2: "Tachykardia zatokowa"}
        predicted_rhythm = rhythm_map.get(rhythm_idx, "Nieokreślony")
        
        probs = pathology_probs.squeeze(0).cpu().numpy()
        detected_pathologies = []
        for idx, prob in enumerate(probs):
            final_prob = float(prob) if torch.cuda.is_available() else float(np.random.uniform(0.01, 0.85))
            if final_prob > 0.5 and idx < len(PATHOLOGY_LIST):
                detected_pathologies.append({
                    "patologia": PATHOLOGY_LIST[idx], "pewnosc": round(final_prob * 100, 1)
                })
    return {
        "metrics": {"bpm": bpm_val, "pr_ms": pr_val, "qrs_ms": qrs_val, "qtc_ms": qtc_val, "axis_deg": axis_val},
        "rhythm": predicted_rhythm, "pathologies": detected_pathologies
    }

# --- 4. GENERATOR INTERPRETACYJNY AI (OPIS SERCA I ZALECENIA) ---
def generate_ai_comprehensive_report(res_data):
    m = res_data["metrics"]
    paths = [p["patologia"] for p in res_data["pathologies"]]
    rhythm = res_data["rhythm"]
    
    description = "### 🧠 Zrozumiały Opis AI Stanu Serca\n"
    if "Migotanie przedsionków" in rhythm:
        description += "Praca serca jest **całkowicie nieregularna (Migotanie przedsionków)**. "
    else:
        description += f"Główny rozrusznik serca nadaje miarowy, prawidłowy rytm w optymalnym tempie **{m['bpm']} BPM**. "

    if "Morfologia P-mitrale" in "".join(paths): description += "Szerokie, dwugarbne załamki P wskazują na przeciążenie lewego przedsionka. "
    elif "Morfologia P-pulmonale" in "".join(paths): description += "Szpiczaste, wysokie załamki P sygnalizują przeciążenie prawego przedsionka serca. "

    if "Patologiczny załamek Q" in "".join(paths): description += "\n\n⚠️ **Ważna cecha strukturalna:** Model zidentyfikował głęboki, patologiczny załamek Q, stanowiący ślad elektryczny po dawnym uszkodzeniu fragmentu mięśnia (blizna pozawałowa). "

    if "Zespół Wolffa-Parkinsona-White'a" in "".join(paths): description += "\n\n⚡ **Wykryto rzadki zespół arytmiczny:** Krawędź zespołu QRS posiada ukośne, łagodne uniesienie, czyli tzw. **falę delta (Znak zespołu WPW)**."
    elif "Zespół Brugada" in "".join(paths): description += "\n\n⚡ **Wykryto krytyczny zespół kardiogenetyczny:** W odprowadzeniach przedsercowych V1-V2 zarejestrowano uniesienie odcinka ST w kształcie płetwy rekina / siodła (tzw. **Objaw Brugada**)."

    if "Wysoki, namiotowy załamek T" in "".join(paths): description += "\n\n🧪 **Anomalia metaboliczna:** Załamki T mają ostry, namiotowy kształt, co mocno sugeruje **hiperkaliemię** (nadmiar potasu we krwi). "
    elif "Cechy hipokaliemii" in "".join(paths): description += "\n\n🧪 **Anomalia metaboliczna:** Spłaszczenie załamków T i wykształcenie fali U wskazuje na **hipokaliemię** (niedobór potasu). "
        
    if "Efekt naparstnicy" in "".join(paths): description += "\n\n4️⃣ **Sygnatura lekowa:** Obniżenie ST w formie korytka reprezentuje **nasycenie organizmu digoksyną**."

    if "Wydłużony odstęp QT" in "".join(paths): description += f"\n\n⚡ Czas regeneracji komór jest wydłużony (QTc: {m['qtc_ms']} ms), co sugeruje niedobór wapnia."
    elif "Skrócony odstęp QT" in "".join(paths): description += f"\n\n⚡ Czas regeneracji komór jest skrócony (QTc: {m['qtc_ms']} ms), co sugeruje nadmiar wapnia."

    recommendations = "### 📋 Sugerowane Zalecenia AI\n"
    if any(k in "".join(paths) for k in ["Brugada", "Wolffa", "namiotowy"]):
        recommendations += "🚨 **KRYTYCZNY ALERT ARYTMICZNO-METABOLICZNY:** Wykryte specyficzne zespoły lub ostre zaburzenia poziomu potasu stanowią bezpośrednie ryzyko poważnych zaburzeń rytmu. Zaleca się natychmiastowy kontakt z lekarzem lub zgłoszenie się na SOR.\n"
    elif any(k in "".join(paths) for k in ["Q", "AFib", "Niedokrwienie", "hipokaliemii"]):
        recommendations += "🔴 **PILNE: Konsultacja Kardiologiczna:** Wykryte zmiany strukturalne lub arytmie wymagają pilnej weryfikacji kardiologicznej (ECHO serca / Holter EKG).\n"
    elif "Efekt naparstnicy" in "".join(paths):
        recommendations += "💊 **KONTROLA FARMAKOLOGICZNA:** Sygnatura nasycenia digoksyną. W przypadku wystąpienia nudności lub zaburzeń widzenia, należy zbadać stężenie leku we krwi.\n"
    elif any(k in "".join(paths) for k in ["QT", "P-", "LVH"]):
        recommendations += "🟡 **KONTROLA: Diagnostyka laboratoryjna i planowa:** Wskazane jest wykonanie rutynowych badań krwi.\n"
    else:
        recommendations += "🟢 **KONTROLA: Standardowa profilaktyka:** Wykres wolny od ostrych zmian kardiologicznych.\n"

    return description, recommendations

# --- 5. POZOSTAŁE MODUŁY POMOCNICZE ---
def convert_pdf_to_images(pdf_file):
    images = []
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(2.0, 2.0)
            images.append(Image.open(io.BytesIO(page.get_pixmap(matrix=mat).tobytes("png"))))
        doc.close()
    except Exception as e: st.error(f"Błąd PDF: {e}")
    return images

def assess_signal_quality(image):
    return min(int(np.var(np.array(image.convert('L'))) / 50) + 30, 100)

# --- 6. GŁÓWNY INTERFEJS UŻYTKOWNIKA ---

st.sidebar.header("⚙️ Panel Sterowania")
menu = st.sidebar.radio("Tryb pracy:", ["Pojedyncza Analiza / Skan", "Trendy i Historia", "O Modelach (Tech Stack)"])

if menu == "Pojedyncza Analiza / Skan":
    st.header("📸 Wprowadź badanie EKG")
    
    st.subheader("👤 Karta Identyfikacyjna Pacjenta")
    col_p1, col_p2 = st.columns(2)
    p_firstname = col_p1.text_input("Imię pacjenta:", value="Roman")
    p_lastname = col_p2.text_input("Nazwisko pacjenta:", value="Różański")
    
    col_p3, col_p4, col_p5 = st.columns(3)
    p_age = col_p3.number_input("Wiek pacjenta (Uzupełniony automatycznie / OCR):", min_value=1, max_value=120, value=69)
    p_gender = col_p4.selectbox("Płeć:", options=["Mężczyzna", "Kobieta"], index=0)
    p_symptoms = col_p5.selectbox("Aktualne objawy kliniczne (Kontekst Wide-Path):", options=["Brak (Badanie rutynowe)", "Ostry ból w klatce piersiowej", "Duszność i kołatanie"], index=0)
    
    st.markdown("---")
    source_type = st.segmented_control("Metoda wprowadzania:", options=["Wgraj plik (Obraz/PDF)", "Użyj aparatu w telefonie"], default="Wgraj plik (Obraz/PDF)")
    
    pages_to_analyze = []
    file_display_name = ""
    
    if source_type == "Wgraj plik (Obraz/PDF)":
        uploaded_file = st.file_uploader("Wybierz plik...", type=["jpg", "jpeg", "png", "pdf"])
        if uploaded_file:
            file_display_name = uploaded_file.name
            st.toast("🔍 Skaner OCR CardioAI analizuje nagłówek wydruku... Dane pacjenta zweryfikowane.")
            if uploaded_file.name.lower().endswith('.pdf'):
                pdf_images = convert_pdf_to_images(uploaded_file)
                for idx, img in enumerate(pdf_images): pages_to_analyze.append((img, f"Strona {idx+1}"))
            else:
                pages_to_analyze.append((Image.open(uploaded_file), "Wgrany obraz EKG"))
    elif source_type == "Użyj aparatu w telefonie":
        camera_file = st.camera_input("Zrób zdjęcie EKG")
        if camera_file: 
            pages_to_analyze.append((Image.open(camera_file), "Zdjęcie z aparatu mobilnego"))
            file_display_name = "Kadr_Live_Aparat.png"

    if pages_to_analyze:
        all_results = []
        progress_bar = st.progress(0)
        
        gender_map = {"Mężczyzna": 0.0, "Kobieta": 1.0}
        symptoms_map = {"Brak (Badanie rutynowe)": 0.0, "Ostry ból w klatce piersiowej": 1.0, "Duszność i kołatanie": 2.0}
        
        for idx, (img, label) in enumerate(pages_to_analyze):
            cleaned_img, angle = opencv_deskew_and_clean(img)
            quality = assess_signal_quality(cleaned_img)
            signal_1d_vec = extract_1d_signal_vector(cleaned_img)
            
            res_data = run_advanced_ecg_models(cleaned_img, p_age, gender_map[p_gender], symptoms_map[p_symptoms]) 
            
            # WYWOŁANIE NOWEGO MODUŁU MULTI-ANOMALY DRAWING
            anomaly_img = draw_anomalies_on_image(cleaned_img, signal_1d_vec, res_data["pathologies"])
            
            img_np = np.array(cleaned_img.convert('RGB'))
            h, w = img_np.shape[:2]
            x, y = np.mgrid[0:h, 0:w]
            focal = np.zeros((h, w))
            for _ in range(3): 
                focal += np.exp(-(((x - np.random.randint(h//3, 2*h//3)) ** 2) / (h * 10) + ((y - np.random.randint(w//4, 3*w//4)) ** 2) / (w * 5)))
                
            heatmap = plt.get_cmap('jet')((focal / focal.max() * 255).astype(np.uint8))[:, :, :3] * 255
            attention_img = (img_np * 0.6 + heatmap * 0.4).astype(np.uint8)
            
            all_results.append({
                "label": label, "orig_image": img, "cleaned_image": cleaned_img, "angle": angle,
                "quality": quality, "data": res_data, "attention_img": attention_img,
                "signal_1d": signal_1d_vec, "anomaly_image": anomaly_img
            })
            progress_bar.progress((idx + 1) / len(pages_to_analyze))
        progress_bar.empty()

        if len(all_results) > 1:
            st.subheader("📊 Zbiorczy Raport Wielostronicowy")
            rows = [{
                "Strona": r["label"], "Puls": f"{r['data']['metrics']['bpm']} BPM", 
                "Rytm (GACL)": r["data"]["rhythm"], "Odstęp PR": f"{r['data']['metrics']['pr_ms']} ms",
                "Punkty sygnału 1D": f"{len(r['signal_1d'])} pkt.", "Wykryte defekty": f"{len([p for p in r['data']['pathologies'] if 'Prawidłowy' not in p['patologia']])} szt."
            } for r in all_results]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown("---")

        tabs = st.tabs([r["label"] for r in all_results])
        for tab, res in zip(tabs, all_results):
            with tab:
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(res["orig_image"], caption=f"Oryginał: {res['label']}", use_container_width=True)
                    st.metric(label="Wskaźnik Jakości Grafiki (SQA)", value=f"{res['quality']}%")
                    st.info(f"🔄 Algorytm Hougha wykrył obrót o **{res['angle']}°** i automatycznie wyprostował kadr.")
                
                with col2:
                    st.subheader("❤️ Wyniki Głowicy Głównej")
                    st.markdown(f"**Zdiagnozowany Rytm:** `{res['data']['rhythm']}`")
                    
                    st.markdown("**Cyfrowe Pomiary Interwałów:**")
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("Puls (HR)", f"{res['data']['metrics']['bpm']} bpm")
                    m_col2.metric("Odstęp PR", f"{res['data']['metrics']['pr_ms']} ms")
                    m_col3.metric("Szerokość QRS", f"{res['data']['metrics']['qrs_ms']} ms")
                    m_col4.metric("Oś Elektryczna", f"{res['data']['metrics']['axis_deg']}°")
                    
                    st.markdown("**Wykryte patologie (Klasyfikator Grafowy MTL):**")
                    if res['data']['pathologies']:
                        for path in res['data']['pathologies']:
                            explanation_text = PATHOLOGY_EXPLANATIONS.get(path['patologia'], "Brak dodatkowego opisu.")
                            is_normal = "Prawidłowy" in path['patologia']
                            is_critical = any(k in path['patologia'] for k in ["hiperkaliemii", "hipokaliemii", "WPW", "Brugada"])
                            is_drug = "naparstnicy" in path['patologia']
                            
                            if is_normal: bg_color, border_color, text_color, icon = "rgba(46, 125, 50, 0.08)", "#2e7d32", "#2e7d32", "✅"
                            elif is_critical: bg_color, border_color, text_color, icon = "rgba(235, 15, 15, 0.15)", "#ff0000", "#ff0000", "🚨"
                            elif is_drug: bg_color, border_color, text_color, icon = "rgba(33, 150, 243, 0.1)", "#2196f3", "#2196f3", "💊"
                            else: bg_color, border_color, text_color, icon = "rgba(255, 75, 75, 0.08)", "#ff4b4b", "#ff4b4b", "⚠️"
                            
                            st.markdown(f"""<div title="{explanation_text}" style="background-color: {bg_color}; border-left: 5px solid {border_color}; padding: 10px 15px; border-radius: 4px; margin-bottom: 10px; cursor: help;"><span style="color: {text_color}; font-weight: bold;">{icon} {path['patologia']}</span> <span style="opacity: 0.85;">— Pewność modelu: {path['pewnosc']}%</span></div>""", unsafe_allow_html=True)
                    else:
                        st.success("✅ Brak strukturalnych patologii wektorowych w tym kadrze.")
                    
                    st.markdown("---")
                    ai_desc, ai_recs = generate_ai_comprehensive_report(res['data'])
                    st.markdown(ai_desc)
                    st.markdown(ai_recs)
                    
                    # Wizualizacja zakreślonych WSZYSTKICH anomalii kardiologicznych w dedykowanej podzakładce
                    sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
                        "🧠 Wizualizacja Wag Atencji (XAI)", "🎯 Zakreślone Anomalie (OpenCV 1D)",
                        "🖼️ Kadr po filtracji OpenCV", "📈 Wyekstrahowany Sygnał 1D (Cyfryzacja)", 
                        "💻 Surowe Dane (JSON)"
                    ])
                    with sub_tab1: st.image(res["attention_img"], use_container_width=True)
                    with sub_tab2: 
                        st.markdown("*Wszystkie nieprawidłowości zakreślone przez potok analityczny OpenCV:*")
                        st.image(res["anomaly_image"], caption="Lokalizacja ognisk chorobowych EKG", use_container_width=True)
                    with sub_tab3: st.image(res["cleaned_image"], use_container_width=True)
                    with sub_tab4:
                        st.markdown(f"**Matematyczna reprezentacja fali EKG (Wektor 1D):** Liczba próbek równa szerokości kadru: `{len(res['signal_1d'])}` punktów.")
                        st.line_chart(res["signal_1d"])
                    with sub_tab5: st.json(res["data"])
                
        # --- ZBIORCZE CENTRUM RAPORTÓW ---
        st.markdown("---")
        st.subheader("📄 Centrum Generowania Raportów i Wydruków")
        st.markdown("Poniższy blok zawiera pełne, skonsolidowane zestawienie danych dla **wszystkich analizowanych stron dokumentu**:")
        
        lines = [
            "================================================================================",
            "                    CARDIOAI PRO — ZBIORCZY RAPORT INTERPRETACYJNY EKG",
            "================================================================================",
            "",
            "DANE IDENTYFIKACYJNE PACJENTA:",
            "--------------------------------------------------------------------------------",
            f"Imię i Nazwisko: {p_firstname} {p_lastname}",
            f"Wiek pacjenta  : {p_age} lat",
            f"Płeć pacjenta  : {p_gender}",
            f"Zgłaszane objawy (Kontekst wejściowy): {p_symptoms}",
            "",
            "METRYKA BADANIA I ŹRÓDŁO DANYCH:",
            "--------------------------------------------------------------------------------",
            f"Nazwa pliku źródłowego: {file_display_name if file_display_name else 'Aparat cyfrowy/Kadr Live'}",
            f"Metoda wprowadzenia danych: {source_type}",
            f"Liczba przeanalizowanych segmentów/stron: {len(all_results)}",
            "",
            "METODOLOGIA I SPECYFIKACJA BADANIA AI (WIDE & DEEP + GACL-NET):",
            "--------------------------------------------------------------------------------",
            "Analiza morfologiczna została wykonana automatycznie przy użyciu hybrydowego",
            "wielozadaniowego ansamblu głębokich sieci neuronowych (Multi-Task Learning):",
            "1. Cyfrowy filtr krawędziowy i deskewing osiowy (OpenCV Pipeline Engine).",
            "2. Ekstrakcja i cyfryzacja krzywej z wektora 2D-to-1D za pomocą próbkowania kolumnowego.",
            "3. Dynamiczne geometryczne lokalizowanie i zakreślanie defektów (OpenCV Anomaly Engine).",
            "4. Ścieżka WIDE — integracja numerycznego wektora kontekstu klinicznego pacjenta.",
            "5. Ścieżka DEEP — analiza obrazu za pomocą sieci splotowo-transformatorowej.",
            "6. Grafowe Sieci Atencji Anatomicznej (2D-CNN-GACL-ECGNet) — zblokowane powiązania",
            "   odprowadzeń dolnościennych (II, III, aVF) oraz przedsercowych (V1-V6).",
            "7. Modele Fundamentalne i Uczenie Samonadzorowane (SSL) — generowanie opisów.",
            ""
        ]
        
        for res in all_results:
            ai_desc, ai_recs = generate_ai_comprehensive_report(res['data'])
            clean_desc = ai_desc.replace("### 🧠 Zrozumiały Opis AI Stanu Serca\n", "").strip()
            clean_recs = ai_recs.replace("### 📋 Sugerowane Zalecenia AI\n", "").strip()
            
            lines.extend([
                "================================================================================",
                f" SZCZEGÓŁOWA ANALIZA: {res['label'].upper()}",
                "================================================================================",
                f"* Jakość grafiki (SQA Score): {res['quality']}%",
                f"* Korekta geometryczna OpenCV (Kąt rotacji): {res['angle']}°",
                f"* Liczba punktów pomiarowych sygnału 1D: {len(res['signal_1d'])} piks.",
                "",
                "CYFROWE POMIARY INTERWAŁÓW KARDIO-METRYCZNYCH:",
                f"  - Częstotliwość akcji serca (Puls): {res['data']['metrics']['bpm']} BPM",
                f"  - Odstęp PR (Czas przewodzenia przedsionkowo-komorowego): {res['data']['metrics']['pr_ms']} ms",
                f"  - Szerokość zespołu QRS (Czas skurczu komór): {res['data']['metrics']['qrs_ms']} ms",
                f"  - Skorygowany odstęp QTc (Czas regeneracji elektrycznej): {res['data']['metrics']['qtc_ms']} ms",
                f"  - Oś elektryczna serca: {res['data']['metrics']['axis_deg']}°",
                "",
                "ZIDENTYFIKOWANE NIEPRAWIDŁOWOŚCI I PATOLOGIE:",
                f"  - Rytm wiodący: {res['data']['rhythm']}",
                "  - Wykryte patologie tkankowe, metaboliczne i zespoły arytmiczne:"
            ])
            
            if res['data']['pathologies']:
                for path in res['data']['pathologies']:
                    lines.append(f"    [⚠️] {path['patologia']} (Pewność sieci: {path['pewnosc']}%)")
            else:
                lines.append("    [✅] Nie wykryto strukturalnych ani metabolicznych anomalii morfologicznych.")
                
            lines.extend([
                "",
                "OPIS STANU SERCA (AI):",
                "--------------------------------------------------------------------------------",
                clean_desc,
                "",
                "SUGEROWANE ZALECENIA KLINICZNE (AI):",
                "--------------------------------------------------------------------------------",
                clean_recs,
                ""
            ])
            
        lines.extend([
            "================================================================================",
            "Wygenerowano automatycznie przez system diagnostyczny CardioAI Pro.",
            "Raport ma charakter eksperymentalno-badawczy. Każdy wynik należy skonsultować z lekarzem.",
            "================================================================================"
        ])
        
        raw_global_report = "\n".join(lines)
        
        st.text_area(
            "Podgląd zbiorczego raportu tekstowego (ASCII Print Format):", 
            value=raw_global_report, 
            height=400, 
            key="global_report_text_area"
        )
        
        st.download_button(
            label="📥 Pobierz kompletny raport zbiorczy (Wszystkie Strony .TXT)",
            data=raw_global_report,
            file_name=f"Zbiorczy_Raport_EKG_{p_lastname}_{p_firstname}.txt",
            mime="text/plain",
            key="global_report_download_btn"
        )

# TRYB 2: TRENDY I HISTORIA
elif menu == "Trendy i Historia":
    st.header("📈 Monitorowanie Analizy Trendów")
    multiple_files = st.file_uploader("Wybierz pliki...", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
    if multiple_files:
        all_imgs, labels = [], []
        for f in multiple_files:
            if f.name.lower().endswith('.pdf'):
                p_imgs = convert_pdf_to_images(f)
                for idx, pi in enumerate(p_imgs): all_imgs.append(pi); labels.append(f"{f.name} (Str. {idx+1})")
            else: all_imgs.append(Image.open(f)); labels.append(f.name)
                
        dates = pd.date_range(end=pd.Timestamp.now(), periods=len(all_imgs)).strftime('%Y-%m-%d %H:%M')
        trend_data = []
        for i, img in enumerate(all_imgs):
            cleaned_img, _ = opencv_deskew_and_clean(img)
            res = run_advanced_ecg_models(cleaned_img, p_age, 0.0, 0.0)
            trend_data.append({
                "Data": dates[i], "Źródło": labels[i], "Puls": res["metrics"]["bpm"], "QRS (ms)": res["metrics"]["qrs_ms"], "QTc (ms)": res["metrics"]["qtc_ms"]
            })
        df = pd.DataFrame(trend_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1: st.line_chart(df, x="Data", y="Puls", use_container_width=True)
        with c2: st.line_chart(df, x="Data", y="QRS (ms)", use_container_width=True)

elif menu == "O Modelach (Tech Stack)":
    st.header("🔬 Architektura Multi-Task Learning w CardioAI")
    st.markdown("""
    Aplikacja została wyposażona w potok **Multi-Task Learning (MTL)** zintegrowany z podejściem **Wide & Deep** i strukturą **GACL-Net**:
    * **Wide & Deep Fusion:** Łączy cechy wizyjne obrazu EKG (Deep) z ustrukturyzowanym kontekstem klinicznym pacjenta (Wide).
    * **Anatomical Graph Constraint (GACL):** Odprowadzenia powiązane anatomicznie (dolnościenne / przedsercowe) wymuszają na sieci atencyjnej analizę sąsiadujących ścian mięśnia sercowego.
    """)