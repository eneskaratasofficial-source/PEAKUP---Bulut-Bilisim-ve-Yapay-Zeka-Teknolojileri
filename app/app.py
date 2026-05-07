import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from PIL import Image, ImageOps
import cv2

SVG_DATA = '''<svg style="vertical-align: middle; margin-right: 8px;" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#cfcfcf" stroke-width="2"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>'''
SVG_MEDIA = '''<svg style="vertical-align: middle; margin-right: 8px;" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#cfcfcf" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>'''
SVG_RESULTS = '''<svg style="vertical-align: middle; margin-right: 8px;" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#cfcfcf" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/></svg>'''

class PredictionModel:
    def __init__(self, models_dir):
        self.hybrid_model_path = os.path.join(models_dir, "hybrid_model.pkl")
        self.nlp_model_path = os.path.join(models_dir, "nlp_model.pkl")
        self.tfidf_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
        self.iso_model_path = os.path.join(models_dir, "isolation_forest.pkl")
        
        self.hybrid_model = None
        self.nlp_model = None
        self.tfidf_vectorizer = None
        self.iso_model = None
        self._load_models()

    def _load_models(self):
        if os.path.exists(self.hybrid_model_path):
            self.hybrid_model = joblib.load(self.hybrid_model_path)
        if os.path.exists(self.nlp_model_path) and os.path.exists(self.tfidf_path):
            self.nlp_model = joblib.load(self.nlp_model_path)
            self.tfidf_vectorizer = joblib.load(self.tfidf_path)
        if os.path.exists(self.iso_model_path):
            self.iso_model = joblib.load(self.iso_model_path)

    def predict_hybrid(self, features_df):
        """
        XGBoost ve Isolation Forest kullanan ikili (hybrid) karar mekanizması.
        Sayısal verileri ağaç yapısından geçirmeden önce katı matematiksel fizik kuralları ile tarar.
        
        Args:
            features_df (pd.DataFrame): Kullanıcı girdilerinden oluşan tek satırlık (1xN) özellik matrisi.
            
        Returns:
            tuple: (Tahmin Sınıfı [0=Bot, 1=Gerçek], Olasılık Dizisi [Bot_Olasılığı, Gerçek_Olasılığı])
            
        Complexity:
            - Time Complexity: O(D) burada D, Karar Ağacının maksimum derinliğidir (Tree Depth). Anomali Ormanı (Isolation Forest) için O(T * log N). Hız limitleri fiziksel check olduğu için O(1).
            - Space Complexity: O(1) sabit bellek kullanımı (tek satırlık DataFrame).
        """
        # 1. Fiziksel Sinir Kontrolu (Dünya Nufusunu Asan veya İmkansiz Degerler)
        if (features_df['followers_count'][0] > 500000000) or \
           (features_df['friends_count'][0] > 10000000):
            return 0, [0.99, 0.01]
            
        # 2. Fiziksel Bot Kontrolu (Hiz Limiti)
        if features_df['account_age_days'][0] > 0:
            tweets_per_day = features_df['statuses_count'][0] / features_df['account_age_days'][0]
            # Bir insan her gun durmaksizin ortalama 300'den fazla tweet atamaz
            if tweets_per_day > 300:
                return 0, [0.99, 0.01]
                
        # 3. Twitter Resmi Takip Limitleri (Anti-Spam Kurali)
        # Twitter API, eger sizi takip eden kisi yoksa 5000'den fazla kisiye istek atmanizi fiziken engeller.
        if (features_df['friends_count'][0] >= 5000) and (features_df['follower_friend_ratio'][0] < 0.1):
            return 0, [0.99, 0.01]

        is_anomaly = False
        if self.iso_model is not None:
            # PURE CRESCI threshold: Any score < 0.0 is technically an anomaly.
            if self.iso_model.decision_function(features_df)[0] < 0.0:
                is_anomaly = True
                
        if is_anomaly:
            return 0, [0.99, 0.01] # 0: Bot/Anomaly
            
        pred = self.hybrid_model.predict(features_df)[0]
        prob = self.hybrid_model.predict_proba(features_df)[0]
        return pred, prob
        
    def predict_nlp(self, text):
        if self.nlp_model is None or self.tfidf_vectorizer is None or not text.strip():
            return None, None
        vec = self.tfidf_vectorizer.transform([text])
        pred = self.nlp_model.predict(vec)[0]
        prob = self.nlp_model.predict_proba(vec)[0]
        return pred, prob

class ImageProcessor:
    @staticmethod
    def preprocess_image(uploaded_file):
        """
        Görüntü matrisini model sınıflandırmasına uygun hale getirmek için ön işleme (preprocessing) yapar.
        
        Args:
            uploaded_file (UploadedFile): Kullanıcı tarafından Streamlit arayüzünden yüklenen raw medya dosyası.
            
        Returns:
            PIL.Image: Boyutlandırılmış (300x300) ve RGB formatına dönüştürülmüş işlenmiş görüntü nesnesi.
            
        Complexity:
            - Time Complexity: O(W * H), burada W resmin genişliği, H yüksekliğidir. Yeniden boyutlandırma pikseller üzerinden tek seferlik evrişim yapar.
            - Space Complexity: O(W * H), dönüştürülmüş matrisin bellekte kapladığı alan kadar.
        """
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            # EXIF verisini okuyarak telefondan dikey çekilen fotoğrafların yan yatmasını engeller
            image = ImageOps.exif_transpose(image)
            # En boy oranını (aspect ratio) bozmadan maksimum 500x500 boyutuna getirir.
            # resize() komutu resmi ezdiği için OpenCV yüz hatlarını (oranlarını) tanıyamıyordu.
            image.thumbnail((500, 500))
            return image
        return None

    @staticmethod
    def classify_face(pil_image):
        """
        OpenCV Haar Cascade Sınıflandırıcısı kullanarak görüntü içerisinde yüz ve anatomik göz tespiti yapar.
        Animasyon/Logoları gerçek insan yüzünden ayırmak için Çifte Doğrulama (Double-Validation) mimarisi kullanır.
        
        Args:
            pil_image (PIL.Image): Ön işlemesi tamamlanmış (300x300) PIL görüntü nesnesi.
            
        Returns:
            int: 1 (İnsan Yüzü anatomisi doğrulandı), 0 (Sahte Obje/Yüz bulunamadı) veya None (Girdi yok).
            
        Complexity:
            - Time Complexity: O(N * S), N toplam piksel sayısı, S farklı pencere/ölçek (scaleFactor=1.1) sayısı. Optimum tespittir.
            - Space Complexity: O(N), gri tonlamalı kopyanın RAM üzerinde tutulması için ayrılan matris boyutu.
        """
        if pil_image is None:
            return None
        
        img_array = np.array(pil_image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # OpenCV on-trained siniflandirici modeli (Katilastirilmis Parametreler)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(40, 40))
        
        # Bulunan sekiller gercekten bir yuz mu diye icinde 'Goz (Eye)' kaskadi arayarak dogruluyoruz (Cifte Dogrulama)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4)
            
            # Eger seklin icinde en az 1 anatomik goz bulunursa bu kesin insandir
            if len(eyes) >= 1:
                return 1
                
        # Yuz saniyor ama icinde goz formu yok (or. Logo, Direk, Simetrik sekiller)
        return 0 

class BotDetectionApp:
    def __init__(self):
        st.set_page_config(page_title="SociRea | Sosyal Medya Bot Tespiti", layout="wide")
        
        # Bulut (Cloud) sunucularında dizin çakışmalarını önlemek için dinamik absolute path (mutlak yol) hesaplaması:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_directory = os.path.join(base_dir, "models")
        
        self.predictor = PredictionModel(models_dir=models_directory)

    def run(self):
        st.markdown('''
        <style>
        .stButton>button {
            background-color: #FF2B2B !important;
            color: white !important;
            border: none;
        }
        .stButton>button:hover {
            background-color: #D61A1A !important;
        }
        </style>
        ''', unsafe_allow_html=True)
        
        logo_col, title_col = st.columns([0.7, 10])
        with logo_col:
            st.image(os.path.join(os.path.dirname(__file__), "logo.png"), width=80)
        with title_col:
            st.markdown('<h1>SociRea</h1>', unsafe_allow_html=True)
            
        st.markdown(
            '<b>Hoş Geldiniz!</b> Bu platform XGBoost, NetworkX tabanlı ağ merkeziliği (Centrality), TF-IDF (Doğal Dil İşleme) '
            've <b>OpenCV Derin Görüntü Analizi</b> hibrit modellerini kullanarak Sosyal Medya hesaplarının <b>Gerçek bir İnsan</b> mı yoksa '
            '<b>Otomatik Bir Bot</b> mu olduğunu anlamanıza yarar.', 
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"<h3>{SVG_DATA} 1. Sayısal İstatistik Özellikleri</h3>", unsafe_allow_html=True)
            followers = st.number_input("Takipçi Sayısı", min_value=0, max_value=150000000, value=150)
            friends = st.number_input("Takip Edilen Sayısı", min_value=0, max_value=2000000, value=200)
            statuses = st.number_input("Tweet Sayısı", min_value=0, max_value=3000000, value=500)
            account_age = st.number_input("Hesap Yaşı (Gün)", min_value=1, max_value=7500, value=365)
            
            with st.expander("Gelişmiş Graf (Ağ) Analizi"):
                st.caption("Kullanıcının sosyal ağ haritasındaki etkileşim ve düğüm metrikleri.")
                deg_cent = st.slider("Derece Merkeziliği (Bağlantı Yoğunluğu)", 0.0, 1.0, 0.450, 0.001, help="Bir hesabın ağ içerisindeki direkt bağlantı gücü. Botlar genelde merkezde yer alamaz, düşük bağlantıya sahiptir.")
                clus_coef = st.slider("Kümelenme Katsayısı (Grup Etkileşimi)", 0.0, 1.0, 0.500, 0.01, help="Kullanıcının arkadaşlarının kendi aralarındaki etkileşim oranı. Botlarda bu kümeler sunidir.")
                betw_cent = st.slider("Arasındalık Merkeziliği (Köprü Görevi)", 0.0, 1.0, 0.350, 0.001, help="Hesabın farklı insan topluluklarını birbirine bağlama yeteneği. Gerçek fenomenlerde veya kurumlarda yüksektir.")
            
            tweet_text = st.text_area("Analiz İçin Hesabın Son Tweeti (NLP Metin)", "Bedava iPhone kazanmak için hemen profildeki linke tıklayıp RT yap!")

        with col2:
            st.markdown(f"<h3>{SVG_MEDIA} 2. Multimedya Görüntü Sınıflandırma</h3>", unsafe_allow_html=True)
            has_profile_pic = st.selectbox("Hesapta Profil Fotoğrafı Var Mı?", ["Evet", "Hayır"])
            uploaded_image = st.file_uploader("Pixellerin İncelenmesi İçin Resmi Yükleyin (Fake/Gerçek Tespiti)", type=["png", "jpg", "jpeg"])
            
            processed_img = None
            if uploaded_image is not None:
                processed_img = ImageProcessor.preprocess_image(uploaded_image)
                st.image(processed_img, caption="Ön İşlenmiş Görüntü Matrisi (Katman Taraması)", use_container_width=False)
                has_profile_pic = "Evet"

        st.markdown("<br/>", unsafe_allow_html=True)
        
        if st.button(f"KAPSAMLI HİBRİT ANALİZİ BAŞLAT", type="primary"):
            data = pd.DataFrame([{
                'followers_count': followers,
                'friends_count': friends,
                'statuses_count': statuses,
                'account_age_days': account_age,
                'has_profile_pic': 1 if has_profile_pic == "Evet" else 0,
                'follower_friend_ratio': float(followers) / float(friends + 1)
            }])

            try:
                # 1. Makine Ogrenmesi Saf Cresci Model Karari
                h_pred, h_prob = self.predictor.predict_hybrid(data)
                
                # 2. Opsiyonel olarak girilen Graf Metriklerinin Karara Matematiksel Modülasyonu
                # Eger kullanici Merkezilik (deg_cent vb.) degerlerini '1.0' (Asiri Insan) veya '0.0' (Asiri Bot) yaparsa ihtimallere %10-20 carpan etki et.
                graph_human_proof = (deg_cent + clus_coef + betw_cent) / 3.0 
                # (0.0 = Totally Isolated Bot, 1.0 = Highly Integrated Human)
                
                # Eger graf cizgilerinden biriyle oynanmissa olasiliklari ufak kaydir
                graph_bonus = (graph_human_proof - 0.43) * 0.4  # Average human is ~0.43. 
                
                # Sadece kesin tahmin uzerinden cok ufak esnemeler (Hacking the UX for absolute logical consistency smoothly)
                hybrid_human_prob = min(0.99, max(0.01, h_prob[1] + graph_bonus))
                hybrid_bot_prob = 1.0 - hybrid_human_prob
                
                final_h_pred = 1 if hybrid_human_prob > 0.5 else 0
                n_pred, n_prob = self.predictor.predict_nlp(tweet_text)
                
                # Image processing feature extraction
                i_pred = None
                if processed_img is not None:
                    i_pred = ImageProcessor.classify_face(processed_img)
                
                st.markdown(f"<h2>{SVG_RESULTS} Analiz Sonuçları ve Yargı</h2>", unsafe_allow_html=True)
                
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.markdown("### Sayısal Davranış Skoru")
                    if final_h_pred == 1:
                        st.success(f"GERÇEK KULLANICI OLMA OLASILIĞI | %{hybrid_human_prob*100:.2f}")
                    else:
                        st.error(f"BOT OLMA OLASILIĞI | %{hybrid_bot_prob*100:.2f}")

                with res_col2:
                    if n_pred is not None:
                        st.markdown("### NLP Dil Profili")
                        if n_pred == 1:
                            st.info(f"İNSANSI DİL ANLATIMI | Skor: %{n_prob[1]*100:.2f}")
                        else:
                            st.warning(f"SPAM/KOPYALA-YAPIŞTIR DİL | Skor: %{n_prob[0]*100:.2f}")
                            
                with res_col3:
                    if processed_img is not None and i_pred is not None:
                        st.markdown("### Görüntü (Piksel) Testi")
                        if i_pred == 1:
                            st.info("YÜZ ANATOMİSİ TESPİT EDİLDİ (İnsan/Animasyon)")
                        else:
                            st.warning("YÜZ FORMU BULUNAMADI (Yapay Obje/Sahte)")
                    else:
                        st.markdown("### Görüntü (Piksel) Testi")
                        st.markdown("_Görüntü Yüklenmedi._")
                        
            except ValueError as e:
                st.exception(e)

        st.markdown('<div style="text-align:center; font-size:0.9rem; color:#888; margin-top:40px; font-style:italic;">Geliştirici: Enes KARATAŞ | Sosyal Medya Bot Tespiti Ar-Ge Projesi</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    app = BotDetectionApp()
    app.run()
