# Sosyal Medya Bot Tespiti: Proje Raporu

## 1. Projenin Amacı ve Kapsamı
Bu projeyi geliştirirken temel amacım, sosyal medyada giderek artan sahte hesapları ve bot ağlarını tespit edebilen bir makine öğrenmesi modeli (Binary Classification) oluşturmaktı. 

Sadece sayısal verilere (takipçi sayısı vb.) bakmak yerine, hesabın attığı tweetlerin sayısını,içeriğini ve profil resmini de inceleyen daha kapsamlı ve gerçekçi bir sistem tasarlamaya çalıştım. Böylece uygulamanın daha isabetli (0 veya 1 şeklinde, İnsan/Bot ayrımı) sonuçlar vermesini hedefledim.

## 2. Geliştirme Aşamaları ve Kullanılan Teknolojiler

### 2.1. Veri Seti ve Ön İşleme
Projede modelimi eğitmek için akademik araştırmalarda sıkça kullanılan **Cresci-2017** veri setini kullandım. Bu veri seti içerisinde gerçek kullanıcılar ve bot hesaplar etiketlenmiş şekilde bulunuyordu.

Veriyi modele vermeden önce bazı ön işlemeler (preprocessing) yapmam gerekti. Örneğin hesapların açılış tarihlerini alıp, gün cinsinden "Hesap Yaşı"na dönüştürdüm. Ayrıca sadece takipçi sayısına bakmanın yeterli olmayacağını düşünerek "Takipçi / Takip Edilen Oranı" gibi yeni değişkenler ürettim. Bazen takipçisi sıfır olup binlerce kişiyi takip eden bariz botlar olabiliyor, bunları oranlarla yakalamak daha kolay oldu.

### 2.2. Model Seçimi ve Eğitimi
Model olarak tablosal verilerde (tabular data) çok başarılı olduğunu bildiğim **XGBoost** algoritmasını seçtim. Karar ağaçları tabanlı bu yöntem iyi sonuçlar verdi. Test setinde elde ettiğim doğruluk (Accuracy) oranı %96 civarındaydı.

Ancak, bazen kullanıcıların veya botların çok uçuk değerler girmesi (örneğin günde 10 bin tweet atmak gibi) modelin yanılmasına veya "Outlier" (aykırı değer) yanılgısına düşmesine sebep olabiliyordu. Bu yüzden sisteme çok belirgin olan spam kurallarını if/else mantığı ile fiziksel bir kural olarak ekledim. Günde 5000 tweet atan bir hesabın insan olamayacağını algoritmaya matematiksel olarak baştan belirttim.

### 2.3. Doğal Dil İşleme (NLP)
Sadece sayılara bakmak botları tespit etmek için her zaman yeterli olmaz. Bu yüzden kullanıcının attığı son tweeti de analiz etmek istedim. Python NLP araçlarını kullanarak (TF-IDF ile kelime vektörizasyonu ve sonrasında Lojistik Regresyon) tweetin içeriğini sınıflandıran ikinci bir kontrol mekanizması geliştirdim.

### 2.4. Arayüz (Streamlit)
Modeli başarıyla eğittikten sonra, insanların bunu kolayca test edebilmesi için **Streamlit** kütüphanesini kullanarak bir web arayüzü hazırladım. Arayüzde sayısal giriş kutuları, tweet girmek için bir metin alanı ve fotoğraf yüklemek için bir kısım bulunuyor. Ayrıca "www.socirea.com" domaini üzerinden Render'a yönlendirme yapılarak modelime ulaşılabiliyor.

Arayüzü kodlarken kodların düzenli ve okunabilir olması için Nesne Yönelimli Programlama (OOP) yapısından faydalandım (sınıflar ve metodlar kullanarak). Ayrıca kod okuyan kişinin rahat anlaması için fonksiyonların başına docstring'ler ekledim.

## 3. Sonuç
Projenin sonunda, bir hesabın bot mu yoksa gerçek mi olduğunu sadece tek bir istatistiğine bakarak değil; hem sayısal verilerine, hem tweet içeriğine hem de profil resmine bakarak tespit eden, doğruluk oranı yüksek bir İkili Sınıflandırma (Binary Classification) projesi ortaya çıkarmış oldum. Proje geliştirme sürecinde başta Antigravity olarak Claude,Chat-GPT gibi yapay zeka agentlarından yararlandım.
