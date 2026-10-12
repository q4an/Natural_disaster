# 🌍 Natural Disaster
[🇹🇷 Türkçe](#turkce) | [🇬🇧 English](#english)

> 🚀 **[Natural Disaster — Advanced](advanced/README.md)**, bu projenin geliştirilmiş sürümüdür. / The enhanced version of this project. [Türkçe](advanced/README.md#turkce) · [English](advanced/README.md#english)

<a id="turkce"></a>
### Bir mesaj, birden fazla ihtiyaç.

> “We need food and drinking water.”

Afet sırasında gelen bir mesaj hem gıda hem su ihtiyacını anlatabilir. **Natural Disaster**, İngilizce afet mesajlarını birden fazla kategoriyle etiketlemeyi araştıran bir makine öğrenmesi projesidir. Ham CSV dosyalarından başlayıp veri temizleme, metin işleme, model eğitimi ve tahmine uzanan süreci içerir.

**Python · pandas · NLTK · scikit-learn · XGBoost · SQLite**

## Mesajdan tahmine

```text
Mesajlar + kategoriler → Veri temizleme → SQLite
                                           ↓
Yeni mesaj → Metin işleme → TF–IDF → XGBoost → Kategori etiketleri
```

- **Çok etiketli sınıflandırma:** Bir mesaj birden fazla kategoriye ait olabilir.
- **Metin hazırlama:** Küçük harfe dönüştürme, noktalama temizleme, tokenization ve lemmatization.
- **Grup bazlı değerlendirme:** Birebir aynı mesajlar eğitim ve test kümelerinde birlikte bulunmaz; çapraz doğrulamada da mesaj grupları korunur.
- **Karşılaştırma:** `DummyClassifier` ile bir başlangıç modeli; micro-F1 ve macro-F1 ölçümleri.
- **Model paketi:** Eğitilen pipeline ve kategori sırası aynı dosyaya kaydedilir.

## Hızlı başlangıç

Python **3.13** ile doğrulanan paket sürümleri `requirements.txt` içinde bulunur. Aşağıdaki PowerShell komutlarını proje klasöründe çalıştırın:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords punkt punkt_tab wordnet omw-1.4
```

### Veriyi hazırlayın

Aşağıdaki iki dosyayı `data/` klasörüne yerleştirin:

| Dosya | Beklenen sütunlar |
|---|---|
| `disaster_messages.csv` | `id`, `message`, `original`, `genre` |
| `disaster_categories.csv` | `id`, `categories` |

`categories` alanı `related-1;request-0;...` biçiminde kategori-değer çiftleri içerir. İki dosya `id` üzerinden birleştirilir. `related=2` değerleri bu öğrenme akışında `1` olarak ele alınır; pozitif örneği olmayan `child_alone` eğitim hedeflerinden çıkarılır.

### Kontrol edin, eğitin, tahmin edin

```powershell
# Veri dağılımı, grup kesişimleri ve başlangıç modeli
python src/test.py

# Temiz veriyi SQLite'a kaydet
python src/procces_data.py

# Parametre araması, eğitim ve test değerlendirmesi
python src/train_data.py

# Kaydedilen modelle örnek mesaj tahmini
python src/predict_messaje.py
```

Eğitim dört parametre kombinasyonunu üç CV bölmesinde değerlendirir; her model birden fazla kategoriyi öğrenir. Süre bilgisayarın gücüne bağlıdır. Sonuç `models/disaster_model_bundle.pkl` dosyasına kaydedilir. Tahmin etmek istediğiniz mesajı `src/predict_messaje.py` içindeki `message` değişkeninden değiştirebilirsiniz.

### Notebook ile keşfedin

```powershell
python -m pip install jupyterlab
python -m jupyter lab walkthrough.ipynb
```

Notebook'u proje kökünden açın ve hücreleri sırayla çalıştırın. Paylaşım sürümünde hücre çıktıları temizlenmiştir; sonuçlar yeniden çalıştırıldığında üretilir.

## Proje yapısı

```text
natural disaster/
├── README.md
├── requirements.txt
├── .gitignore
├── walkthrough.ipynb
├── src/
│   ├── procces_data.py       # CSV → temiz veri → SQLite
│   ├── text_processing.py   # Metin işleme
│   ├── train_data.py        # Eğitim ve değerlendirme
│   ├── test.py              # Veri bölmeleri ve başlangıç modeli
│   └── predict_messaje.py   # Örnek tahmin
├── data/                    # Yerel veri; Git dışında
└── models/                  # Yerel modeller; Git dışında
```

## Sonuçları nasıl okumalı?

Kategoriler dengeli dağılmadığından yalnız accuracy yeterli değildir. **Micro-F1**, tüm etiket kararlarını birlikte değerlendirir; **macro-F1**, seyrek kategorilere de eşit ağırlık verir. Eğitim scripti kategori bazında precision, recall, F1 ve accuracy raporlar.

Yerel veriyle `src/test.py` kontrolünde 20.966 eğitim ve 5.249 test satırı elde edildi. Üç CV bölmesinde `DummyClassifier` ortalama **micro-F1: 0,3680**, **macro-F1: 0,0248** üretti. Bunlar XGBoost modelinin başarı skorları değil, karşılaştırma için başlangıç değerleridir.

Mesaj grupları tam metin eşitliğine dayanır; benzer mesajların veya aynı afet olayının farklı anlatımlarının ayrıldığını garanti etmez. Model İngilizce metinlerle çalışır. Bu proje bir öğrenme çalışmasıdır; saha kullanımı veya otomatik acil yardım yönlendirmesi için doğrulanmış değildir.

## Veri ve kaynaklar

Yerel CSV dosyalarının ilk kaynağı ve yeniden dağıtım koşulları henüz doğrulanmamıştır. Ham veri, SQLite veritabanları, eğitilmiş modeller ve ortam dosyaları `.gitignore` ile paylaşım dışında tutulur. GitHub'a kod, notebook ve dokümantasyon gönderilir. Notebook'u çalıştırdıktan sonra paylaşmadan önce **Restart Kernel and Clear Outputs of All Cells** seçeneğiyle ham mesaj çıktıları tekrar temizlenmelidir.

---

<a id="english"></a>

## 🇬🇧 English

[🇹🇷 Türkçe](#turkce) | [🇬🇧 English](#english)

### One message. Multiple needs.

> “We need food and drinking water.”

A disaster message can describe both food and water needs at the same time. **Natural Disaster** explores multilabel classification of English disaster messages. It follows the journey from raw CSV files to data cleaning, text preprocessing, model training and prediction.

**Python · pandas · NLTK · scikit-learn · XGBoost · SQLite**

### From message to prediction

```text
Messages + categories → Data cleaning → SQLite
                                           ↓
New message → Text preprocessing → TF–IDF → XGBoost → Category labels
```

- **Multilabel classification:** A single message can belong to several categories.
- **Text preprocessing:** Lowercasing, punctuation removal, tokenization and lemmatization.
- **Grouped evaluation:** Identical message texts stay out of separate training and test partitions; message groups are also preserved during cross-validation.
- **Baseline comparison:** A `DummyClassifier` baseline evaluated with micro-F1 and macro-F1.
- **Model bundle:** The trained pipeline and category order are saved together.

### Quick start

`requirements.txt` lists the package versions checked with Python **3.13**. Run these PowerShell commands from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords punkt punkt_tab wordnet omw-1.4
```

#### Prepare the data

Place these two files in the `data/` directory:

| File | Expected columns |
|---|---|
| `disaster_messages.csv` | `id`, `message`, `original`, `genre` |
| `disaster_categories.csv` | `id`, `categories` |

The `categories` field contains category-value pairs such as `related-1;request-0;...`. The files are merged on `id`. This learning workflow maps `related=2` to `1` and excludes `child_alone` from the training targets because it has no positive examples in the current dataset.

#### Check, train and predict

```powershell
# Label distribution, group overlap checks and baseline evaluation
python src/test.py

# Save the cleaned data to SQLite
python src/procces_data.py

# Parameter search, training and test evaluation
python src/train_data.py

# Predict categories for an example message using the saved model
python src/predict_messaje.py
```

Training evaluates four parameter combinations across three cross-validation folds, with each model learning multiple categories. Runtime depends on your hardware. The result is saved to `models/disaster_model_bundle.pkl`. To try your own message, edit the `message` variable in `src/predict_messaje.py`.

#### Explore the notebook

```powershell
python -m pip install jupyterlab
python -m jupyter lab walkthrough.ipynb
```

Open the notebook from the project root and run the cells in order. Cell outputs have been cleared in the shared version; running the notebook generates them again.

### Project structure

```text
natural disaster/
├── README.md
├── requirements.txt
├── .gitignore
├── walkthrough.ipynb
├── src/
│   ├── procces_data.py       # CSV → cleaned data → SQLite
│   ├── text_processing.py   # Text preprocessing
│   ├── train_data.py        # Training and evaluation
│   ├── test.py              # Data splits and baseline evaluation
│   └── predict_messaje.py   # Example prediction
├── data/                    # Local data; excluded from Git
└── models/                  # Local models; excluded from Git
```

### Reading the results

The categories are imbalanced, so accuracy alone is insufficient. **Micro-F1** evaluates all label decisions together; **macro-F1** gives each category equal weight, including rare ones. The training script reports precision, recall, F1 and accuracy for each category.

Running `src/test.py` on the local dataset produced 20,966 training rows and 5,249 test rows. Across three cross-validation folds, `DummyClassifier` achieved an average **micro-F1 of 0.3680** and **macro-F1 of 0.0248**. These are baseline comparison values, not XGBoost performance scores.

Message groups use exact text equality. This does not guarantee separation of similar messages or different accounts of the same disaster. The model works with English text. This is a learning project and has not been validated for field deployment or automatic emergency response routing.

### Data and sharing

The original source and redistribution conditions of the local CSV files have not yet been verified. Raw data, SQLite databases, trained models and environment files are excluded by `.gitignore`. Code, the notebook and documentation are intended for sharing on GitHub. After running the notebook, use **Restart Kernel and Clear Outputs of All Cells** before publishing to remove raw message outputs again.
