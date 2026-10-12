# Deney incelemesi — 5 Ekim 2026

Koşu: [run_20261005T145652_150141Z](run_20261005T145652_150141Z/summary.md).
Bu inceleme gerçekten çalıştırılan deneyin sonucudur; daha önce notebook'ta duran çıktılara dayanmaz.

## Yapılanlar

| Önce eksik olan | Uygulama ve kanıt |
|---|---|
| Tekrar/çelişki politikası | `DATA_POLICY.md`; `data_audit.json`, `quarantine.csv` |
| Bağımsız train/test/CV | Dış üç bölme ve üç CV katında sıfır mesaj grubu kesişimi |
| EDA | Development etiket/kaynak/uzunluk grafiği ve etiket birlikte görülme tablosu |
| Referans/model karşılaştırması | Dummy, LogisticRegression C=1/C=4, XGBoost; aynı üç CV katı |
| Kalıcı deney sonuçları | Koşu kimliği, log, config, model/veri/kod SHA256, kat sonuçları |
| Hata analizi | Etiket başına FP/FN mesajları; genre ve uzunluk segmentleri |
| Overfitting/underfitting incelemesi | Eğitim-CV farkları ve sabit katlı üç noktalı öğrenme eğrisi |
| Eşik/kalibrasyon | Validation eşik optimizasyonu; reliability/Brier tanısı (kalibrasyon fit edilmedi) |
| Yorumlanabilirlik | Etiket başına pozitif/negatif katsayılar ve demo içinde yerel sözcük katkıları |
| Dokümantasyon/sürümler | README, requirements, veri politikası, model paketi ve kod snapshot'ı |
| Demo | İngilizce mesaj girişi, kategori/eşik/skor ve sözcük katkısı; loopback üzerinde |

## Veri ne söylüyor?

Kaynak başına 26.248 satır vardı. Mesaj dosyasında 68, kategori dosyasında 32 tam tekrar çıkarıldı.
36 kimliğin çelişkili anotasyonları karantinaya alındı. Bundan sonra kalan veride 187 adet `related=2`,
bir boş mesaj, sekiz çelişkili normalize mesaj grubu bulundu; bunlar da karantinaya alındı.
12 tutarlı normalize metin tekrarı tekilleştirildi. Son veri 25.926 mesajdır.
Bu sayılar işlem sırasına bağlıdır: örneğin ham verideki tüm `related=2` satırlarının sayısı değildir.
Karantina iki kaynağın kayıtlarını da içerdiğinden 314 karantina satırı tekil dışlanan mesaj sayısı olarak okunmaz.

Development 15.555, validation 5.185, test 5.186 mesaj. `child_alone` yalnız development hedef kontrolüyle
öğrenilen şemadan çıkarıldı; kalan 35 etiketin her CV validation katında en az 21 pozitif örneği vardı.
Yine de 21 örnek nadir etiketlerde kararlı performans garantisi değildir.
Veri politikasının değiştirdiği örnek kümesi nedeniyle eski 0,3680 baseline ile gelişim farkı hesaplanmamalı.

## Modelleri karşılaştırma

| Model | CV micro-F1 | CV macro-F1 | Eğitim micro-F1 | Ortalama fit süresi |
|---|---:|---:|---:|---:|
| Dummy | 0,3640 | 0,0248 | 0,3640 | 0,02 sn |
| Logistic C=1 | 0,6124 | 0,2108 | 0,6729 | 2,11 sn |
| Logistic C=4 | **0,6527** | 0,3040 | 0,8127 | 2,33 sn |
| XGBoost derinlik 4 | 0,6473 | **0,3582** | 0,7009 | 51,79 sn |

Fit süreleri bu makinedeki ölçümlerdir; tahmin süresini içermez.
Başlangıçta belirlenen seçim ölçütü CV micro-F1 olduğu için Logistic C=4 seçildi.
XGBoost macro-F1'de daha iyi: model seçimi nadir ihtiyaçlara verilen önceliğe göre değişebilir.
Üç katlık yakın micro-F1 farkını istatistiksel üstünlük olarak sunmuyoruz.
Seçili model aynı veri üzerindeki Dummy'ye göre micro-F1'de 0,2888 mutlak artış sağladı.

## Öğrenme eğrisi: ne teşhis edebiliriz?

Seçilen modelde eğitim–CV micro-F1 farkı yaklaşık 0,160. C=1'de fark 0,060 olsa da CV skoru daha düşük.
Yani yalnız farkı küçültmek başarı değildir: daha güçlü düzenlileştirme burada bazı yararlı sinyalleri de bastırıyor.

Tek sabit CV katında 2.592 / 5.185 / 10.370 eğitim mesajı kullanıldığında validation micro-F1
0,600 / 0,627 / 0,654 oluyor; eğitim skoru yaklaşık 0,81 kalıyor.
Bu, eğitim örneklerine fazla uyum ve daha fazla veriyle azalan bir genelleme açığıyla uyumludur.
Eğri henüz düzleşmemiş; daha fazla kaliteli veri faydalı olabilir. Tek katlı eğri, kesin teşhis veya güven aralığı değildir.
Tüm model için kesin bir underfitting hükmü veremeyiz; bazı etiketlerde temsil ve anotasyon sorunları ayrıca var.

## Eşikler ve son test

Validation eşikleri seçilirken micro recall 0,5748'den 0,7140'a çıktı; precision 0,8014'ten 0,6641'e indi.
Bu validation verisi eşik seçiminde kullanıldığı için iyileşmenin tarafsız ölçümü değildir.
Model ve eşikler kaydedildikten sonraki test sonucu:

| Test metriği | 0,5 eşik | Önceden dondurulmuş etiket eşikleri |
|---|---:|---:|
| Micro-F1 | 0,6606 | **0,6769** |
| Macro-F1 | 0,3312 | **0,4756** |
| Micro precision | **0,8074** | 0,6623 |
| Micro recall | 0,5590 | **0,6922** |
| Hamming loss (düşük iyi) | **0,0527** | 0,0606 |
| Tüm etiketlerin tam eşleşmesi | **0,2877** | 0,2374 |

Eşik seçimi her metriği iyileştirmiyor: daha çok ihtiyacı yakalarken daha çok yanlış alarm üretiyor.
Bu yüzden bu modeli gerçek yönlendirme için hazır sayamayız; insan inceleme kapasitesi ve hata maliyetleri tanımlanmalı.
Test `medical_help` recall'u 0,413; `search_and_rescue` recall'u 0,311. `shops` için 25 pozitifin hiçbiri yakalanmadı.
Bu sonuçlar sınırlılık olarak raporlandı; testteki hatalara bakıp eşikler tekrar değiştirilmedi.

## Hata analizi ve iş yorumu

Validation üzerinde `medical_help` için 229 yanlış negatif, 157 yanlış pozitif var.
İlk örneklerde üç olası örüntü görüldü (sistematik yaygınlıkları ayrıca sayılmalı):

1. Açlık/susuzluk veya genel yardım talebiyle sağlık ihtiyacı dolaylı anlatılıyor. Bunların bir kısmında
   metinden sağlık etiketi açıkça çıkarılamıyor; anotasyon kapsamı da incelenmeli.
2. Yazım/çeviri sorunları ve kısa ifadeler var. Bir örnekte ilk yardım açıkça istenmesine rağmen düşük skor alınmış.
3. Bazı yanlış pozitiflerde ilaç, bakım veya mülteci ihtiyacı geçiyor. `medical_help` ile
   `medical_products` sınırı model kadar anotasyon politikasını da ilgilendiriyor.

Kaynak bazında validation micro-F1: direct 0,7314 (2.133 örnek), news 0,6545 (2.595), social 0,6974 (457).
Haberler doğrudan talep mesajlarından daha zor görünüyor; dağılım ve etiket bileşimi de farklı olabilir.
500 karakter üstünde skor 0,6119, ancak yalnız 19 örnek var; bunu güçlü genelleme olarak yorumlamayın.
Bu segment sonuçları eşik seçilen validation üzerindedir ve iyimser olabilir.

## Model hangi sözcüklere dayanıyor?

`water` için water/thirst/thirsty; `search_and_rescue` için rescue/trapped/stranded;
`medical_help` için medical/injured/health güçlü pozitif katsayılara sahip.
`need food` ifadesinin `medical_help` için negatif katsayısı var: aynı mesajda gıda ile sağlık ihtiyacı
birlikte geçtiğinde modelin ne yaptığı incelenmeye değer. Katsayı tek başına hata nedeni veya nedensellik kanıtı değildir;
diğer sözcükler, TF–IDF ağırlıkları ve model sabit terimi de sonucu etkiler.
Demo pozitif yerel TF–IDF × katsayı katkılarını gösterir; bunlar tam açıklama veya olasılık değildir.

## Skorları olasılık sayabilir miyiz?

Reliability grafiğinde water için 0,20–0,30 skor aralığındaki 65 mesajın ortalama skoru 0,243,
gerçek pozitif oranı 0,462. Bu aralıkta model ihtiyacı olduğundan düşük puanlıyor.
Bazı yüksek medical_help aralıklarında ise az örnek var; grafikler dikkatle okunmalı.
Brier skorları etiket tablolarında. Nadir etiketlerde düşük Brier tek başına iyi kalibrasyon anlamına gelmez.
Kalibrasyon uygulanmadı; istenen "eşik optimizasyonu veya kalibrasyon" adımı eşik optimizasyonuyla karşılandı.
Olasılığa dayalı operasyon kararı gerekirse ayrıca bağımsız/group-aware kalibrasyon deneyi gerekir.

## Doğrulama ve açık sınırlar

Altı davranış testi geçti: veri politikası, geçersiz etiket şeması, deterministik grup bölme,
eşik destek/tie kuralı, validation sözcüğünün vocabulary'ye girmemesi ve inference eşik/input davranışları.
Tüm aday CV eğitimleri ve son test tamamlandı. Kaydedilmiş paket yeniden yüklenerek demo üzerinden tahmin edildi.
HTTP'de boş/çok uzun/yanlış tipli girdiler 400, normal ve sözlük dışı mesajlar tanımlı çıktılar verdi.
Tarayıcıda örnek mesaj, kategori etiketleri, eşikler ve sözcük katkıları görüntülendi.

Veri kaynağı/lisansı doğrulanmadı; olay/zaman bazlı dış doğrulama yok; yakın anlamlı tekrarlar tamamen yakalanmıyor.
Demo yerel eğitim aracıdır, production deployment değildir. Eski scriptler öğrenme geçmişi olarak tutuldu;
güncel akış ve raporlar README'de açıkça ayrılıyor.

## Tek sonraki öğrenme görevi

`validation_errors.csv` içinden on `medical_help` yanlış negatifini okuyup
"açık sağlık ifadesi", "dolaylı ihtiyaç", "anotasyon belirsizliği" diye elle gruplandır.
Yeni model eklemeden önce bu gözlemlerden bir hipotez seç. Bu, rastgele model aramaktan daha açıklayıcı bir sonraki adımdır.
