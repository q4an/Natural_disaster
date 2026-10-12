# Veri politikası — v2.0

Amaç: İngilizce afet mesajlarını birden fazla ihtiyaç/olay kategorisine ayırmak.
Ham CSV dosyaları değişmez. Yeni deney akışı `src/data_quality.py` üzerinden çalışır.

## Birleştirme ve tekrarlar

1. Her kaynakta bütün sütunları aynı kayıtları tekilleştir.
2. Aynı `id` için farklı mesaj veya etiket kaydı varsa o kimliğin kayıtlarını karantinaya al.
   İlk kaydı seçme, etiketleri OR ile birleştirme: doğru anotasyon bilinmiyor.
3. Eşleşmeyen kimlikleri raporla; kalan kaynakları `validate="one_to_one"` ile birleştir.
4. Etiket adları/sırası ve değerleri doğrulanır. Tanınmayan şema sessizce kabul edilmez.
5. `related=2` anlamı doğrulanmadığı için bu satırlar geçici olarak karantinadadır.
   Bu bir domain gerçeği değil, belirsiz etiketleri öğrenmeye katmama kararıdır.
6. Eksik veya yalnız boşluk içeren mesajları çıkar. `original` eksikliği, İngilizce `message` mevcutsa kayıt silme sebebi değildir.
7. Grup anahtarı Unicode NFKC + casefold + boşluk birleştirmedir. Noktalama, sözcükler ve olumsuzluk korunur.
8. Aynı anahtardaki farklı etiketli mesajların tamamını karantinaya al. Tutarlı tekrarlar için en küçük kaynak kimliğini tut.

Karantina, silme veya yeniden etiketleme değildir: ham dosyalar ve `quarantine.csv` korunur.
Karantina tablosunda iki kaynak ayrı satırlar oluşturabilir; toplam karantina satırı, temiz veriden çıkarılmış tekil mesaj sayısı değildir.
Kaynak/genre bilgisi farklı tutarlı tekrarların en küçük kimliği seçilir; bu tercih kaynak dağılımını biraz etkileyebilir.
Yakın anlamlı tekrarlar, çeviri eşdeğerleri veya aynı afet olayı bu anahtarla kesin yakalanmaz.
Bu deney görülmemiş metne genellemeyi ölçer; yeni bir afet olayına veya başka dile genellemeyi kanıtlamaz.

## Hedef ve feature politikası

`id`, `original`, `genre`, grup anahtarı modele feature olarak girmez. Model yalnız `message` kullanır.
Hedef seçimi yalnız development bölümünde yapılır; tek değerli hedefler öğrenilen çıktıdan çıkarılır.
`child_alone` mevcut veride pozitifsizdir; ham/temiz tablodan silinmez, model bu hizmeti sunduğunu iddia etmez.
Yeni veride pozitif örnekler varsa tekrar öğrenilebilir. Kategori sırası ve eşikler modelle birlikte kaydedilir.
Uzun mesajlar sırf uzun oldukları için atılmaz; uzunluk segmentinde performans ayrıca incelenir.

## Bölme ve seçim

- %20 test, kalan havuzun %25'i validation: yaklaşık %60 development / %20 validation / %20 test.
- İki dış bölme `GroupShuffleSplit`, development CV'si üç katlı `GroupKFold`; tohum 42.
- Grup kesişimleri doğrulanır; CV etiket destekleri ayrı dosyadadır. Bu yöntem multilabel stratification değildir.
- TF–IDF her CV eğitim bölümünde pipeline içinde fit edilir.
- Model/config seçimi development CV ortalama micro-F1; macro-F1 ve train-validation farkı da raporlanır.
- Etiket eşikleri yalnız validation üzerinde F1 ile seçilir. En az 20 pozitif ve 20 negatif yoksa 0.5 korunur.
- Eşik aralığı 0.05–0.95, adım 0.05; eşit F1 için 0.5'e yakın eşik tercih edilir.
- Model validation'a yeniden fit edilmez: eşiklerin seçildiği skor dağılımı korunur.
- Test, model ve eşikler kaydedildikten sonra bir kez değerlendirilir. Sonuca göre seçim yapılmaz.

Eşik seçiminin yapıldığı validation skoru iyimser olabilir. Yeni denemeleri test skoruna göre seçmeyin.
Tekrar çalıştırmak aynı testin yeniden görülmesini sağlar; yeni bağımsız değerlendirme oluşturmaz.
İnsan inceleme kapasitesi/yanlış yönlendirme maliyeti verilmediği için F1, iş maliyetinin vekilidir.

## Sürüm ve kaynak

Her koşu ham CSV SHA256 özetlerini, kod kopyasını/özetlerini, bağımlılık sürümlerini, bölme kimliklerini ve parametreleri kaydeder.
Veri politikası v2.0 eski notebook/`train_data.py` temizliğiyle aynı değildir; eski 0.3680 baseline ile doğrudan fark hesaplanmaz.
Veri kaynağı ve yeniden dağıtım lisansı yerel dosyalardan doğrulanamıyor. Notebook'taki referans,
bu dosyaların lisansının doğrulandığı anlamına gelmez. README'de bu eksik açık tutulur; ham veri için lisans uydurulmaz.
