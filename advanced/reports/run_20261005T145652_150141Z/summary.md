# Deney: run_20261005T145652_150141Z

Seçilen model: **logistic_c4** (development CV micro-F1).

Veri politikası v2: çelişkili ve anlamı doğrulanmamış etiketler karantinada; eski skorlarla doğrudan kıyaslanmaz.

| Model | CV micro-F1 | CV macro-F1 | Train-CV farkı |
|---|---:|---:|---:|
| dummy | 0.3640 | 0.0248 | 0.0000 |
| logistic_c1 | 0.6124 | 0.2108 | 0.0605 |
| logistic_c4 | 0.6527 | 0.3040 | 0.1599 |
| xgboost_depth4 | 0.6473 | 0.3582 | 0.0536 |

## Eşik seçimi ve son test

Validation micro-F1: varsayılan 0.6694, eşik seçimi sonrası 0.6882 (seçimde kullanılan veri).
Son test micro-F1: 0.6769; macro-F1: 0.4756.
Test sonucu hiçbir model veya eşik kararında kullanılmadı. Bu test artık görülmüştür; sonraki geliştirmeler validation üzerinde yapılmalıdır.

## İnceleme dosyaları

- `validation_errors.csv`: etiket başına yanlış pozitif ve negatif mesajlar.
- `validation_segments.csv`: kaynak ve uzunluğa göre performans.
- `learning_curve.png`: tek sabit CV katında veri miktarı deneyi; kesin overfitting/underfitting teşhisi değildir.
- `feature_effects.csv`: doğrusal katsayılar veya ağaç özellik önemleri; nedensel açıklama değildir.
- `reliability.png`: skorların güvenilirlik incelemesi; kalibrasyon uygulanmadı.
- `quarantine.csv`: elle incelenecek dışlanan kayıtlar.

## Deney günlüğü

**Değişiklik:** Veri politikası, bağımsız bölmeler, model karşılaştırması ve validation eşikleri.
**Hipotez:** Sözcük bilgisi kullanan modeller çoğunluk baseline'ını geçer.
**Validation sonucu:** logistic_c4, CV micro-F1 0.6527.
**Önceki sonuç:** Aynı veride dummy CV micro-F1 0.3640.
**Değişim:** 0.2888 mutlak F1.
**Yorum:** Eşik seçimi ayrı validation üzerinde; test yalnız son ölçümde açıldı.
**Sonraki karar:** Validation yanlış negatiflerini oku; yeni özellik deneyi için hipotez yaz.
