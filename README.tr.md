# reference-dissector (Türkçe özet)

Bir referans şarkıyı parçalarına ayırıp Ableton Live'da yeniden kurar. Adımlar:

1. Tempo gridini, ölçüyü ve akordu bulur.
2. Her davul vuruşunu (üst üste binenler dahil) ve 808 notalarını cent hassasiyetinde çıkarır.
3. Melodik katmanları ve akorları çıkarır.
4. Kütüphanendeki aynı sample'ları ya da en yakınlarını, perdeleri kaydırılmış olsa bile tanır.
5. 808'i sentezler.
6. Sonucu, A/B için orijinal kanallar altta susturulmuş olarak, [ableton-live-mcp](https://github.com/CanArslanDev/ableton-live-mcp) ile Live'a kurar.

**Replika modu** ([docs/12](docs/12-replica-mode.md)): trap ya da hip-hop bir referansı Live'da birebir yeniden kurmayı hedefler. Sonuç, her cevabı bilinen 48 sentetik beat üzerinde ölçülür.

Çıkış sorusu şuydu: *Duyamayan bir yapay zekâ, koyduğu sesin şarkıdaki ses olup olmadığını nasıl anlar?* Dinlemeyi okuyabileceği ölçümlere (sayılar ve görseller) çevirip bu ölçümleri cevabı bilinen örneklerle kalibre ederek. Bu repo kodu, ölçümleri ve yol boyunca öğrenilenleri içerir. Örnek şarkı Mavi – *PIRAMIT*; ayrıca kalibrasyon için üretilmiş sentetik beat'ler kullanıldı.

## Öne çıkan sonuçlar

**PIRAMIT** (replika moduyla, 22.10.2026 çalıştırması):
- **Tempo:** 137,00 BPM (136,999), ilk ölçü çizgisi 23,4 ms'de. Beat tracker'ın verdiği 136,36 BPM'de vuruşlar ölçü başına 8 ms kayıyor: 7. ölçüde 50 ms, şarkı sonunda 0,59 s (hesaplandı, ölçülmedi).
- **Akort:** melodik kanallar A440'ta (other −2, piano +1,5 cent). Sadece **808 +11,8 cent**: akortsuz olan şarkı değil, 808 sample'ı. Bu, 808 şablonunun oturduğu perde (32,93 Hz). Replika modundan önce okunan +33 cent, 808'in başındaki perde düşüşünü de duyuyordu. Prototipteki "C mi C# mı" karışıklığı da notaların bu düşüşle, C'nin 30 cent kadar üstünde okunmasından geliyor. (Önceki sürüm her şeyi +0,28 yarım ton okuyordu; bu `librosa.estimate_tuning`'in hatasıydı.)
- **Hi-hat:** Kütüphanedeki bir hat, kendi perdesinde.
  - Benzerlik (NCC) 0,984.
  - 327 vuruşun 310'unda en iyi eşleşme o.
  - Referanstan çıkarılınca kalan kısım −14,9 dB (sonraki aday −10,9 dB).
  - Karar artık "same family" (aynı aile), "same sample" (aynı sample) değil: her perdede arayınca başka bir kitin 3 yarım ton yukarıdaki hat'i 0,958'e çıkıyor ve fark (margin) 0,078'den 0,025'e düşüyor.
- **Tek ölçülerin snare'i:** Kütüphanedeki bir sample'ın 1 yarım ton aşağı çekilmiş hali.
  - Benzerlik 0,923.
  - 28/28 vuruşta en iyi eşleşme.
  - Aynı dosya 24–31. ölçülerde kendi perdesinde çalıyor (benzerlik 0,986).
- **Çift ölçülerin snare'i:** Kütüphanede yok; en yakını 0,47.
- Önceki sürümde tek ve çift ölçüler ters yazılmıştı.
- **808:** 94 nota, 79'u C1. **Hatalı:** 48. ölçüden 57. ölçüye tek bir B0 notası 15,3 s tutuluyor; arada 1,0 ve 7,2 saniyelik iki sessizlik ve 808'in tek başına çaldığı dört nota var. Henüz düzeltilmedi. Sentezlenen ses dalga biçimi ve zarfta kütüphanedeki en iyi 808'den daha yakın (NCC: sentez 0,938, kütüphane 0,894; zarf sapması: 1,3 ve 1,4 dB). Üst harmoniklerde (distortion) ise kütüphanedeki 808 daha yakın (sapma: kütüphane 8,7 dB, sentez 12,9 dB).

**Replika modu, kalibrasyon v2 (48 sentetik beat, 22 senaryo):**
- Puanlar katı: ofset çıkarılmaz, bir vuruş sesin başladığı yerin 20 ms içindeyse sayılır.
- Sayılar senaryo ortalamaları, htdemucs stem'leriyle. "Analiz" boru hattının duyduğu, "plan" Live'ın çalacağı şey (Live dışında, offline çalınıyor).
- **Davul vuruşları ve 808 başlangıçları, F1@20:** analizde 0,368 → 0,886, planda 0,282 → 0,667.
- **Davullar:** 0,397 → 0,880 (analiz), 0,283 → 0,612 (plan).
  - Kick 0,25 → 0,87, snare 0,70 → 0,96, hat 0,51 → 0,86 (analiz).
  - Kick zamanlaması (medyan): +16,6 ms → 0,0 ms.
- **808 notaları (50 ms, 50 cent), F1:** 0,278 → 0,936 (analiz), 0,229 → 0,926 (plan). Perde hatasının medyanı 55 → 6 cent.
- **Beat / downbeat F:** 0,72 / 0,64 → 0,96 / 0,94.
- **Melodik notalar:** F1 0,63. **Akorlar (majmin):** 0,98 (4 beat).
- **Planın seviyeleri**, kusursuz bir kitten uzaklık (medyan): 7,2 → 3,7 dB.
- **Plan analizin gerisinde:** bir şablon ancak 12 vuruştan itibaren pad alıyor. Açık hat analizde 0,979, planda 0,043.
- **htdemucs varsayılan kalıyor:** htdemucs_ft'nin davulları daha iyi (0,893 / 0,648), ama 808 notaları ve snare'leri daha kötü.
- **Live'da doğrulanmadı:** pad başına Detune, 808'in Detune ve Glide ayarları ve kurulumun bütünü henüz Live'da kaydedilip ölçülmedi.

**Kalibrasyon v1 (6 sentetik beat):**
- Vuruş zamanları bilindiğinde 30 sesin 27'si 8.854 sample arasında doğru bulunuyor.
- Hiçbir şey bilinmeden snare transkripsiyonu F1 1,00 ve tanıma 6/6.
- Hat ve kick daha zayıf: F1 0,77 ve 0,70.

## Yeni komutlar

- `melodic`: her tonal kanal için notalar ve akorlar (Basic Pitch).
- `index-speckle`: sample'ları herhangi bir perdede eşleştirmek için speckle indeksi. `index-library` bunu artık kendisi yazıyor; bu komut daha önce indekslenmiş bir kütüphane içindir.
- `suite build`, `suite run`, `suite report`: kalibrasyon v2'yi kurar, her etiket (tag) için çalıştırır ve puanlar. `suite run` şunları alır: `--steps`, `--reuse`, `--jobs`, `--render`, `--model`, `--resume`.

Komutların tamamı README.md'nin "Quick start" bölümünde, ayrıntıları docs/12'de.

## Belgeler

Ayrıntılar `docs/` altında, İngilizce:
- boru hattı;
- PIRAMIT vaka çalışması;
- kalibrasyon (v1 ve 48 beat'lik kalibrasyon v2);
- sample tanıma;
- 808 sentezi;
- "duyamayan model nasıl dinler";
- Live kurulumu;
- tuzaklar;
- telif;
- yol haritası;
- replika modu.

## Önemli

Şarkılar, stem'ler, şarkıdan kesilen vuruşlar, transkripsiyonlar ve kütüphane dosya adları repoya girmez; senin çalışma klasöründe kalır. Şarkıdan kesilen sesler yalnızca kişisel çalışma içindir. Kalibrasyon v2'nin beat'leri de senin kütüphanenden yapıldığı için yerelde kalır; yalnızca senaryo ortalamaları yayımlanır.
