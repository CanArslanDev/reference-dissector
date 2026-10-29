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

**PIRAMIT** (replika moduyla, 29.10.2026 çalıştırması; replika modunun ilk çalıştırması 22.10.2026'daydı):
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
- **808:** 101 nota, 87'si C1. Artık hiçbir nota sessizliğin üzerinden tutulmuyor: 22.10.2026'da tek bir B0 notası D ve E bölümleri boyunca 15,3 s tutuluyordu; arada 1,2 ve 7,4 saniyelik iki sessizlik (808 bandı, notaların medyan tepesinin 40 dB altında) ve 808'in D'de tek başına çaldığı notalar vardı.
  - Bir nota artık 808'in sustuğu yerde bitiyor, 9 dB'lik bir yükseliş de yeni bir nota başlatıyor: D bölümünde dört C1 notası var.
  - Stem'de orada beşinci bir atak daha görünüyor. Çalmakta olan notanın üzerine yalnızca 6,5 dB yükseldiği için hâlâ o notanın içinde kalıyor (kulakla kontrol edilmedi).
  - Sentezlenen ses dalga biçimi ve zarfta kütüphanedeki en iyi 808'den daha yakın (NCC: sentez 0,938, kütüphane 0,894; zarf sapması: 1,3 ve 1,4 dB). Üst harmoniklerde (distortion) ise kütüphanedeki 808 daha yakın (sapma: kütüphane 8,7 dB, sentez 12,9 dB).
- **Bölümler:** A 1–24 · B 25–28 · C 29–32 · A 33–47 · D 48–51 · E 52–56 · A 57–71 · D 72–73. D, 808'in tek başına çaldığı yer; E, breakdown. 22.10.2026'da tutulan 808 notası ikisini tek bir bölümde (D 49–56) birleştiriyordu.
- **Live:** planlandı, henüz kurulmadı.
  - 10 Drum Rack pad'inde 630 davul notası (22.10.2026'da 7 pad'de 607), Transpose +36 ve Detune +2 ile 101 808 notası, susturulmuş üç melodik kanal, 8 lokator.
  - Bir ses artık burada 12 değil 5 vuruşla pad alıyor, kalıntı geçişinden (residual pass) gelen bir sesin ise ölçüden ölçüye tekrar etmesi gerekiyor. Böylece üç pad eklendi: bir perc, snare bandında bulunan hat dosyası ve snare'e benzer bir ses (her biri 7–8 nota, kulakla kontrol edilmedi).

**Replika modu, kalibrasyon v2 (48 sentetik beat, 22 senaryo):**
- Puanlar katı: ofset çıkarılmaz, bir vuruş sesin başladığı yerin 20 ms içindeyse sayılır.
- Sayılar senaryo ortalamaları, htdemucs stem'leriyle; replika modundan önceki kod → 29.10.2026 düzeltmeleriyle replika modu (m3 etiketi). "Analiz" boru hattının duyduğu, "plan" Live'ın çalacağı şey (Live dışında, offline çalınıyor).
- **Davul vuruşları ve 808 başlangıçları, F1@20:** analizde 0,368 → 0,888, planda 0,282 → 0,876.
- **Davullar:** 0,397 → 0,883 (analiz), 0,283 → 0,868 (plan).
  - Kick 0,25 → 0,88, snare 0,70 → 0,96, hat 0,51 → 0,86 (analiz).
  - Kick zamanlaması (medyan): +16,6 ms → +0,1 ms.
- **808 notaları (50 ms, 50 cent), F1:** 0,278 → 0,936 (analiz), 0,229 → 0,921 (plan). Perde hatasının medyanı 55 → 6 cent. Yazılan nota 0,970 oranında doğru.
- **Beat / downbeat F:** 0,72 / 0,64 → 0,96 / 0,94.
- **Melodik notalar:** F1 0,63. **Akorlar (majmin):** 0,98 (4 beat).
- **Pad'ler kanıta göre veriliyor:** bir sesin pad alması için en az 3 vuruşu (uzun bir şarkıda 16 ölçüde bir vuruş), temiz bir şablonu ve kütüphanede yakın bir eşleşmesi ya da beat'in omurgasında (backbone) bir yeri olmalı. Kalıntı geçişinden gelen bir ses ayrıca ölçüden ölçüye tekrar etmeli.
  - Eski 12 vuruş eşiğiyle açık hat planda 0,043'tü, şimdi 0,981.
  - Planın yanlış vuruşları beat başına 23,0'dan 9,6'ya düştü (bu düzeltmelerden önceki birleşik kodla, m2g-vel, karşılaştırıldığında).
  - Her davul notası, pad'deki sample'ın kendi atağı kadar erken başlıyor.
- **Planın seviyeleri kötüleşti** (m2g-vel → m3): kusursuz bir kitten uzaklık (medyan) 3,7 → 4,1 dB, denge 2,2 → 3,0 dB (replika modundan önce 7,2 dB). Pad'lerin kendi seviyesi yok: her biri sample'ını 0 dB'de çalıyor, yalnızca davul fader'ı ayarlanıyor. Yeni çalınan açık hat'ler kusursuz kitin medyan 11,4 dB üstünde. Açık konu.
- **808'in plandaki perdesi** (m2g-vel → m3): yazılan nota daha sık doğru (0,947 → 0,970), ama plan bass_low-1'in 808'ini medyan 41,7 cent sapmayla çalıyor (önce 17,5); planın 808 nota F1'i 0,926 → 0,921. Açık konu.
- **htdemucs varsayılan kalıyor:** htdemucs_ft'nin davulları daha iyi (0,895 / 0,898), ama 808 notaları ve snare'leri daha kötü.
- **Live'da doğrulanmadı:** pad başına Detune, 808'in Detune ve Glide ayarları ve kurulumun bütünü henüz Live'da kaydedilip ölçülmedi.

**Kalibrasyon v1 (6 sentetik beat):**
- Vuruş zamanları bilindiğinde 30 sesin 27'si 8.854 sample arasında doğru bulunuyor.
- Hiçbir şey bilinmeden snare transkripsiyonu F1 1,00 ve tanıma 6/6.
- Hat ve kick daha zayıf: F1 0,77 ve 0,70.

## Yeni komutlar

- `melodic`: her tonal kanal için notalar ve akorlar (Basic Pitch).
- `index-speckle`: sample'ları herhangi bir perdede eşleştirmek için speckle indeksi. `index-library` bunu artık kendisi yazıyor; bu komut daha önce indekslenmiş bir kütüphane içindir.
- `suite build`, `suite run`, `suite report`: kalibrasyon v2'yi kurar, her etiket (tag) için çalıştırır ve puanlar. `suite run` şunları alır: `--steps`, `--reuse`, `--jobs`, `--render`, `--model`, `--resume`, `--cache`.
- `analyze` ve `fit-808`, 808 okuyucusunun sonucunu ve 808 fit'ini çalışma klasörünün `cache/` alt klasöründe saklar. Stem'ler (ya da şablon), ayarlar, numpy, scipy ve soxr sürümleri ve paketin bütün `.py` dosyaları aynıysa sonraki çalıştırma bunları geri okur ve log'a hangi dosyayı okuduğunu yazar. `--no-cache` hepsini yeniden çalıştırır. `suite run` ise `--cache` verilmedikçe hiçbir şey saklamaz; böylece ölçülen süreler kodun kendi süreleridir.

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
