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

**PIRAMIT** (replika moduyla, 29.10.2026 çalıştırması; `fit-808` ve `plan` 74931ce'nin son koduyla yeniden çalıştırıldı; replika modunun ilk çalıştırması 22.10.2026'daydı):
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
  - Planın çaldığı one-shot artık, perde düşüşü oturana kadar (269 ms) şablonun kendi perdesini izliyor. 120–250 ms'lik C1 notaları m3 kodunun one-shot'ıyla 41 centlik bir Detune düzeltmesi istiyordu, şimdi −0,1 cent. Plan çaldığında 101 notanın 91'i analizin duyduğu perdenin 20 cent yakınında (önce 82). Fit referansına karşı dalga biçimi NCC'si 0,921'den 0,880'e düştü (kulakla kontrol edilmedi).
- **Bölümler:** A 1–24 · B 25–28 · C 29–32 · A 33–47 · D 48–51 · E 52–56 · A 57–71 · D 72–73. D, 808'in tek başına çaldığı yer; E, breakdown. 22.10.2026'da tutulan 808 notası ikisini tek bir bölümde (D 49–56) birleştiriyordu.
- **Live:** planlandı, henüz kurulmadı.
  - 10 Drum Rack pad'inde 630 davul notası (22.10.2026'da 7 pad'de 607), Transpose +36 ve Detune +1 ile 101 808 notası, susturulmuş üç melodik kanal, 8 lokator.
  - Her pad, stem'lerin o sesi tuttuğu yere göre kendi Volume'unda çalıyor: −12,5 dB'lik davul fader'ının altında −12,1 ile 0 dB arası (kulakla kontrol edilmedi). En yüksek pad bir perc; çoğunlukla hat ile paylaştığı vuruşlarda okunuyor.
  - Bir ses artık burada 12 değil 5 vuruşla pad alıyor, kalıntı geçişinden (residual pass) gelen bir sesin ise ölçüden ölçüye tekrar etmesi gerekiyor. Böylece üç pad eklendi: bir perc, snare bandında bulunan hat dosyası ve snare'e benzer bir ses (her biri 7–8 nota, kulakla kontrol edilmedi).

**Replika modu, kalibrasyon v2 (48 sentetik beat, 22 senaryo):**
- Puanlar katı: ofset çıkarılmaz, bir vuruş sesin başladığı yerin 20 ms içindeyse sayılır.
- Sayılar senaryo ortalamaları, htdemucs stem'leriyle; replika modundan önceki kod → 29.10.2026 düzeltmeleriyle replika modu (m4 etiketi: m3'e pad başına seviye ve şablonun perdesini izleyen 808 one-shot'ı eklenmiş hali). "Analiz" boru hattının duyduğu, "plan" Live'ın çalacağı şey (Live dışında, offline çalınıyor).
- **Davul vuruşları ve 808 başlangıçları, F1@20:** analizde 0,368 → 0,888, planda 0,282 → 0,876.
- **Davullar:** 0,397 → 0,883 (analiz), 0,283 → 0,868 (plan).
  - Kick 0,25 → 0,88, snare 0,70 → 0,96, hat 0,51 → 0,86 (analiz).
  - Kick zamanlaması (medyan): +16,6 ms → +0,1 ms.
- **808 notaları (50 ms, 50 cent), F1:** 0,278 → 0,936 (analiz), 0,229 → 0,934 (plan). Perde hatasının medyanı 55 → 6 cent (analiz), 70 → 4,5 cent (plan). Yazılan nota 0,970 oranında doğru.
- **Beat / downbeat F:** 0,72 / 0,64 → 0,96 / 0,94.
- **Melodik notalar:** F1 0,63. **Akorlar (majmin):** 0,98 (4 beat).
- **Pad'ler kanıta göre veriliyor:** bir sesin pad alması için en az 3 vuruşu (uzun bir şarkıda 16 ölçüde bir vuruş), temiz bir şablonu ve kütüphanede yakın bir eşleşmesi ya da beat'in omurgasında (backbone) bir yeri olmalı. Kalıntı geçişinden gelen bir ses ayrıca ölçüden ölçüye tekrar etmeli.
  - Eski 12 vuruş eşiğiyle açık hat planda 0,043'tü, şimdi 0,981.
  - Planın yanlış vuruşları beat başına 23,0'dan 9,6'ya düştü (bu düzeltmelerden önceki birleşik kodla, m2g-vel, karşılaştırıldığında).
  - Her davul notası, pad'deki sample'ın kendi atağı kadar erken başlıyor.
- **Her davul pad'i kendi seviyesinde çalıyor** (m3 → m4). Bir pad'in seviyesi, stem'lerin o sesi tuttuğu yerden okunuyor: her vuruşun atağından itibaren 50 ms boyunca, sesin rolüne ait bantta, stem'ler pad'in 0 dB'deki kendi kanalına karşı; vuruşların medyanı. Kick, davul ve bas stem'lerinin toplamından planın kendi 808'i çıkarılarak okunuyor, çünkü Demucs kick'in bir kısmını bas stem'ine veriyor; diğer sesler davul stem'inden. En yüksek pad 0 dB'de, diğerleri onun altında: süitte −19,7 ile 0 dB arası.
  - Kusursuz bir kitten uzaklık (medyan) 4,09 → 1,25 dB, denge 3,03 → 0,55 dB (htdemucs_ft'de 3,70 → 1,23 dB ve 2,84 → 0,50 dB). Pad'ler kanıta göre verilmeden önce, m2g-vel'de, 3,67 ve 2,24 dB'ydi; replika modundan önce 7,18 dB.
  - Her pad 0 dB'deyken hat'ler kusursuz kitin 6,2 dB, açık hat'ler 11,6 dB üstündeydi; şimdi 1,1 ve 0,09 dB altında.
  - Açık konu: kick ile snare arasındaki denge. Kusursuz kite göre beat başına |kick − snare| farkının senaryo ortalaması 1,45 dB (m3'te 4,67; htdemucs_ft'de 1,16, m3-ft'de 2,86), ama 42 beat'in 2'sinde m3'tekinden uzak (htdemucs_ft'de 43'ün 6'sı). bass_low-2'de kick ve hat pad'leri birbirinin dosyasını çalıyor; tempo95-2'nin kick'i yeni one-shot'ın altında 1,7 dB düşük okunuyor.
- **808 one-shot'ı kopyaladığı 808'i izliyor** (m3 → m4). Fit edilen sesin zarfını ve distorsiyonunu koruyor, ama şablonun perde düşüşü oturana kadar şablonla aynı yerlerde sıfırdan geçiyor ve atağı, şablonun başlangıcından sonraki atağıyla aynı yerde. Fit edilen ses tek başına kopyaladığı 808'lerden hızlı düşüyordu; tek bir Detune notaları uzunluklarına göre farklı sapmayla çalıyordu: bass_low-1 medyan 41,7 cent sapıyordu.
  - Planın 808 nota F1'i 0,921 → 0,934 (htdemucs_ft 0,910 → 0,934), perde hatasının medyanı 8,3 → 4,5 cent (13,6 → 9,6), bass_low-1'in nota F1'i 0,72 → 1,00 (0,45 → 0,98).
  - full ise düştü: 0,890 → 0,851 (0,910 → 0,890). Analiz notalarından birini yarım ton tiz yazıyor ve 5 kısa notayı 41 cent pes duyuyor; m3'ün one-shot'ı bu iki hatayı tesadüfen telafi ediyordu.
- **htdemucs varsayılan kalıyor:** htdemucs_ft'nin davulları daha iyi (0,895 / 0,898), ama analizdeki 808 notaları ve snare'leri daha kötü.
- **Live'da doğrulanmadı:** pad başına Detune ve Volume, 808'in one-shot'ı, Detune ve Glide ayarları ve kurulumun bütünü henüz Live'da kaydedilip ölçülmedi. 0 dB'nin altında bir pad Volume'u da, şablonun perdesini izleyen one-shot da henüz Live'da kurulmadı.

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
