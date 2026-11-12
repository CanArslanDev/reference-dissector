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

**PIRAMIT** (replika moduyla, 29.10.2026 çalıştırması; `fit-808` 74931ce'nin koduyla, `plan` ise ilk Live kontrolünün düzeltmelerini birleştiren a4f2358'in koduyla yeniden çalıştırıldı; replika modunun ilk çalıştırması 22.10.2026'daydı):
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
- **808:** 101 nota, 87'si C1. Artık hiçbir nota sessizliğin üzerinden tutulmuyor: 22.10.2026'da tek bir nota D ve E bölümleri boyunca tutuluyordu; arada iki sessizlik (808 bandı, notaların medyan tepesinin 40 dB altında) ve 808'in D'de tek başına çaldığı notalar vardı.
  - Bir nota artık 808'in sustuğu yerde bitiyor, 9 dB'lik bir yükseliş de yeni bir nota başlatıyor: D bölümünde dört C1 notası var.
  - Stem'de orada beşinci bir atak daha görünüyor. Çalmakta olan notanın üzerine yalnızca 6,5 dB yükseldiği için hâlâ o notanın içinde kalıyor (kulakla kontrol edilmedi).
  - Sentezlenen ses dalga biçimi ve zarfta kütüphanedeki en iyi 808'den daha yakın (NCC: sentez 0,938, kütüphane 0,894; zarf sapması: 1,3 ve 1,4 dB). Üst harmoniklerde (distortion) ise kütüphanedeki 808 daha yakın (sapma: kütüphane 8,7 dB, sentez 12,9 dB).
  - Planın çaldığı one-shot artık, perde düşüşü oturana kadar (269 ms) şablonun kendi perdesini izliyor. 120–250 ms'lik C1 notaları m3 kodunun one-shot'ıyla 41 centlik bir Detune düzeltmesi istiyordu, şimdi −0,1 cent. Plan çaldığında 101 notanın 91'i analizin duyduğu perdenin 20 cent yakınında (önce 82). Fit referansına karşı dalga biçimi NCC'si 0,921'den 0,880'e düştü (kulakla kontrol edilmedi).
- **Bölümler:** A 1–24 · B 25–28 · C 29–32 · A 33–47 · D 48–51 · E 52–56 · A 57–71 · D 72–73. D, 808'in tek başına çaldığı yer; E, breakdown. 22.10.2026'da tutulan 808 notası ikisini tek bir bölümde (D 49–56) birleştiriyordu.
- **Live:** Live 12.4.6'da kuruldu ve kaydedildi (29.10.2026, ikinci Live kontrolü; plan 5 `build_tracks` çağrısıyla gönderildi).
  - 11 Simpler'ın hepsi planın dosyasını tutuyordu, 1.881 nota gönderildiği gibi geri okundu, 630 davul vuruşu ve 101 808 notası `render.py`'nin çaldığı yerin 0,04 ve 0,11 ms yakınında çaldı, 8 lokatorun hepsi planlanan yerdeydi. Kulakla kontrol edilmedi; şarkıya karşı ölçülecek bir doğru cevap da yok.
  - 10 Drum Rack pad'inde 630 davul notası (22.10.2026'da 7 pad'de 607), Transpose +36 ve Detune +1 ile 101 808 notası, susturulmuş üç melodik kanal, 4/4 ölçü, 8 lokator.
  - Live, bir pad'in hâlâ çalan vuruşunu kesen yeni vuruşu 1 ms geç başlatıyor; plan bu notaları 1 ms erken yazıyor: 630 davul notasının 98'i, 101 808 notasının 62'si (808'de aynı perdedeki notayı yeniden başlatanlar).
  - Her pad, stem'lerin o sesi tuttuğu yere göre kendi Volume'unda çalıyor: −12,6 dB'lik davul fader'ının altında −12,1 ile 0 dB arası (kulakla kontrol edilmedi). En yüksek pad bir perc; çoğunlukla hat ile paylaştığı vuruşlarda okunuyor.
  - İlk Live kontrolünde 74931ce'nin planı bir pad'de durdu: Live'ın tarayıcı (browser) araması tek karakterlik bir dosya adını bulamadı. Başka bir pad de aynı adı taşıyan bir FX sesini çaldı. Artık her pad'in dosyası, içeriğinden türetilen bir adla yükleniyor ve kurulumdan sonra geri okunuyor.
  - Bir ses artık burada 12 değil 5 vuruşla pad alıyor, kalıntı geçişinden (residual pass) gelen bir sesin ise ölçüden ölçüye tekrar etmesi gerekiyor. Böylece üç pad eklendi: bir perc, snare bandında bulunan hat dosyası ve snare'e benzer bir ses (her biri 7–8 nota, kulakla kontrol edilmedi).

**Replika modu, kalibrasyon v2 (48 sentetik beat, 22 senaryo):**
- Puanlar katı: ofset çıkarılmaz, bir vuruş sesin başladığı yerin 20 ms içindeyse sayılır.
- Sayılar senaryo ortalamaları, htdemucs stem'leriyle; replika modundan önceki kod → 29.10.2026 düzeltmeleriyle replika modu (m5 etiketi). m4, m3'e pad başına seviye ve şablonun perdesini izleyen 808 one-shot'ı eklenmiş hali; m5 de m4'e ilk Live kontrolünün düzeltmelerinin eklenmiş hali: her pad tek ses (voice) olarak çalıyor, 808 Live'ın 50 ms'lik release'iyle bitiyor, plan ölçüyü taşıyor ve pad'ler içerikleriyle yükleniyor. "Analiz" boru hattının duyduğu, "plan" Live'ın çalacağı şey (Live dışında, offline çalınıyor).
- **Davul vuruşları ve 808 başlangıçları, F1@20:** analizde 0,368 → 0,888, planda 0,282 → 0,876.
- **Davullar:** 0,397 → 0,883 (analiz), 0,283 → 0,868 (plan).
  - Kick 0,25 → 0,88, snare 0,70 → 0,96, hat 0,51 → 0,86 (analiz).
  - Kick zamanlaması (medyan): +16,6 ms → +0,1 ms.
- **808 notaları (50 ms, 50 cent), F1:** 0,278 → 0,936 (analiz), 0,229 → 0,934 (plan). Perde hatasının medyanı 55 → 6 cent (analiz), 70 → 4,4 cent (plan). Yazılan nota 0,970 oranında doğru.
- **Beat / downbeat F:** 0,72 / 0,64 → 0,96 / 0,94. Planın ölçüsü senaryoların 0,955'inde doğru (m4'te 0,909): meter34 artık 3/4 planlanıyor.
- **Melodik notalar:** F1 0,63. **Akorlar (majmin):** 0,98 (4 beat).
- **Pad'ler kanıta göre veriliyor:** bir sesin pad alması için en az 3 vuruşu (uzun bir şarkıda 16 ölçüde bir vuruş), temiz bir şablonu ve kütüphanede yakın bir eşleşmesi ya da beat'in omurgasında (backbone) bir yeri olmalı. Kalıntı geçişinden gelen bir ses ayrıca ölçüden ölçüye tekrar etmeli.
  - Eski 12 vuruş eşiğiyle açık hat planda 0,043'tü, şimdi 0,981.
  - Planın yanlış vuruşları beat başına 23,0'dan 9,6'ya düştü (bu düzeltmelerden önceki birleşik kodla, m2g-vel, karşılaştırıldığında).
  - Her davul notası, pad'deki sample'ın kendi atağı kadar erken başlıyor.
- **Her davul pad'i kendi seviyesinde çalıyor** (m3 → m4). Bir pad'in seviyesi, stem'lerin o sesi tuttuğu yerden okunuyor: her vuruşun atağından itibaren 50 ms boyunca, sesin rolüne ait bantta, stem'ler pad'in 0 dB'deki kendi kanalına karşı; vuruşların medyanı. Kick, davul ve bas stem'lerinin toplamından planın kendi 808'i çıkarılarak okunuyor, çünkü Demucs kick'in bir kısmını bas stem'ine veriyor; diğer sesler davul stem'inden. En yüksek pad 0 dB'de, diğerleri onun altında: süitte −19,7 ile 0 dB arası.
  - Kusursuz bir kitten uzaklık (medyan) 4,09 → 1,25 dB, denge 3,03 → 0,55 dB (htdemucs_ft'de 3,70 → 1,23 dB ve 2,84 → 0,50 dB). Pad'ler kanıta göre verilmeden önce, m2g-vel'de, 3,67 ve 2,24 dB'ydi; replika modundan önce 7,18 dB.
  - Her pad 0 dB'deyken hat'ler kusursuz kitin 6,2 dB, açık hat'ler 11,6 dB üstündeydi; m4'te 1,1 ve 0,09 dB altındaydılar (m5'te hat'ler 0,90 dB altında, açık hat'ler 0,15 dB üstünde).
  - Açık konu: kick ile snare arasındaki denge. m4'te kusursuz kite göre beat başına |kick − snare| farkının senaryo ortalaması 1,45 dB (m3'te 4,67; htdemucs_ft'de 1,16, m3-ft'de 2,86), ama 42 beat'in 2'sinde m3'tekinden daha kötü (htdemucs_ft'de 43'ün 6'sı). bass_low-2'de kick'in vuruşları hat'in dosyasını çalıyor: analiz kick sesine hat'in dosyasını, hat'lerine (29 ms erken bulunmuş) ise iki sesin de olmayan bir kick dosyasını verdi. tempo95-2'de kick m3'teki yerinde çalıyor, snare ise 2,2 dB daha yüksek. Kick'i yeni one-shot'ın altında m3'tekinden 1,7 dB düşük okunuyor: kick'in Volume'u o kadar düştü, RMS ile eşlenen davul fader'ı da yaklaşık o kadar yükselip diğer pad'leri yükseltti.
- **808 one-shot'ı kopyaladığı 808'i izliyor** (m3 → m4). Fit edilen sesin zarfını ve distorsiyonunu koruyor, ama şablonun perde düşüşü oturana kadar şablonla aynı yerlerde sıfırdan geçiyor ve atağı, şablonun başlangıcından sonraki atağıyla aynı yerde. Fit edilen ses tek başına kopyaladığı 808'lerden hızlı düşüyordu; tek bir Detune notaları uzunluklarına göre farklı sapmayla çalıyordu: bass_low-1 medyan 41,7 cent sapıyordu.
  - Planın 808 nota F1'i 0,921 → 0,934 (htdemucs_ft 0,910 → 0,934), perde hatasının medyanı 8,3 → 4,5 cent (13,6 → 9,6), bass_low-1'in nota F1'i 0,72 → 1,00 (0,45 → 0,98).
  - full ise düştü: 0,890 → 0,851 (0,910 → 0,890). Analiz notalarından birini yarım ton tiz yazıyor ve 5 kısa notayı 41 cent pes duyuyor; m3'ün one-shot'ı bu iki hatayı tesadüfen telafi ediyordu.
- **Live'ın ölçülen çalma davranışı artık planda** (m4 → m5). İlk Live kontrolü şunları buldu: bir pad, hâlâ çalan kendi vuruşunu kesiyor ve yeni vuruşu 1 ms geç başlatıyor; 808 her notayı 50 ms'lik bir release ile bitiriyor; 3/4'lük bir beat 4/4 kuruluyordu. `render.py` ve tahminler artık her pad'i tek ses olarak, 808'i de release'i ve yeniden başlayışıyla çalıyor; Live'ın geç başlattığı notalar 1 ms erken yazılıyor ve plan ölçüyü taşıyor.
  - Kusursuz kitten uzaklık 1,25 → 1,20 dB, denge 0,55 → 0,51 dB (htdemucs_ft'de 1,23 → 1,15 ve 0,50 → 0,48). Bütün F1'ler aynı.
  - Kötüleşen: dynamics, kitten 1,27 → 2,16 dB. Pad'ler tek ses olarak çalınca dynamics-2'nin davul kanalı 0 dB'de 1,3 dB daha kısık çalıyor, RMS ile eşlenen fader'ı da o kadar yükseliyor.
- **htdemucs varsayılan kalıyor:** htdemucs_ft'nin davulları daha iyi (0,895 / 0,898), ama analizdeki 808 notaları ve snare'leri daha kötü.
- **Live'da** (Live 12.4.6, 29.10.2026): iki kontrol, planları kurulumun kendi çağrılarıyla kurup kaydetti. Dört m5 planında Live her davul vuruşunu `render.py`'nin 0,04 ms, her 808 notasını 0,11 ms yakınında çaldı. Kayıttan okunan rol başına davul F1@20'si `render.py`'ninkiyle aynı. Seviyeler yalnızca Simpler'ın tizlerde kaybettiği kadar farklı; `render.py` bunu modellemiyor (hat'lerde 0,97 dB'ye kadar). 0 dB'nin altındaki pad Volume'ları (0,05 dB'lik bir gösterim adımı içinde), Detune, "Vol < Vel" ve şablonun perdesini izleyen one-shot, planın varsaydığı gibi çalıyor.
- **808'in kaymaları (slide), istenirse** (`--slides`; g1 ve g1-ft etiketleri, 05.11.2026). Yukarıdaki sayılar varsayılanın; varsayılan legato nota yazmıyor ve m5'le aynı. `--slides` ile okuyucu bir kaymayı 808 bandının perdesinden okuyor: bir tutuş, bir tam transpozisyona düz bir geçiş, yine bir tutuş ve yeniden başlama yok; geçişin altındaki bir kick ya da bandı yükselmeyen bir şablon uyumu onu artık atak yapmıyor.
  - Doğrunun legato notalarından bulunanlar: eşiklerin seçildiği seed 1'de iki modelde de 10'un 8'i, hiçbiri yanlış değil, doğrunun başlangıcından medyan 3,4 ms uzakta (htdemucs_ft 4,8); kural tasarlanırken kayma beat'leri görülmüş olan seed 2'de 14'ün 4'ü (htdemucs_ft'de 14'ün 3'ü). Süitin kitleri ve yeni pattern'lerle yapılmış glide beat'lerinde, seed 1'in kitini kullanan yarıda 18'in 7'si (8'i), kit ve pattern olarak dışarıda tutulan (held-out) yarıda 22'nin 5'i (6'sı); yeni kitlerle 52'nin 14'ü (19'u).
  - 48 beat üzerinde 808 nota F1@50: analizde 0,9360 → 0,9391 (htdemucs_ft 0,9245 → 0,9263), planda 0,9342 → 0,9377 (0,9345 → 0,9360); glides 0,791 → 0,851 (0,839 → 0,884). Her kazancın yarısı seed 1'de (örneklem içi); yalnız glides-2: 0,7037 → 0,7652 (0,8000 → 0,8319). glides-1 ve glides-2'nin planları Glide'ı açıyor.
  - Kapalı kalıyor: kayması olmayan beat'ler yanlış legato notalar alıyor (htdemucs'ta bass_low-2, meter68-2, swing66-2 ve tempo160-2; htdemucs_ft'de bass_low-2 ve üç notayla swing66-2, ki onun planı da Glide'ı açıyor); 808'in başlangıç hatasının p90'ı (7,98 → 8,03 ms) ve htdemucs_ft'nin seviyeleri (kusursuz kitten 1,1516 → 1,1536 dB) biraz kötüleşiyor. PIRAMIT'te iki durumda da kayma yok.
  - Live'da (05.11.2026) glides-1'in Glide'lı planından 8 ölçülük bir parça, kaymalarını `render.py`'nin 1,21 cent (p90) yakınında çaldı; 808'in `render.py`'den farkı render'ın 31,5 dB altında. 808 okuyucusu bu kaymalardan Live'ın sesinde 3'ünü, `render.py`'ninkinde 1'ini buluyor (ilk raporda yanlış anahtarı okuyan bir betik yüzünden hiç bulmadığı yazılmıştı).
- **Live'da doğrulanmadı:** `refdissect live-build`'in kendisi uçtan uca (Live'ın Remote Script'i en fazla 8 istemciye hizmet veriyor ve 8'i de doluydu; kontroller aynı çağrıları kendi oturumlarından gönderdi), htdemucs_ft planları, Glide'lı bir planın tamamı, choke grupları ve Live'ın ızgarasına oturmayan lokatorlar (Live bunları kaydırıyor, kurulum da siliyor).

**Kalibrasyon v1 (6 sentetik beat):**
- Vuruş zamanları bilindiğinde 30 sesin 27'si 8.854 sample arasında doğru bulunuyor.
- Hiçbir şey bilinmeden snare transkripsiyonu F1 1,00 ve tanıma 6/6.
- Hat ve kick daha zayıf: F1 0,77 ve 0,70.

## Yeni komutlar

- `melodic`: her tonal kanal için notalar ve akorlar (Basic Pitch).
- `index-speckle`: sample'ları herhangi bir perdede eşleştirmek için speckle indeksi. `index-library` bunu artık kendisi yazıyor; bu komut daha önce indekslenmiş bir kütüphane içindir.
- `suite build`, `suite run`, `suite report`: kalibrasyon v2'yi kurar, her etiket (tag) için çalıştırır ve puanlar. `suite run` şunları alır: `--steps`, `--reuse`, `--jobs`, `--render`, `--model`, `--resume`, `--cache`, `--subset`.
- `analyze` ve `fit-808`, 808 okuyucusunun sonucunu ve 808 fit'ini çalışma klasörünün `cache/` alt klasöründe saklar. Stem'ler (ya da şablon), ayarlar, numpy, scipy ve soxr sürümleri ve adımın `.py` dosyaları (sonucu üreten modül ve paketten içe aktardığı her modül) aynıysa sonraki çalıştırma bunları geri okur ve log'a hangi dosyayı okuduğunu yazar. Başka bir modüldeki değişiklik (örneğin `live.py`) saklananları bozmaz. `--no-cache` hepsini yeniden çalıştırır. `suite run` ise `--cache` verilmedikçe hiçbir şey saklamaz; böylece ölçülen süreler kodun kendi süreleridir.

- `suite run --subset dev`: süitin her senaryo ailesinden birer beat alan 11 beat'lik geliştirme alt kümesi; `--cache` ile denemeler için.
- `suite clean DIR --keep a,b`: tutulmayan etiketlerin çalıştırma klasörlerini boşaltır, her birinin `scores.json`'unu saklar. `--yes` verilmezse yalnızca neyin silineceğini listeler.
- `live-build --bars A-B`: planın yalnızca A–B ölçülerini, 0'dan başlatarak kurar; stem'ler aynı anlarda kesilir. Kısa bir Live kontrolü içindir; henüz Live'da çalıştırılmadı.

Komutların tamamı README.md'nin "Quick start" bölümünde, ayrıntıları docs/12'de.

## Geliştirme

`uv sync` pytest, pytest-xdist ve ruff'ı kurar. Hiçbir test kütüphaneni, Demucs'u, Live'ı ya da model indirmeyi gerektirmez.

```bash
uv run pytest -m "not slow" -n auto     # hızlı döngü: -n 4 ile 306 test 22 s
uv run pytest -n auto                   # bütün testler, her commit'ten önce ve CI'da: -n 4 ile 94 s (tek süreçte 307 s)
uv run ruff check .
```

- `slow` işaretli 48 test fonksiyonu (parametreleriyle 62 test), tek süreçte her biri 2 s ya da daha uzun sürdü: 308 s'lik test süresinin 250 s'si. Hızlı döngü bunları atlar; commit'ten önceki kontrol yine bütün testlerdir.
- `-n` ile boşta kalan bir işçi, meşgul olanın testlerini alıyor (`--dist worksteal`, pyproject'te). -n 4'te bütün testler 132 s'den 94–96 s'ye indi.
- Üzerinde çalıştığın bir bilgisayarda `-n 4` çekirdek bırakır.
- Ölçüm 10 çekirdekli Apple M4'te (04–05.11.2026): 366 geçti, 3 atlandı, 2 beklenen hata (tek süreç süresi, kütüphane deposunun 1 s'den kısa süren 3 testinden önce, 363 testle ölçüldü). CI `-n auto` ile çalışıyor: bir iş ubuntu'da 3 dk 50 s, macOS'ta 4 dk 40 s (PR #2), 4 dk 44 s ve 7 dk 55 s (PR #3), her birleştirmeden sonra main'de 2 dk 51 s ve 2 dk 14 s, sonra 6 dk 10 s ve 7 dk 23 s sürdü; tek süreçte 7,4 ve 12,6 dk (PR #1'in son çalışması). GitHub'ın makineleri çalışmadan çalışmaya değişiyor.

Boru hattında bir değişiklik denerken önce geliştirme alt kümesinde dene, sonra bütün beat'lerde ölç:

```bash
uv run refdissect suite run "$S" --library "$W/work/library.npz" --library-808 "$W/work/library_808.npz" \
    --tag try --subset dev --cache --jobs 2 --render
uv run refdissect suite report "$S" --tags <alt kümenin temel etiketi>,try
uv run refdissect suite clean "$S" --keep m5,m5-ft          # kuru çalıştırma; --yes diğer etiketlerin çalıştırma klasörlerini boşaltır
uv run --extra live refdissect live-build "$W/song/refdissect" --bars 9-16   # kısa bir Live kontrolü
```

- Geliştirme alt kümesi 11 beat'tir, her senaryo ailesinden bir tane ([03](docs/03-calibration.md#a-dev-subset)). 2 işle (job), beat'lerinin `cache/` klasörü 808 okuyucusunun sonuçlarını ve fit'leri tuttuktan sonra 260–268 s sürdü; doldurmak 672–681 s (cache'siz 673 s). Bütün süit bir model için yaklaşık 45 dakika.
- Bir alt küme çalıştırmasını aynı alt kümenin bir çalıştırmasıyla karşılaştır. Son sayılar her iki modelde de 48 beat'in tamamında ölçülür.
- `suite clean` her etiketin `scores.json` dosyasını tutar; temizlenmiş bir etiket yine beat beat karşılaştırılabilir. Süitte olmayan bir etiketi `--keep` ile tutmak istersen reddeder.
- `live-build --bars A-B` bir planın A'dan B'ye kadar olan ölçülerini 0'dan kurar, stem'ler de aynı yerden kesilir ([07](docs/07-building-in-live.md#a-short-live-check)). Henüz Live'da çalıştırılmadı.
- Kütüphanenin dizileri bellek eşlemeli (memory-mapped) okunuyor; bir çalıştırmanın işleri tek kopyayı paylaşıyor. Depo, .npz'nin boyutu ve CRC-32'leri ile her .npy'nin boyutu ve değişiklik zamanı yazıldığı gibi kaldıkça kullanılır; değişmiş, eksik ya da kısalmışsa yeniden yazılır. 2 işte her işin en yüksek fiziksel belleği 1,7–1,8 GB'tan 1,0 GB'a indi, süre aynı kaldı ([12](docs/12-replica-mode.md#the-suite)).
- Bunların hiçbiri bir çıktıyı değiştirmiyor: geliştirme alt kümesinin 11 beat'inde bu kodla yapılan taze bir çalıştırmanın `log.txt` dışındaki her dosyası, süreler ve çalıştırma kayıtları dışında, m5'inkiyle ve m5-ft'ninkiyle aynı.

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
