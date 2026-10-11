# reference-dissector (Türkçe özet)

Bir referans şarkıyı parçalarına ayırıp Ableton Live'da yeniden kurar. Adımlar:

1. Tempo gridini ve akordu bulur.
2. Her davul vuruşunu ve 808 notalarını çıkarır.
3. Kütüphanendeki aynı sample'ları ya da en yakınlarını tanır.
4. 808'i sentezler.
5. Sonucu, A/B için orijinal kanallar altta susturulmuş olarak, [ableton-live-mcp](https://github.com/CanArslanDev/ableton-live-mcp) ile Live'a kurar.

Çıkış sorusu şuydu: *Duyamayan bir yapay zekâ, koyduğu sesin şarkıdaki ses olup olmadığını nasıl anlar?* Dinlemeyi okuyabileceği ölçümlere (sayılar ve görseller) çevirip bu ölçümleri cevabı bilinen örneklerle kalibre ederek. Bu repo kodu, ölçümleri ve yol boyunca öğrenilenleri içerir. Örnek şarkı Mavi – *PIRAMIT*; ayrıca kalibrasyon için üretilmiş sentetik beat'ler kullanıldı.

## Öne çıkan sonuçlar

**PIRAMIT:**
- **Tempo:** 137,00 BPM. Beat tracker'ın 136,36'sı şarkı sonunda 49 ms kayıyor.
- **Akort:** +0,28 yarım ton (A4 ≈ 447 Hz). Prototipteki 808'in "C mi C# mı" karışıklığı buradan geliyor; davullar ise akortsuz.
- **Hi-hat:** Kütüphanedeki bir sample'ın kendisi.
  - Benzerlik (NCC) 0,981.
  - 189 vuruşun 187'sinde en iyi eşleşme o.
  - Referanstan çıkarılınca kalan kısım −14,3 dB.
- **Çift ölçülerin snare'i:** Kütüphanedeki bir sample'ın 1 yarım ton aşağı çekilmiş hali.
  - Benzerlik 0,953.
  - 27/27 vuruşta en iyi eşleşme.
- **Tek ölçülerin snare'i:** Kütüphanede yok; en yakını 0,47.
- **808:** 126 nota, %83'ü C1. Sentezlenen ses dalga biçimi ve zarfta kütüphanedeki en iyi 808'den daha yakın. Harmoniklerde (distortion) ise kütüphanedeki 808 daha yakın.

**Kalibrasyon (6 sentetik beat):**
- Vuruş zamanları bilindiğinde 30 sesin 27'si 8.854 sample arasında doğru bulunuyor.
- Hiçbir şey bilinmeden snare transkripsiyonu F1 1,00 ve tanıma 6/6.
- Hat ve kick daha zayıf: F1 0,77 ve 0,70.

## Belgeler

Ayrıntılar `docs/` altında, İngilizce:
- boru hattı;
- PIRAMIT vaka çalışması;
- kalibrasyon;
- sample tanıma;
- 808 sentezi;
- "duyamayan model nasıl dinler";
- Live kurulumu;
- tuzaklar;
- telif;
- yol haritası.

## Önemli

Şarkılar, stem'ler, şarkıdan kesilen vuruşlar, transkripsiyonlar ve kütüphane dosya adları repoya girmez; senin çalışma klasöründe kalır. Şarkıdan kesilen sesler yalnızca kişisel çalışma içindir.
