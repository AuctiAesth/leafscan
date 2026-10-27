# LeafScan

> **© 2026 the LeafScan student research team.** This repository is public so the app can be used, but it is
> not open source: please do not copy, republish or present LeafScan as your own work. See [LICENSE.md](LICENSE.md)
> for what is allowed, and for the licences of the training data, the model and the other parts.

**Open the app:** https://auctiaesth.github.io/leafscan/

LeafScan identifies plant diseases from a photo of a leaf. It recognizes 120 diseases on 34 plants, plus
healthy leaves of 31 plants (including the snake plant common in Philippine classrooms), sets aside common
ornamentals and garden plants it cannot check, and explains each result with summaries from published expert sources and the
scientific names of the plant and the cause.

- Works on phones, tablets and computers in any modern browser; install it with *Add to Home Screen*.
- Runs entirely on your device: photos are never uploaded.
- Works offline after the first visit (about 8 MB download). The model downloads in small checked parts, so a
  dropped connection continues where it stopped, and app updates never download the model again.
- Move and pinch the photo under a fixed square to choose the leaf. LeafScan checks the photo once when it
  opens, then again only when you tap **Check this leaf** (or, if you turn it on, when you let go). The page
  never scrolls by itself, and each new check shows how the confidence changed.
- Each photo is checked three ways (the square, the square mirrored, the whole photo) and the answers are
  combined. Confidence is shown as a word, a percentage and how often results that sure were right on 3,583
  test field photos from public datasets (about 9, 7 or 4 in 10; not yet Philippine photos).
- Cucumber, squash and zucchini leaves look alike, so without a chosen plant LeafScan names the problem for
  the whole squash family (right answers on those photos rose from 63% to 79%).
- If LeafScan guessed the wrong plant, choose the right one: it then ranks only that plant's problems. On the
  same test photos this raised accuracy from 77% to 91%.
- Says "Not recognized" instead of guessing when the photo may not be a leaf or a plant it knows.
- Study mode (for the research team) logs each photo with the study's photo-log fields and exports a ZIP with
  the photos and `photo_log.csv`; nothing leaves the phone until you export it.
- **Philippine information:** results add Philippine notes, local names (for example PhilRice's regional rice
  disease names), Philippine steps and quarantine rules, from PhilRice, DA regional offices, ATI, BPI, PCA,
  DOST-PCAARRD and UPLB. The Plants list and results also describe 17 common Philippine problems LeafScan cannot
  recognize from photos yet (such as bacterial wilt, rice tungro and bacterial leaf blight, corn downy mildew,
  banana bugtok and bract mosaic, kamote kulot and coconut cadang-cadang), and the Guide lists Philippine rules
  for treating plants and the Filipino and Cebuano words extension guides use for symptoms.
- Choosing Squash (kalabasa) also considers the zucchini classes (same genus), so squash viruses can be named.
- Warns about blurry or dark photos, and can combine up to 3 leaves of the same plant into one result.
- When a photo is uncertain, it asks up to 4 yes/no questions about visible symptoms, one at a time, and
  hides chemical treatments until the diagnosis is confident.
- "Show to a technician" puts the leaf photos, results and answers on one screen to share or copy.
- "Was this right?" answers are kept on the device, so you can see how often LeafScan was right for you.
- The model runs in a background worker, so moving the photo stays smooth on slow phones.

This is a student research prototype. Results can be wrong. Confirm any diagnosis with an agricultural
technician or a plant clinic (for example the UPLB Pest Clinic, which accepts samples by courier, or a DA
Regional Crop Protection Center; BPI's Crop Pest Management Division diagnoses samples free of charge) before
spraying or removing plants. Use only products registered with the Fertilizer and Pesticide Authority, or with
DA-BAFS for organic and biological products, following the label, and do not burn sick plants in the open
(RA 9003).

## How it works

A MobileNetV3-Large image recognition model runs in the browser with ONNX Runtime Web: the study model is 17.9 MB;
the app ships a newer model (v3) with 8-bit weights (4.9 MB) that adds snake plant and an "other plant"
class. Each
result links to a record in the knowledge base (`data/kb.json`) with symptoms, signs to check, management
steps and the expert sources they were summarized from.

## Credits and licenses

The app's code, design and the team's own texts are © 2026 the LeafScan student research team (see
[LICENSE.md](LICENSE.md)). The model was trained on the data below and the team claims no rights over it beyond
what those licenses allow. The data and knowledge sources keep their own licenses; because of them, any permitted
use is non-commercial only.

- **Training data:** PlantVillage, CC BY-SA 3.0 (Hughes & Salathé, 2015); PlantSeg, CC BY 4.0, and PlantWild,
  CC BY-NC-ND 4.0 (Wei et al., 2024); PlantDoc, CC BY 4.0 (Singh et al., 2020), used for testing only; for the v3 model, 3,223
  snake plant and ornamental-plant photos from GBIF.org (mostly iNaturalist observers) under CC0, CC BY 4.0 or
  CC BY-NC 4.0, credited individually in `credits/gbif_photos.csv`. No dataset images are included here. The
  photos in PlantSeg and PlantWild were collected from websites, and their original photographers keep their
  rights. None of PlantSeg's photos comes from a Philippine website, which is why the study tests LeafScan on
  Philippine plants in the field.
- **Knowledge base:** summaries written from the PlantVillage disease library (Penn State University), CABI
  Plant Health Knowledge for AI / PlantwisePlus Pest Management Decision Guides (CC BY-NC-SA 4.0), and
  university extension and government fact sheets. Every record cites and links its sources. The summaries
  are shared under CC BY-NC-SA 4.0 and have not yet been reviewed by a plant pathologist.
- **Scientific names:** checked against the GBIF Backbone Taxonomy.
- **Model:** MobileNetV3 (Howard et al., 2019) with ImageNet weights from torchvision.
- **ONNX Runtime Web 1.29.0:** MIT License; see `THIRD_PARTY_NOTICES.md`.
- **Spectral typeface:** SIL Open Font License 1.1; see `fonts/OFL.txt`.
