# LeafScan

**Open the app:** https://auctiaesth.github.io/leafscan/

LeafScan identifies plant diseases from a photo of a leaf. It recognizes 120 diseases on 34 plants, plus
healthy leaves of 30 plants, and explains each result with summaries from published expert sources and the
scientific names of the plant and the cause.

- Works on phones, tablets and computers in any modern browser; install it with *Add to Home Screen*.
- Runs entirely on your device: photos are never uploaded.
- Works offline after the first visit (about 32 MB download).
- Move and pinch the photo under a fixed square to choose the leaf. LeafScan checks the photo once when it
  opens, then again only when you tap **Check this leaf** (or, if you turn it on, when you let go). The page
  never scrolls by itself, and each new check shows how the confidence changed.
- Confidence is shown as a word, a percentage and how often results that sure were right on 3,100 farm
  photos (about 9, 7 or 4 in 10).
- Warns about blurry or dark photos, and can combine up to 3 leaves of the same plant into one result.
- When a photo is uncertain, it asks yes/no questions about visible symptoms to narrow down the result, and
  hides chemical treatments until the diagnosis is confident.

This is a student research prototype. Results can be wrong. Confirm any diagnosis with an agricultural
technician or a plant clinic (for example the UPLB Pest Clinic, which accepts samples by courier, or a DA
Regional Crop Protection Center) before spraying or removing plants, and use only pesticides registered with
the Fertilizer and Pesticide Authority, following the label.

## How it works

A MobileNetV3-Large image recognition model (17.9 MB, ONNX) runs in the browser with ONNX Runtime Web. Each
result links to a record in the knowledge base (`data/kb.json`) with symptoms, signs to check, management
steps and the expert sources they were summarized from.

## Credits and licenses

Non-commercial use only, because of the licenses of the data and knowledge sources below.

- **Training data:** PlantVillage (Hughes & Salathé, 2015); PlantSeg, CC BY 4.0, and PlantWild, CC BY-NC-ND 4.0
  (Wei et al., 2024); PlantDoc, CC BY 4.0 (Singh et al., 2020), used for testing only. No dataset images are
  included here.
- **Knowledge base:** summaries written from the PlantVillage disease library (Penn State University), CABI
  Plant Health Knowledge for AI / PlantwisePlus Pest Management Decision Guides (CC BY-NC-SA 4.0), and
  university extension and government fact sheets. Every record cites and links its sources. The summaries
  are shared under CC BY-NC-SA 4.0 and have not yet been reviewed by a plant pathologist.
- **Scientific names:** checked against the GBIF Backbone Taxonomy.
- **Model:** MobileNetV3 (Howard et al., 2019) with ImageNet weights from torchvision.
- **ONNX Runtime Web 1.29.0:** MIT License; see `THIRD_PARTY_NOTICES.md`.
