# LeafScan: copyright and use

Copyright © 2026 the LeafScan student research team.

LeafScan is a student research project ("Developing a Prototype for Plant Disease Detection Using Image
Recognition"). This repository is public so that the app can be used; it is **not** open source.

**You may** use the app at https://auctiaesth.github.io/leafscan/ and link to it.

**You may not**, without written permission from the team:
- copy, modify, republish or redistribute the app's code, design or the texts the team wrote;
- present LeafScan, or work based on it, as your own, for example in a school project, research paper,
  competition or app store.

To ask permission, open an issue on this repository.

## The trained model

The image recognition model (`model/`) was trained on photos that other people published under the licences
below. The team claims no rights over the model beyond what those licences allow, does not include any of the
training photos here, and offers the model only as part of this non-commercial app. Whether a trained model counts
as an adaptation of its training photos is not settled; anyone who wants to reuse the model must check these
licences themselves (and, for PlantWild, ask its authors).

| Training data | Licence |
|---|---|
| PlantVillage (Hughes & Salathé, 2015) | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), as stated on its dataset card |
| PlantSeg v3 (Wei et al., 2024a) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| PlantWild v1 and v2 (Wei et al., 2024b) | [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) |
| GBIF photos (snake plant and other plants) | CC0 1.0, CC BY 4.0 or CC BY-NC 4.0; each photo credited in `credits/gbif_photos.csv` |
| PlantDoc (Singh et al., 2020), used only for testing | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

The photos in PlantSeg and PlantWild were collected from websites; their original photographers keep their rights.

## Other parts under other licences

- **Knowledge-base summaries** (`data/kb.json`) are shared under
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), because some of their sources (CABI
  PlantwisePlus) require it. Every record cites its sources.
- **ONNX Runtime Web** (`vendor/`) is under the MIT License, and the **Spectral** typeface (`fonts/`) under the
  SIL Open Font License 1.1; see `THIRD_PARTY_NOTICES.md`.
