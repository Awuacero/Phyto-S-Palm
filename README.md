## License and Copyright

**Copyright (c) 2026 Román-Mateo A., Castro, A.J., et al. All Rights Reserved.**

This repository and its contents are currently closed-source. The code and web application are provided publicly for peer-review and visualization purposes only, in association with an unpublished manuscript. 

**No permission is granted to use, copy, modify, or distribute this software.** 

Once the associated scientific paper is officially published, this repository will be updated with an open-source license.

*Note: This project integrates third-party components (MusiteDeep) which are distributed under the MIT License. See the `CREDITS.md` file for full details and attributions.*

# Phyto-S-Palm 🌿

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://phyto-s-palm.streamlit.app/)

A biophysical pre-filtering pipeline (UHS & Kyte-Doolittle) integrated with an optimized MusiteDeep ensemble model for plant S-acylation prediction.

**🌐 Try the Web App:** [https://phyto-s-palm.streamlit.app/](https://phyto-s-palm.streamlit.app/)

## Overview

Characterizing the plant S-acylome presents significant challenges due to the high false-positive rates inherent to chemical enrichment methods (such as the Biotin-Switch assay) and the scarcity of predictive models optimized specifically for plant proteomes. 

**Plant-S-Acyl-Predictor** overcomes these limitations by acting as a high-stringency "digital negative control". This tool couples sequence-based deep learning architectures with biophysical surface accessibility filters, allowing researchers to purify raw experimental mass spectrometry datasets without the need for double biological sample inputs.

## Features

Built as an interactive Streamlit application, this pipeline executes a multi-layered filtration process:

*   **Sequence Pre-filtering:** Parses uploaded `.fasta` files and automatically discards any sequences lacking target Cysteine ('C') substrates.
*   **Biophysical Accessibility Analysis:** Utilizes a sliding window algorithm to evaluate local surface exposure of cysteines. It strictly filters candidates using two complementary scales: the Kyte-Doolittle hydropathy scale and the Unified Hydrophobicity Scale (UHS).
*   **Deep Learning Integration:** Structurally viable candidates with solvent-exposed cysteines are then seamlessly passed to a locally executed MusiteDeep model for S-palmitoylation site prediction.
*   **Data Export:** Generates downloadable CSV files with the high-confidence accessible cysteines and full textual reports of the MusiteDeep predictions.

## Usage (Web App)

The easiest way to use the pipeline is through our hosted Streamlit application. No installation is required.

1. Navigate to [Phyto-S-Palm on Streamlit](https://phyto-s-palm.streamlit.app/).
2. Upload your `.fasta` file.
3. Adjust the sliding window size and the maximum hydrophobicity cutoffs (KD and UHS).
4. Download the filtered sequence or run the deep learning prediction directly.

---

## Installation (Local Deployment)

If you prefer to run this pipeline locally, you will need Python installed along with the required dependencies.

1. Clone this repository:
   ```bash
   git clone [https://github.com/tu-usuario/Phyto-S-Palm.git](https://github.com/tu-usuario/Phyto-S-Palm.git)
   cd Phyto-S-Palm
