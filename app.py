import streamlit as st
import pandas as pd
from Bio import SeqIO
import io
import os
import subprocess
import tempfile
import sys

st.set_page_config(layout="wide", page_title="Phyto-S-Palm")

# --- Encabezado y Descripción ---
st.title("🍃Phyto-*S*-Palm")
st.markdown("### **An interactive bioinformatics tool to predict S-acylation sites in plant proteomes.**")

st.markdown("""
Phyto-*S*-Palm automates and facilitates the computational workflow for S-acylation site prediction. It processes FASTA proteomes by filtering cysteine-lacking sequences and evaluating biophysical surface accessibility using adjustable Kyte-Doolittle and Unified Hydrophobicity (UHS) scales. For the deep learning inference phase, a modernized MusiteDeep core (CNN and CapsNet) was migrated to TensorFlow's `tf_keras` with tensor-level optimizations. The tool accepts dynamic decision thresholds (defaulting to a stringent 0.65 cutoff). The modified source code and plant-trained models are publicly hosted on [GitHub](https://github.com/Awuacero/Phyto-S-Palm), and the platform is freely accessible at [https://phyto-s-palm.streamlit.app/](https://phyto-s-palm.streamlit.app/).
""")

st.warning("⚠️ **Notice: Under Peer Review**\n\nThis tool is part of an unpublished scientific manuscript. The methodology, underlying code, and pipeline are under Copyright (c) 2026. All rights reserved. Please do not distribute.")
st.markdown("---")

# --- About Us ---
st.header("About us")
st.markdown("""
Phyto-*S*-Palm aims to provide interactive bioinformatics tools to predict S-acylation sites in plant proteomes. The current version evaluates biophysical cysteine exposure and leverages custom deep learning architectures optimized for high-confidence target discovery in plants.

The tool has been developed by researchers from the [Estación Experimental del Zaidín (EEZ)-CSIC](https://www.eez.csic.es/) and the [Universidad de Málaga (UMA)](https://www.uma.es/), and [Instituto de Hortofruticultura Subtropical y Mediterránea "La Mayora" (IHSM-UMA-CSIC)](https://www.ihsm.uma-csic.es/).

Please, Contact us if you have any question or request: 📧 [andrea.roman@eez.csic.es](mailto:andrea.roman@eez.csic.es)
""")
st.markdown("---")

# --- 1. Sequence Filtering ---
st.header("1. Sequence Filtering")
st.markdown("Upload a FASTA file (`.fasta` or `.fa`) or use our sample sequence to discard sequences without Cysteine ('C').")

col_sample, col_upload = st.columns([1, 2])
with col_sample:
    use_sample = st.checkbox("Use sample FASTA sequence")
with col_upload:
    uploaded_file = st.file_uploader("Choose a FASTA file", type=["fasta", "fa"], disabled=use_sample)

filtered_sequences = []
sequences = []

# Carga directa desde el archivo físico
if use_sample:
    try:
        with open("sample.fasta", "r") as handle:
            sequences = list(SeqIO.parse(handle, "fasta"))
    except FileNotFoundError:
        st.error("No se encontró el archivo 'sample.fasta'. Asegúrate de que esté subido en la misma carpeta de tu repositorio en GitHub.")
elif uploaded_file is not None:
    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    sequences = list(SeqIO.parse(stringio, "fasta"))

if sequences:
    for seq_record in sequences:
        if 'C' in str(seq_record.seq).upper():
            filtered_sequences.append(seq_record)

    st.success(f"Parsed {len(sequences)} sequences. Retained {len(filtered_sequences)} sequences containing at least one Cysteine.")

    if filtered_sequences:
        with st.expander("View Filtered Sequence IDs"):
            for seq_record in filtered_sequences:
                st.write(f"- `{seq_record.id}`")
    else:
        st.warning("No sequences containing Cysteine were found.")

st.markdown("---")

# --- 2. Cys Biophysical Accessibility Analysis ---
st.header("2. Cysteine Biophysical Accessibility Analysis")
st.markdown("A sliding window evaluates local surface exposure using Kyte-Doolittle and Unified Hydrophobicity Scales (UHS).")

kyte_doolittle_scale = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

uhs_scale = {
    'A': 0.31, 'R': -0.58, 'N': -0.60, 'D': -0.82, 'C': 0.19,
    'Q': -0.22, 'E': -0.99, 'G': 0.00, 'H': -0.41, 'I': 0.73,
    'L': 0.73, 'K': -0.87, 'M': 0.26, 'F': 0.77, 'P': -0.06,
    'S': -0.26, 'T': -0.18, 'W': 0.39, 'Y': 0.10, 'V': 0.54
}

def calculate_hydrophobicity(sequence, scale):
    if not sequence: return 0.0
    hydro_values = [scale.get(aa.upper(), 0.0) for aa in sequence]
    return sum(hydro_values) / len(hydro_values) if hydro_values else 0.0

if filtered_sequences:
    col1, col2 = st.columns(2)
    with col1:
        window_size = st.slider("Sliding Window Size (aa)", min_value=3, max_value=21, value=9, step=2)

    st.subheader("Maximum Hydrophobicity Cutoffs for Solvent Accessibility")
    st.caption("Lower or negative values represent higher solvent exposure/accessibility. Cysteines ABOVE these cutoffs are discarded as buried.")

    col3, col4 = st.columns(2)
    with col3:
        kd_threshold = st.number_input("Max Kyte-Doolittle Cutoff", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    with col4:
        uhs_threshold = st.number_input("Max UHS Cutoff", min_value=-1.0, max_value=1.0, value=0.10, step=0.01)

    st.markdown("---")

    # --- 3. Intersection of Accessibility Criteria ---
    st.header("3. High-Confidence Accessibility Filtering")
    st.markdown("Only Cysteines meeting **BOTH** exposure cutoffs ($\le$ Cutoff) are preserved.")

    results = []
    valid_protein_ids = set()

    for seq_record in filtered_sequences:
        sequence_str = str(seq_record.seq).upper()
        has_valid_cys = False
        for i, aa in enumerate(sequence_str):
            if aa == 'C':
                start = max(0, i - (window_size // 2))
                end = min(len(sequence_str), i + (window_size // 2) + 1)
                subsequence = sequence_str[start:end]

                kd_hydro = calculate_hydrophobicity(subsequence, kyte_doolittle_scale)
                uhs_hydro = calculate_hydrophobicity(subsequence, uhs_scale)

                is_kd_exposed = kd_hydro <= kd_threshold
                is_uhs_exposed = uhs_hydro <= uhs_threshold
                is_exposed_combined = is_kd_exposed and is_uhs_exposed

                if is_exposed_combined:
                    has_valid_cys = True

                results.append({
                    "Protein ID": seq_record.id,
                    "Cys Position": i + 1,
                    "Subsequence": subsequence,
                    "KD Score": round(kd_hydro, 3),
                    "KD Valid": is_kd_exposed,
                    "UHS Score": round(uhs_hydro, 3),
                    "UHS Valid": is_uhs_exposed,
                    "Passes Both Filters": is_exposed_combined
                })
        if has_valid_cys:
            valid_protein_ids.add(seq_record.id)

    if results:
        df_results = pd.DataFrame(results)
        df_filtered_cys = df_results[df_results["Passes Both Filters"] == True]

        st.subheader("Results Overview")
        st.write(f"Total Cysteines evaluated: **{len(df_results)}** | Validated Surface Cysteines: **{len(df_filtered_cys)}**")

        st.dataframe(df_results, use_container_width=True)

        if not df_filtered_cys.empty:
            csv_data = df_filtered_cys.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Accessible Cysteines (CSV)",
                data=csv_data,
                file_name="accessible_cysteines.csv",
                mime="text/csv"
            )
            
            st.markdown("---")
            
            # --- 4. MusiteDeep Prediction Integration ---
            st.header("4. MusiteDeep Prediction Pipeline")
            st.markdown("Generate a refined FASTA file containing only proteins that passed the biophysical filter and run deep learning inference for S-palmitoylation.")

            col_cutoff, _ = st.columns([1, 1])
            with col_cutoff:
                musite_cutoff = st.slider(
                    "Prediction Threshold (Cutoff)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.65,
                    step=0.05,
                    help="Based on our benchmarking, we recommend a stringent cutoff of 0.65 to minimize false positives in plant proteomes."
                )

            passing_sequences = [seq for seq in filtered_sequences if seq.id in valid_protein_ids]

            fasta_io = io.StringIO()
            SeqIO.write(passing_sequences, fasta_io, "fasta")
            fasta_str = fasta_io.getvalue()

            st.download_button(
                label="📥 Download Filtered FASTA for MusiteDeep",
                data=fasta_str,
                file_name="filtered_input_for_musitedeep.fasta",
                mime="text/plain"
            )

            if st.button("🚀 Run MusiteDeep Prediction"):
                with st.spinner("Running deep learning models (CNN & CapsNet)... Please wait."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta", mode="w") as tmp_input:
                        tmp_input.write(fasta_str)
                        tmp_input_path = tmp_input.name

                    output_prefix = tempfile.mktemp(prefix="musite_out_")

                    try:
                        cmd = [
                            sys.executable, "predict_multi_batch.py",
                            "-input", tmp_input_path,
                            "-output", output_prefix,
                            "-model-prefix", "models/S-palmitoyl_cysteine",
                            "-cutoff", str(musite_cutoff)
                        ]
                        
                        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        
                        result_file_path = output_prefix + "_results.txt"
                        if os.path.exists(result_file_path):
                            st.success("Prediction completed successfully!")
                            
                            with open(result_file_path, "r") as res_f:
                                prediction_output = res_f.read()
                            
                            st.subheader("Prediction Results")
                            st.text(prediction_output)
                            
                            st.download_button(
                                label="📥 Download Full Prediction Report (.txt)",
                                data=prediction_output,
                                file_name="s_acylation_prediction_results.txt",
                                mime="text/plain"
                            )
                        else:
                            st.error("The prediction finished, but the results file was not generated correctly.")
                            with st.expander("View Error Logs"):
                                st.text(process.stderr)

                    except subprocess.CalledProcessError as e:
                        st.error("An error occurred during model execution.")
                        with st.expander("View Error Details"):
                            st.text(e.stderr)
                    
                    finally:
                        if os.path.exists(tmp_input_path):
                            os.remove(tmp_input_path)

    else:
        st.info("No Cysteines found matching the biophysical parameters.")
else:
    st.info("Please upload a FASTA file or check the sample sequence to begin processing.")

st.markdown("---")

# --- Funding ---
st.header("Funding")
st.markdown("This work was funded by the Spanish Ministry of Science and Innovation (MICINN/AEI) through EU/ERDF-cofinanced grants PID2020-113324GB-100, … and … , and by CSIC (grant no. 2025AEP052).")
st.markdown("---")

# --- How to cite ---
st.header("How to cite Phyto-S-Palm")
st.markdown("Please, when using this web site or its data, cite us using the reference:")
st.markdown("> Román Mateo, A., Gallego, F., Santos, J., Alché, J. D., Claros, G., Veredas, F. J., & Castro, A. J (2026). *Integrative Computational and Experimental S-Acylation Profiling Reveals a Conserved Pollen S-Acylome in Angiosperms*. (Unpublished manuscript).")

st.markdown("<p align='center'><img src='https://raw.githubusercontent.com/Awuacero/Phyto-S-Palm/main/logos.jpg' alt='Institution Logos' style='max-width: 100%; width: 100%; height: auto;'/></p>", unsafe_allow_html=True)
st.markdown("---")

# --- Bibliography ---
st.header("Bibliography")
st.markdown("""
**Biophysical Scales (Surface Accessibility):**
* Koehler, J., Woetzel, N., Staritzbichler, R., Sanders, C. R., & Meiler, J. (2009). A unified hydrophobicity scale for multispan membrane proteins. *Proteins*, 76(1), 13-29. [https://doi.org/10.1002/prot.22315](https://doi.org/10.1002/prot.22315)
* Kyte, J., & Doolittle, R. F. (1982). A simple method for displaying the hydropathic character of a protein. *Journal of Molecular Biology*, 157(1), 105–132. [https://doi.org/10.1016/0022-2836(82)90515-0](https://doi.org/10.1016/0022-2836(82)90515-0)

**Deep Learning Architecture:**
* Wang, D., Guan, Y., Shu, Q., Song, B., & Xu, D. (2017). MusiteDeep: A deep-learning framework for general and kinase-specific phosphorylation site prediction. *Bioinformatics*, 33(24), 3909–3916. [https://doi.org/10.1093/bioinformatics/btx496](https://doi.org/10.1093/bioinformatics/btx496)
""")
