import streamlit as st
import tempfile
import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from rdkit import Chem
import plotly.graph_objects as go

# -----------------------------------------------------------
# 🌟 Page Config
# -----------------------------------------------------------
st.set_page_config(page_title="ML Binding Explorer", layout="wide")

# -----------------------------------------------------------
# 🎨 Glassmorphic Transparent Grid Theme
# -----------------------------------------------------------
st.markdown("""
    <style>
        body, .stApp {
            background: linear-gradient(135deg, #e3f2fd 0%, #f8fbff 100%) !important;
            font-family: 'Poppins', sans-serif;
            color: #1a1a1a;
        }

        /* Title Styling */
        .main-title {
            text-align: center;
            font-size: 50px;
            font-weight: 800;
            background: linear-gradient(90deg, #667eea, #00f2fe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 1px 1px 6px rgba(0,0,0,0.1);
            margin-bottom: 10px;
        }

        /* Transparent Grid Box */
        .glass-box {
            background: rgba(255, 255, 255, 0.55);
            backdrop-filter: blur(12px);
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.4);
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
            padding: 25px 30px;
            margin: 15px 0;
            transition: 0.3s ease;
        }
        .glass-box:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 28px rgba(0,0,0,0.15);
        }

        /* Buttons */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #4facfe, #6a82fb);
            color: white;
            border: none;
            border-radius: 50px;
            padding: 12px 36px;
            font-size: 17px;
            font-weight: 600;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 6px 15px rgba(0,0,0,0.15);
        }
        div.stButton > button:hover {
            transform: scale(1.07);
            box-shadow: 0 10px 22px rgba(0,0,0,0.2);
        }

        /* Results Table */
        .styled-table {
            border-collapse: collapse;
            margin: 0 auto;
            font-size: 15px;
            width: 60%;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        .styled-table thead tr {
            background: #4facfe;
            color: #ffffff;
            text-align: center;
            font-weight: 600;
        }
        .styled-table th, .styled-table td {
            padding: 10px 15px;
            text-align: center;
        }
        .styled-table tbody tr:nth-of-type(even) {
            background-color: rgba(230,240,255,0.6);
        }
        .styled-table tbody tr:hover {
            background-color: rgba(200,220,255,0.6);
        }

        /* Metric Box Centered */
        .metric-box {
            text-align: center;
            background: rgba(255, 255, 255, 0.55);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(255,255,255,0.5);
            border-radius: 15px;
            color: #003366;
            padding: 20px 30px;
            width: 45%;
            margin: 25px auto;
            box-shadow: 0 6px 18px rgba(0,0,0,0.2);
            font-size: 20px;
        }
        .metric-box span {
            font-size: 28px;
            font-weight: 700;
            color: #000;
        }

        /* 3D Viewer Cards */
        .viewer-card {
            background: rgba(255, 255, 255, 0.55);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.4);
            border-radius: 18px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
            padding: 15px;
            margin: 15px;
            text-align: center;
            transition: all 0.3s ease;
        }
        .viewer-card h4 {
            margin-bottom: 12px;
            font-size: 18px;
            font-weight: 600;
            color: #003366;
        }
        .viewer-card:hover {
            transform: scale(1.03);
            box-shadow: 0 12px 30px rgba(0,0,0,0.2);
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 🌟 Title Section
# -----------------------------------------------------------
st.markdown("<h1 class='main-title'>💫 ML-based Protein-Ligand Binding Site Explorer 💫</h1>", unsafe_allow_html=True)

# -----------------------------------------------------------
# 🧬 Input Section (Glass Box)
# -----------------------------------------------------------
st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧩 Protein Input")
    protein_input_type = st.radio('', ['Upload PDB file', 'Paste FASTA sequence'])
    if protein_input_type == 'Upload PDB file':
        pdb_file = st.file_uploader('Upload PDB File', type=['pdb','txt'])
        fasta_sequence = None
    else:
        fasta_sequence = st.text_area('Paste FASTA Sequence')
        pdb_file = None

with col2:
    st.markdown("### 💊 Ligand Input")
    ligand_input_type = st.radio('', ['Upload SDF/MOL2/MOL file', 'Paste SMILES'])
    if ligand_input_type == 'Upload SDF/MOL2/MOL file':
        ligand_file = st.file_uploader('Upload Ligand File', type=['sdf','mol2','mol','txt'])
        ligand_smiles = None
    else:
        ligand_smiles = st.text_input('Paste SMILES')
        ligand_file = None

ml_model_choice = st.selectbox('⚙️ Choose ML Model', [
    'RandomForest', 'GradientBoosting', 'AdaBoost', 'MLP', 'SVM',
    'LogisticRegression', 'XGBoost', 'LightGBM', 'CatBoost'
])
submit_button = st.button('🚀 Run Binding Affinity Analysis')
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# 🧠 Utility functions
# -----------------------------------------------------------
def read_pdb(file_content):
    parser = PDBParser(QUIET=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdb') as tmp:
        tmp.write(file_content)
        tmp.flush()
        structure = parser.get_structure('prot', tmp.name)
    os.unlink(tmp.name)
    return structure

def read_ligand_from_file(file_bytes):
    mol = Chem.MolFromMolBlock(file_bytes.decode('utf-8', errors='ignore'))
    if mol is None:
        mol = Chem.MolFromMolFile(file_bytes, sanitize=True)
    return mol

def read_ligand_from_smiles(smiles):
    return Chem.MolFromSmiles(smiles)

def compute_all_binding_scores(protein_structure, ligand_mol, n_samples=10):
    return np.random.uniform(-12.0, -5.0, size=n_samples)

# -----------------------------------------------------------
# 🚀 Analysis Section
# -----------------------------------------------------------
if submit_button:
    try:
        if not (pdb_file or fasta_sequence) or not (ligand_file or ligand_smiles):
            st.error("⚠️ Please provide both protein and ligand inputs.")
            st.stop()

        protein_structure = read_pdb(pdb_file.read()) if pdb_file else None
        ligand_mol = read_ligand_from_file(ligand_file.read()) if ligand_file else read_ligand_from_smiles(ligand_smiles)
        st.success("✅ Input files processed successfully!")

        # Compute dummy scores
        all_scores = compute_all_binding_scores(protein_structure, ligand_mol)
        best_score = np.min(all_scores)

        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        st.markdown("### 📊 Binding Affinity Results")
        df = pd.DataFrame({"Sample": range(1, 11), "Binding Score (kcal/mol)": np.round(all_scores, 2)})
        st.markdown(df.to_html(index=False, classes="styled-table"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"<div class='metric-box'>🌟 Best Binding Affinity: <span>{best_score:.2f} kcal/mol</span><br>🔍 Model Used: {ml_model_choice}</div>", unsafe_allow_html=True)

        # Radar Chart
        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        st.subheader("📈 Algorithm Comparison")
        algorithms = ['RF','GB','Ada','MLP','SVM','LR','XGB','LGB','CB']
        values = np.random.uniform(-12,-6,len(algorithms))
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(values)+[values[0]],
            theta=algorithms+ [algorithms[0]],
            fill='toself',
            line_color='#0072ff'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False, height=450)
        st.plotly_chart(fig, use_container_width=True)

        # 🏆 Best Performing Model
        best_idx = np.argmin(values)
        best_model_name = algorithms[best_idx]
        best_model_score = values[best_idx]

        st.markdown(f"""
        <div class='metric-box'>
            🏆 Best Performing Model: <span>{best_model_name}</span><br>
            🔬 Best Score: <span>{best_model_score:.2f} kcal/mol</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 3D Viewer Section
        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        st.markdown("### 🧬 3D Visualization of Structures")
        colA, colB = st.columns(2)

        import py3Dmol
        with colA:
            st.markdown("<div class='viewer-card'><h4>💊 Ligand Structure</h4>", unsafe_allow_html=True)
            viewer_ligand = py3Dmol.view(width=420, height=420)
            viewer_ligand.addModel(Chem.MolToMolBlock(ligand_mol), "mol")
            viewer_ligand.setStyle({'stick': {'colorscheme': 'cyanCarbon'}})
            viewer_ligand.zoomTo()
            st.components.v1.html(viewer_ligand._make_html(), height=450, width=450)
            st.markdown("</div>", unsafe_allow_html=True)

        with colB:
            st.markdown("<div class='viewer-card'><h4>🧫 Protein Structure</h4>", unsafe_allow_html=True)
            pdb_file.seek(0)
            pdb_str = pdb_file.read().decode("utf-8", errors="ignore")
            viewer_protein = py3Dmol.view(width=420, height=420)
            viewer_protein.addModel(pdb_str, "pdb")
            viewer_protein.setStyle({'cartoon': {'color': 'spectrum'}})
            viewer_protein.zoomTo()
            st.components.v1.html(viewer_protein._make_html(), height=450, width=450)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")
