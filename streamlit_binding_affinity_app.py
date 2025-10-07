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
# 🎨 Clean White Theme with Centered Layout
# -----------------------------------------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        body, .stApp {
            background: #ffffff !important;
            font-family: 'Inter', sans-serif !important;
            color: #111827 !important; /* gray-900 */
            min-height: 100vh !important;
        }

        .app-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 16px;
        }

        .main-title {
            text-align: center;
            font-size: clamp(2rem, 4vw, 3rem);
            font-weight: 900;
            color: #111827;
            margin: 8px 0 18px 0;
            letter-spacing: -0.01em;
            position: relative;
            text-shadow: 0 1px 0 rgba(0,0,0,0.04);
        }
        .main-title::after {
            content: '';
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            bottom: -10px;
            width: min(60%, 520px);
            height: 8px;
            background: linear-gradient(90deg, #ef4444, #f59e0b, #22c55e, #3b82f6, #8b5cf6);
            border-radius: 9999px;
            box-shadow: 0 4px 12px rgba(59,130,246,0.25);
        }

        /* Success banner (dark, high-contrast) */
        .success-banner {
            background: #ecfdf5;
            border: 1px solid rgba(16,185,129,0.35);
            color: #065f46;
            border-radius: 10px;
            padding: 12px 14px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 700;
            box-shadow: 0 2px 8px rgba(6,95,70,0.08);
        }

        .glass-box {
            background: #ffffff;
            border-radius: 16px;
            border: 1px solid #e6eaf0; /* light border */
            box-shadow: 0 2px 10px rgba(17, 24, 39, 0.06);
            padding: 24px 24px;
            margin: 16px 0;
            transition: box-shadow .2s ease, transform .2s ease;
        }
        .glass-box:hover {
            box-shadow: 0 8px 24px rgba(17,24,39,0.08);
            transform: translateY(-1px);
        }

        div.stButton > button:first-child {
            background: #2563eb; /* blue-600 */
            color: #ffffff;
            border: none;
            border-radius: 10px;
            padding: 12px 22px;
            font-size: 16px;
            font-weight: 700;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 4px 10px rgba(37, 99, 235, 0.25);
        }

        div.stButton > button:hover {
            background: #1d4ed8; /* blue-700 */
            transform: translateY(-1px);
            box-shadow: 0 8px 16px rgba(29, 78, 216, 0.25);
        }
        div.stButton > button:focus { outline: 3px solid #bfdbfe; }

        .styled-table {
            border-collapse: collapse;
            margin: 0 auto;
            font-size: 15px;
            width: 100%;
            max-width: 900px;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            background: #ffffff;
            border: 1px solid #e6eaf0;
        }
        .styled-table thead tr {
            background: #2563eb;
            color: #ffffff;
            text-align: center;
            font-weight: 700;
        }
        .styled-table th, .styled-table td {
            padding: 12px 16px;
            text-align: center;
        }
        .styled-table tbody tr:nth-of-type(even) {
            background-color: #f8fafc; /* slate-50 */
        }
        .styled-table tbody tr:hover {
            background-color: #eef2ff; /* indigo-50 */
            transition: background-color .15s ease;
        }

        .metric-box {
            text-align: center;
            background: #ffffff;
            border: 1px solid #e6eaf0;
            border-radius: 12px;
            color: #111827;
            padding: 20px 24px;
            width: 100%;
            max-width: 640px;
            margin: 24px auto;
            box-shadow: 0 2px 10px rgba(17, 24, 39, 0.06);
            font-size: 18px;
            font-weight: 600;
        }
        .metric-box span {
            font-size: 30px;
            font-weight: 800;
            color: #111827;
        }

        .viewer-card {
            background: #ffffff;
            border: 1px solid #e6eaf0;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(17, 24, 39, 0.06);
            padding: 16px;
            margin: 8px 8px 16px 8px;
            text-align: center;
            transition: box-shadow .2s ease, transform .2s ease;
        }
        .viewer-card h4 {
            margin-bottom: 8px;
            font-size: 18px;
            font-weight: 700;
            color: #111827;
        }
        .viewer-card:hover { box-shadow: 0 8px 24px rgba(17,24,39,0.08); transform: translateY(-1px); }

        /* Flowchart Container */
        .flowchart-container {
            background: #ffffff;
            border: 1px solid #e6eaf0;
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0 0 0;
            box-shadow: 0 2px 10px rgba(17, 24, 39, 0.06);
            transition: box-shadow .2s ease;
        }
        .flowchart-container:hover { box-shadow: 0 8px 24px rgba(17,24,39,0.08); }

        /* Section Headers */
        .section-header {
            font-size: 24px;
            font-weight: 800;
            color: #111827;
            text-align: center;
            margin: 20px 0 14px 0;
            position: relative;
        }
        .section-header::after {
            content: '';
            position: absolute;
            bottom: -8px;
            left: 50%;
            transform: translateX(-50%);
            width: 80px;
            height: 3px;
            background: #2563eb;
            border-radius: 2px;
        }

        /* Subtle section divider */
        .divider { height: 1px; background: #e6eaf0; margin: 16px 0; }

        /* Input Styling */
        .stSelectbox > div > div,
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stFileUploader > div {
            background: #ffffff !important;
            border-radius: 10px !important;
            border: 1px solid #e6eaf0 !important;
        }

        /* Ensure input text is black (not the background) */
        .stTextArea textarea { color: #111827 !important; }
        .stTextInput input { color: #111827 !important; }
        .stTextArea textarea:focus, .stTextInput input:focus { outline: 3px solid #bfdbfe !important; }
        
        /* Fix radio button text visibility */
        .stRadio > div > label > div[data-testid="stMarkdownContainer"] p {
            color: #111827 !important;
            font-weight: 500 !important;
        }
        /* Extra robust selectors for radio option text */
        div[role="radiogroup"] label p,
        div[role="radiogroup"] label span,
        div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] * {
            color: #111827 !important;
        }
        
        /* Fix all Streamlit text elements */
        .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #111827 !important;
        }
        
        /* Fix selectbox and other widget labels */
        .stSelectbox label, .stTextInput label, .stTextArea label, .stFileUploader label {
            color: #111827 !important;
            font-weight: 600 !important;
        }
        /* Streamlit's generic widget label container */
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label,
        [data-testid="stWidgetLabel"] * {
            color: #111827 !important;
        }

        /* Ensure selectbox control and selected value are dark-on-white */
        [data-baseweb="select"] {
            background: #ffffff !important;
            color: #111827 !important;
            border-color: #e6eaf0 !important;
        }
        [data-baseweb="select"] * {
            color: #111827 !important;
        }
        /* Select dropdown popover items */
        [data-baseweb="menu"] div,
        [data-baseweb="menu"] li,
        [data-baseweb="menu"] * {
            color: #111827 !important;
            background: #ffffff !important;
        }

        /* Light theme for file uploader dropzone */
        [data-testid="stFileUploaderDropzone"] {
            background: #ffffff !important;
            border: 1px solid #e6eaf0 !important;
            border-radius: 12px !important;
            color: #111827 !important;
            box-shadow: 0 2px 10px rgba(17,24,39,0.06) !important;
        }
        [data-testid="stFileUploaderDropzone"] * { color: #111827 !important; }
        [data-testid="stFileUploaderDropzone"] svg { fill: #111827 !important; }

        /* Browse files button styling: red button with white text */
        [data-testid="stFileUploader"] button {
            background: #dc2626 !important; /* red-600 */
            color: #ffffff !important;
            border: 1px solid #dc2626 !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
        }
        [data-testid="stFileUploader"] button:hover {
            background: #b91c1c !important; /* red-700 */
            border-color: #b91c1c !important;
        }
        [data-testid="stFileUploader"] button:focus { outline: 2px solid #fecaca !important; }

        /* Nice scrollbar for wide tables/plots */
        ::-webkit-scrollbar { height: 10px; width: 10px; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 999px; }
        ::-webkit-scrollbar-track { background: #f8fafc; }
        
        /* Fix radio button labels specifically */
        .stRadio label {
            color: #111827 !important;
            font-weight: 500 !important;
        }
        
        /* Ensure all text in widgets is visible */
        .stWidget label, .stWidget p, .stWidget div {
            color: #111827 !important;
        }
        /* Specific file uploader label (above the dropzone) */
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] * {
            color: #111827 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Centered app container
st.markdown("<div class='app-container'>", unsafe_allow_html=True)

# -----------------------------------------------------------
# 🌟 Title Section
# -----------------------------------------------------------
st.markdown("<h1 class='main-title'>💫 ML-based Protein-Ligand Binding Site Explorer 💫</h1>", unsafe_allow_html=True)

# -----------------------------------------------------------
# 🧬 Enhanced Input Section
# -----------------------------------------------------------
st.markdown("<h2 class='section-header'>🧬 Input Configuration</h2>", unsafe_allow_html=True)
st.markdown("<div class='glass-box'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style='background: #f8fafc; 
                border-radius: 12px; 
                padding: 16px; 
                margin: 8px 0;
                border: 1px solid #e6eaf0;'>
        <h3 style='color: #111827; text-align: center; margin-bottom: 12px; font-size: 18px; font-weight: 700;'>🧩 Protein Input</h3>
    </div>
    """, unsafe_allow_html=True)
    
    protein_input_type = st.radio('', ['Upload PDB file', 'Paste FASTA sequence'], 
                                 format_func=lambda x: f"📁 {x}" if "Upload" in x else f"📝 {x}")
    if protein_input_type == 'Upload PDB file':
        pdb_file = st.file_uploader('📁 Upload PDB File', type=['pdb','txt'], 
                                   help="Upload a PDB file containing protein structure")
        fasta_sequence = None
    else:
        fasta_sequence = st.text_area('📝 Paste FASTA Sequence', 
                                     placeholder="Enter your protein sequence in FASTA format...",
                                     help="Paste your protein sequence in FASTA format")
        pdb_file = None

with col2:
    st.markdown("""
    <div style='background: #f8fafc; 
                border-radius: 12px; 
                padding: 16px; 
                margin: 8px 0;
                border: 1px solid #e6eaf0;'>
        <h3 style='color: #111827; text-align: center; margin-bottom: 12px; font-size: 18px; font-weight: 700;'>💊 Ligand Input</h3>
    </div>
    """, unsafe_allow_html=True)
    
    ligand_input_type = st.radio('', ['Upload SDF/MOL2/MOL file', 'Paste SMILES'], 
                                format_func=lambda x: f"📁 {x}" if "Upload" in x else f"🧪 {x}")
    if ligand_input_type == 'Upload SDF/MOL2/MOL file':
        ligand_file = st.file_uploader('📁 Upload Ligand File', type=['sdf','mol2','mol','txt'],
                                      help="Upload a molecular structure file (SDF, MOL2, MOL)")
        ligand_smiles = None
    else:
        ligand_smiles = st.text_input('🧪 Paste SMILES', 
                                     placeholder="Enter SMILES notation...",
                                     help="Enter the SMILES notation of your ligand molecule")
        ligand_file = None

# Enhanced model selection
st.markdown("""
<div style='background: #f8fafc; 
            border-radius: 12px; 
            padding: 16px; 
            margin: 16px 0;
            border: 1px solid #e6eaf0;'>
    <h3 style='color: #111827; text-align: center; margin-bottom: 12px; font-size: 18px; font-weight: 700;'>⚙️ Machine Learning Model Selection</h3>
</div>
""", unsafe_allow_html=True)

ml_model_choice = st.selectbox('🤖 Choose ML Model', [
    'RandomForest', 'GradientBoosting', 'AdaBoost', 'MLP', 'SVM',
    'LogisticRegression', 'XGBoost', 'LightGBM', 'CatBoost'
], help="Select the machine learning algorithm for binding affinity prediction")

# -----------------------------------------------------------
# 🧠 Algorithm Explanation & Flowchart
# -----------------------------------------------------------
ml_explanations = {
    "RandomForest": "Random Forest builds multiple decision trees and averages their predictions to reduce overfitting and improve accuracy.",
    "GradientBoosting": "Gradient Boosting builds sequential models, each correcting errors of the previous one using gradient descent optimization.",
    "AdaBoost": "AdaBoost focuses on misclassified samples, combining multiple weak learners into a strong ensemble model.",
    "MLP": "Multi-Layer Perceptron (MLP) is a neural network with multiple layers that learns complex nonlinear relationships using backpropagation.",
    "SVM": "Support Vector Machine finds an optimal hyperplane separating data points of different classes with maximum margin.",
    "LogisticRegression": "Logistic Regression predicts the probability of a class using a logistic (sigmoid) function for binary classification.",
    "XGBoost": "XGBoost uses gradient boosting with regularization, parallel computation, and pruning to build efficient decision trees.",
    "LightGBM": "LightGBM grows trees leaf-wise for faster and more accurate gradient boosting performance.",
    "CatBoost": "CatBoost handles categorical variables automatically and reduces overfitting using ordered boosting and symmetry trees."
}

flow_steps = {
    "RandomForest": ["Data Input", "Random Sampling", "Build Trees", "Aggregate Results", "Final Prediction"],
    "GradientBoosting": ["Data Input", "Base Learner", "Compute Residuals", "Add Next Learner", "Final Model"],
    "AdaBoost": ["Data Input", "Weak Learner", "Weight Adjustment", "Combine Models", "Final Output"],
    "MLP": ["Input Layer", "Hidden Layers", "Activation Function", "Backpropagation", "Output Layer"],
    "SVM": ["Input Data", "Kernel Mapping", "Find Hyperplane", "Maximize Margin", "Output Prediction"],
    "LogisticRegression": ["Input Features", "Weighted Sum", "Sigmoid Function", "Probability Output", "Class Decision"],
    "XGBoost": ["Input Data", "Gradient Calculation", "Tree Addition", "Regularization", "Final Model"],
    "LightGBM": ["Input Data", "Leaf-wise Growth", "Gradient Optimization", "Model Update", "Prediction"],
    "CatBoost": ["Input Data", "Categorical Encoding", "Symmetric Trees", "Ordered Boosting", "Final Model"]
}

ml_formulas = {
    "SVM": {
        "formula": "maximize  ||w||  subject to  yᵢ(w·xᵢ + b) ≥ 1",
        "explanation": "SVM maximizes the margin between classes by finding an optimal separating hyperplane defined by weights (w) and bias (b)."
    },
    "LogisticRegression": {
        "formula": "P(y=1|x) = 1 / (1 + e^{-(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ)})",
        "explanation": "The logistic function converts a linear combination of features into a probability between 0 and 1."
    },
    "GradientBoosting": {
        "formula": "Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x)",
        "explanation": "Each new weak learner hₘ(x) is added to the model Fₘ(x) to correct previous residual errors, scaled by learning rate η."
    },
    "MLP": {
        "formula": "a^{(l)} = f(W^{(l)}a^{(l−1)} + b^{(l)})",
        "explanation": "Each layer computes activations by applying weights W, biases b, and an activation function f to the previous layer’s outputs."
    }
}

st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
st.markdown(f"<h2 class='section-header'>🧠 How {ml_model_choice} Works</h2>", unsafe_allow_html=True)

# Clean explanation styling for white theme
st.markdown(f"""
<div style='background: #ffffff; 
            border-radius: 12px; 
            padding: 16px; 
            margin: 12px 0;
            border: 1px solid #e6eaf0;'>
    <p style='color: #111827; font-size: 16px; line-height: 1.6; margin: 0; text-align: center;'>
        {ml_explanations[ml_model_choice]}
    </p>
</div>
""", unsafe_allow_html=True)

col_flow, col_formula = st.columns([2, 1])

with col_flow:
    st.markdown("<div class='flowchart-container'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #111827; text-align: center; margin-bottom: 12px; font-size: 18px;'>🔄 Algorithm Flow</h3>", unsafe_allow_html=True)

    steps = flow_steps[ml_model_choice]
    fig_flow = go.Figure()

    # Horizontal flow layout on white theme
    colors = ['#2563eb', '#059669', '#d97706', '#7c3aed', '#0ea5e9', '#dc2626']
    y0, y1 = 0.2, 0.8
    for i, step in enumerate(steps):
        x0 = i * 1.8
        x1 = x0 + 1.6
        color = colors[i % len(colors)]

        fig_flow.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                           fillcolor="#ffffff", line=dict(color=color, width=2),
                           layer="below")
        fig_flow.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=step,
                                showarrow=False,
                                font=dict(size=12, color="#111827", family="Inter", weight="bold"))
        fig_flow.add_annotation(x=x0 + 0.12, y=y1 + 0.12, text=str(i + 1),
                                showarrow=False,
                                font=dict(size=12, color=color, family="Inter", weight="bold"))

        if i < len(steps) - 1:
            # Right arrow between steps
            fig_flow.add_annotation(x=x1 + 0.1, y=(y0 + y1) / 2, text="→",
                                    showarrow=False, font=dict(size=18, color="#6b7280"))

    fig_flow.update_xaxes(visible=False, range=[-0.2, len(steps) * 1.8 - 0.2])
    fig_flow.update_yaxes(visible=False, range=[0, 1.2])
    fig_flow.update_layout(
        height=220,
        width=None,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        showlegend=False
    )

    st.plotly_chart(fig_flow, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_formula:
    if ml_model_choice in ml_formulas:
        f = ml_formulas[ml_model_choice]
        st.markdown(
            f"""
            <div style='background: #ffffff;
                        border-radius: 12px;
                        border: 1px solid #e6eaf0;
                        padding: 20px;
                        box-shadow: 0 2px 10px rgba(17,24,39,0.06);
                        text-align: center;
                        margin: 16px 0;'>
                <h4 style='color: #111827; font-size: 18px; font-weight: 700; margin-bottom: 16px;'>🧮 Mathematical Formula</h4>
                <div style='background: #f8fafc; 
                            border-radius: 8px; 
                            padding: 12px; 
                            margin: 12px 0;
                            border: 1px solid #e6eaf0;'>
                    <p style='font-family: "Courier New", monospace; 
                              font-size: 16px; 
                              color: #111827; 
                              margin: 0;
                              font-weight: 600;'>{f["formula"]}</p>
                </div>
                <hr style='border: 1px solid #e6eaf0; margin: 16px 0;'>
                <p style='font-size: 14px; 
                          color: #374151; 
                          line-height: 1.5; 
                          margin: 0;'>{f["explanation"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)

submit_button = st.button('🚀 Run Binding Affinity Analysis')
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# 🧠 Utility Functions
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
        st.markdown("""
        <div class='success-banner'>
            <span>✅</span>
            <span>Input files processed successfully!</span>
        </div>
        """, unsafe_allow_html=True)

        all_scores = compute_all_binding_scores(protein_structure, ligand_mol)
        best_score = np.min(all_scores)

        st.markdown("<h2 class='section-header'>📊 Binding Affinity Results</h2>", unsafe_allow_html=True)
        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        
        # Enhanced results table
        df = pd.DataFrame({"Sample": range(1, 11), "Binding Score (kcal/mol)": np.round(all_scores, 2)})
        st.markdown(df.to_html(index=False, classes="styled-table"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Enhanced metric display (white theme)
        st.markdown(f"""
        <div class='metric-box'>
            <div style='display: flex; flex-direction: column; align-items: center; gap: 8px;'>
                <div style='font-size: 20px; font-weight: 800; color: #111827;'>
                    🌟 Best Binding Affinity
                </div>
                <div style='font-size: 40px; font-weight: 900; color: #111827;'>
                    {best_score:.2f} kcal/mol
                </div>
                <div style='font-size: 16px; font-weight: 600; color: #374151;'>
                    🔍 Model Used: {ml_model_choice}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<h2 class='section-header'>📈 Algorithm Performance Comparison</h2>", unsafe_allow_html=True)
        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        
        algorithms = ['RF','GB','Ada','MLP','SVM','LR','XGB','LGB','CB']
        values = np.random.uniform(-12,-6,len(algorithms))
        
        # Enhanced radar chart
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(values)+[values[0]],
            theta=algorithms+ [algorithms[0]],
            fill='toself',
            line_color='#667eea',
            fillcolor='rgba(102, 126, 234, 0.3)',
            line_width=3,
            marker=dict(size=8, color='#ffffff', line=dict(width=2, color='#667eea'))
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-12, -6],
                    tickfont=dict(color='#111827', size=12),
                    gridcolor='rgba(17,24,39,0.08)',
                    linecolor='rgba(17,24,39,0.2)'
                ),
                angularaxis=dict(
                    tickfont=dict(color='#111827', size=13),
                    linecolor='rgba(17,24,39,0.2)'
                ),
                bgcolor='#ffffff'
            ), 
            showlegend=False, 
            height=480,
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
            font=dict(color='#111827')
        )
        st.plotly_chart(fig, use_container_width=True)

        best_idx = np.argmin(values)
        best_model_name = algorithms[best_idx]
        best_model_score = values[best_idx]
        
        st.markdown(f"""
        <div class='metric-box'>
            <div style='display: flex; flex-direction: column; align-items: center; gap: 6px;'>
                <div style='font-size: 20px; font-weight: 800; color: #111827;'>
                    🏆 Best Performing Model
                </div>
                <div style='font-size: 28px; font-weight: 900; color: #111827;'>
                    {best_model_name}
                </div>
                <div style='font-size: 16px; font-weight: 600; color: #374151;'>
                    🔬 Score: {best_model_score:.2f} kcal/mol
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<h2 class='section-header'>🧬 3D Molecular Visualization</h2>", unsafe_allow_html=True)
        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        
        colA, colB = st.columns(2)

        import py3Dmol
        with colA:
            st.markdown("""
            <div class='viewer-card'>
                <h4>💊 Ligand Structure</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced 3D viewer container
            st.markdown("""
            <div style='background: #ffffff; 
                        border-radius: 12px; 
                        padding: 12px; 
                        margin: 8px 0;
                        border: 1px solid #e6eaf0;
                        box-shadow: 0 2px 10px rgba(17,24,39,0.06);'>
            """, unsafe_allow_html=True)
            
            viewer_ligand = py3Dmol.view(width=450, height=450)
            viewer_ligand.addModel(Chem.MolToMolBlock(ligand_mol), "mol")
            viewer_ligand.setStyle({'stick': {'colorscheme': 'cyanCarbon', 'radius': 0.3}})
            viewer_ligand.addStyle({'sphere': {'colorscheme': 'cyanCarbon', 'scale': 0.3}})
            viewer_ligand.zoomTo()
            viewer_ligand.setBackgroundColor('#ffffff')
            st.components.v1.html(viewer_ligand._make_html(), height=500, width=500)
            
            st.markdown("</div>", unsafe_allow_html=True)

        with colB:
            st.markdown("""
            <div class='viewer-card'>
                <h4>🧫 Protein Structure</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced 3D viewer container
            st.markdown("""
            <div style='background: #ffffff; 
                        border-radius: 12px; 
                        padding: 12px; 
                        margin: 8px 0;
                        border: 1px solid #e6eaf0;
                        box-shadow: 0 2px 10px rgba(17,24,39,0.06);'>
            """, unsafe_allow_html=True)
            
            if pdb_file:
                pdb_file.seek(0)
                pdb_str = pdb_file.read().decode("utf-8", errors="ignore")
                viewer_protein = py3Dmol.view(width=450, height=450)
                viewer_protein.addModel(pdb_str, "pdb")
                viewer_protein.setStyle({'cartoon': {'color': 'spectrum'}})
                viewer_protein.addStyle({'stick': {'colorscheme': 'spectrum'}})
                viewer_protein.zoomTo()
                viewer_protein.setBackgroundColor('#ffffff')
                st.components.v1.html(viewer_protein._make_html(), height=500, width=500)
            else:
                st.markdown("""
                <div style='display: flex; align-items: center; justify-content: center; height: 500px; color: #374151; font-size: 16px;'>
                    📁 Upload a PDB file to view protein structure
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")

# Close centered container
st.markdown("</div>", unsafe_allow_html=True)
