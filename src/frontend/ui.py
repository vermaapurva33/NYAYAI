import streamlit as st
import requests
import fitz  # PyMuPDF
from PIL import Image
import io

# --- CONFIG ---
API_URL = "http://localhost:8000/detect-mistakes"
st.set_page_config(layout="wide", page_title="NyayAI Phase 1")

st.title("NyayAI: Document Verification")

# --- STATE MANAGEMENT ---
if "is_analyzing" not in st.session_state:
    st.session_state.is_analyzing = False

if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None

# --- HELPER ---
def get_original_page_image(pdf_bytes, page_idx):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_idx < 0 or page_idx >= doc.page_count:
        return None
    page = doc.load_page(page_idx)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

# --- MAIN UI ---
uploaded_file = st.file_uploader("Upload Legal Document (PDF)", type="pdf")

if uploaded_file is not None:
    # Reset if new file
    if st.session_state.current_file_name != uploaded_file.name:
        st.session_state.is_analyzing = False
        st.session_state.current_file_name = uploaded_file.name

    file_bytes = uploaded_file.getvalue()
    
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        total_pages = doc.page_count

    # --- CLEAN CONTROL ROW ---
    # No stop button here anymore, just the page selector
    col_ctrl, _ = st.columns([1, 5])
    with col_ctrl:
        page_number = st.number_input("Select Page", 1, total_pages, 1)

    # --- IMAGE COLUMNS ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        original_img = get_original_page_image(file_bytes, page_number - 1)
        st.image(original_img, use_container_width=True)

    with col2:
        st.subheader("Mistakes Detected")

        # 1. State: Not Started Yet
        if not st.session_state.is_analyzing:
            st.info("Ready to Analyze")
            if st.button("Start Analysis Session"):
                st.session_state.is_analyzing = True
                st.rerun()

        # 2. State: Analyzing (Automatic)
        else:
            with st.spinner(f"Analyzing Page {page_number}..."):
                try:
                    files = {"file": (uploaded_file.name, file_bytes, "application/pdf")}
                    data = {"page_number": page_number}
                    
                    response = requests.post(API_URL, files=files, data=data)
                    
                    if response.status_code == 200:
                        processed_img = Image.open(io.BytesIO(response.content))
                        st.image(processed_img, use_container_width=True)
                    else:
                        st.error(f"Backend Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")