import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------
# ส่วนที่ 1: การตั้งค่าคอนฟิกและการตกแต่ง UI โทนโมเดิร์น
# ----------------------------------------------------
st.set_page_config(
    page_title="Information Retrieval System Sandbox",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ใช้ Custom CSS เล็กน้อยเพื่อขับให้ UI ดูพรีเมียม ไร้กลิ่นอายเทมเพลตดิบๆ
st.markdown("""
    <style>
    .main { background-color: #fcfcfd; }
    h1 { color: #1e293b; font-weight: 800 !important; letter-spacing: -0.5px; }
    h3 { color: #334155; font-weight: 600 !important; }
    .stButton>button {
        width: 100%;
        background-color: #0f172a;
        color: white;
        border-radius: 6px;
        border: none;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #334155;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🖥️ ระบบแบบจำลองการค้นคืนสารสนเทศ")
st.markdown("ระบบจำลองสถาปัตยกรรมทางวิศวกรรมข้อมูลตามโมเดล **Information Retrieval Pipeline**")
st.markdown("---")

# ----------------------------------------------------
# ส่วนที่ 2: Engine ประมวลผลดรรชนีและโครงสร้าง AI
# ----------------------------------------------------
@st.cache_resource
def load_ai_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_ai_model()

# ----------------------------------------------------
# ส่วนที่ 3: คลังสารสนเทศ (Document Store)
# ----------------------------------------------------
if 'documents' not in st.session_state:
    st.session_state.documents = [
        "การเรียนรู้ของเครื่อง (Machine Learning) เป็นส่วนหนึ่งของ AI ที่ช่วยให้ระบบคิดเองได้",
        "ระบบการค้นคืนสารสนเทศ (Information Retrieval) ช่วยให้เราค้นหาเอกสารที่ต้องการได้รวดเร็ว",
        "ภาษา Python เป็นภาษาที่นิยมมากที่สุดในการพัฒนา AI และวิเคราะห์ข้อมูล Data Science",
        "การประมวลผลภาษาธรรมชาติ (NLP) คือการทำให้คอมพิวเตอร์เข้าใจภาษาที่มนุษย์ใช้พูดและเขียน",
        "วิธีการเดินทางไปสยามพารากอน สามารถนั่งรถไฟฟ้า BTS มาลงที่สถานีสยามได้เลย"
    ]

# ออกแบบระบบจัดเก็บสถานะดรรชนีเพื่อลด Latency (ไม่ Encode ใหม่โดยไม่จำเป็น)
if 'doc_embeddings' not in st.session_state or len(st.session_state.documents) != len(st.session_state.get('last_indexed_docs', [])):
    st.session_state.doc_embeddings = model.encode(st.session_state.documents)
    st.session_state.last_indexed_docs = st.session_state.documents.copy()

# แบ่งพื้นที่ทำงานออกเป็น 2 คอลัมน์หลักตาม Architecture Diagram
col_left, col_right = st.columns([1.1, 0.9], gap="large")

# ====================================================
# [ฝั่งขวา] Document Store & Indexing Model
# ====================================================
with col_right:
    with st.container(border=True):
        st.markdown("### 📁 Document Store & Indexing Model")
        st.caption("ทำหน้าที่จัดเก็บและแปลงคลังข้อความสารสนเทศให้อยู่ในรูปเวกเตอร์ดรรชนี")
        
        # ฟังก์ชันเพิ่มข้อมูลเข้าระบบ
        new_doc = st.text_input("➕ เพิ่มระเบียบข้อมูลใหม่เข้าสู่คลัง (Ingestion):", placeholder="พิมพ์ข้อความที่ต้องการจัดเก็บ...")
        if st.button("ดำเนินการจัดทำดรรชนี (Index Document)"):
            if new_doc.strip():
                st.session_state.documents.append(new_doc.strip())
                st.toast("เพิ่มข้อมูลและอัปเดตดรรชนีเรียบร้อย!", icon="💾")
                st.rerun()
        
        st.markdown("**รายการดรรชนีปัจจุบันในระบบ (Registered Vector Indexes):**")
        for i, doc in enumerate(st.session_state.documents):
            st.markdown(f"""
            <div style="background-color: #f1f5f9; padding: 10px 14px; border-radius: 6px; margin-bottom: 8px; border-left: 4px solid #64748b;">
                <span style="font-size: 12px; color: #64748b; font-weight: bold;">DOC_ID_{i}</span><br>
                <span style="color: #334155; font-size: 14px;">{doc}</span>
            </div>
            """, unsafe_allow_html=True)

# ====================================================
# [ฝั่งซ้าย] User Interface, Query Processing & Engine
# ====================================================
with col_left:
    st.markdown("### 👤 User Query Interface")
    query = st.text_input("🔍 ป้อนคำค้นหาเพื่อเข้าสู่ระบบ Pipeline (Query):", placeholder="ระบุคีย์เวิร์ด หรือประโยคคำถามที่ต้องการค้นหา...")
    
    if query:
        # 1. Query Processing
        query_embedding = model.encode([query])
        
        # 2. Search & Match + 3. Ranking Engine
        similarity_scores = cosine_similarity(query_embedding, st.session_state.doc_embeddings)[0]
        ranked_indices = np.argsort(similarity_scores)[::-1]
        
        # แสดงแถบ Pipeline ข้อมูลเบื้องหลังแบบเท่ๆ ดูเหมือนระบบหลังบ้านของโปรแกรมเมอร์จริงๆ
        with st.expander("🛠️ ระบบตรวจสอบสเตตัสการประมวลผล (Pipeline Execution Logs)", expanded=False):
            st.code(f"""
[STATUS] Query Received: "{query}"
[STEP 1] Query Processing Complete -> Matrix Generated.
[STEP 2] Similarity Calculation Complete -> Vector Spaced Matrix Checked.
[STEP 3] Matrix Ranking Engine Complete -> Document Sorted.
            """, language="bash")
            
        st.markdown("---")
        st.markdown("### 🎯 ผลลัพธ์เอกสารผ่านการจัดอันดับ (Ranked Results)")
        
        for idx in ranked_indices:
            score = similarity_scores[idx]
            
            # กรองเกณฑ์ความแม่นยำความใกล้เคียง
            if score > 0.35:
                st.markdown(f"""
                <div style="background-color: #f8fafc; padding: 16px; border-radius: 8px; margin-bottom: 12px; border: 1px solid #e2e8f0; border-left: 5px solid #0f172a;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="background-color: #e2e8f0; color: #0f172a; font-size: 11px; padding: 2px 8px; border-radius: 4px; font-weight: bold;">MATCH ID: {idx}</span>
                        <span style="color: #0f172a; font-weight: 700; font-size: 13px;">Relevance Score: {score:.4f}</span>
                    </div>
                    <div style="color: #1e293b; font-size: 15px; line-height: 1.5;">{st.session_state.documents[idx]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # เอกสารที่ไม่ผ่านเกณฑ์ขั้นต่ำจะถูกกรองเป็นสีจางเพื่อให้ผู้ใช้รู้ว่าระบบฉลาดพอที่จะคัดออก
                st.markdown(f"""
                <div style="background-color: #ffffff; padding: 10px; border-radius: 6px; margin-bottom: 8px; border: 1px dashed #e2e8f0; opacity: 0.4;">
                    <span style="font-size: 12px; color: #94a3b8;">[Filtered Out - Score: {score:.4f}] {st.session_state.documents[idx]}</span>
                </div>
                """, unsafe_allow_html=True)