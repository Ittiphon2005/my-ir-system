import streamlit as st
import numpy as np
import re
import math
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords

# ระบบควบคุมและรีเซ็ตหน่วยความจำเก่า (Session State Guard)
# เพื่อป้องกันบราวเซอร์จำคำศัพท์ที่มีเครื่องหมายวงเล็บค้างจากโค้ดเวอร์ชันเก่า
if 'ir_theory_verified_v6' not in st.session_state:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.ir_theory_verified_v6 = True

st.set_page_config(
    page_title="Thai TF-IDF Information Retrieval System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ตกแต่ง UI ให้สวยงาม scannable ง่ายต่อการตรวจทาน
st.markdown("""
    <style>
    .main { background-color: #fcfcfd; }
    h1 { color: #1e293b; font-weight: 800 !important; }
    h3 { color: #334155; font-weight: 600 !important; margin-top: 15px; }
    .stButton>button { width: 100%; background-color: #0f172a; color: white; border-radius: 6px; }
    .stButton>button:hover { background-color: #334155; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("🖥️ ระบบแบบจำลองการค้นคืนสารสนเทศ (Thai TF-IDF Engine)")
st.markdown("💡 **สถาปัตยกรรมข้อมูล:** Vector Space Model ที่ได้รับการปรับปรุงค่าน้ำหนักคำศัพท์ (Term Weighting) ตามทฤษฎี IR มาตรฐาน")
st.markdown("---")

# ====================================================
# [แก้ไขข้อที่ 2] ทฤษฎี TEXT PREPROCESSING & DATA CLEANING
# ก่อนคำนวณ IDF เอกสารทั้งหมดต้องสะอาด เพื่อป้องกันค่า Document Frequency (DF) ผิดเพี้ยน
# ====================================================
def text_pipeline(text):
    text = text.strip().lower()
    
    # ดำเนินการลบสัญลักษณ์พิเศษ เครื่องหมายวรรคตอน และวงเล็บ () [] {} ออกทั้งหมดก่อนตัดคำ
    # หากไม่ลบ วงเล็บจะติดไปกับคำ เช่น '(machine' ทำให้นับคำซ้ำซ้อนและค่า IDF เพี้ยน
    text = re.sub(r'[\(\)\[\]\{\}\-\+\,\.\?]', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Tokenization: ตัดคำภาษาไทยด้วย PyThaiNLP
    raw_words = word_tokenize(text, engine="newmm")
    
    # Stop Words Removal: กำจัดคำที่ไม่มีความหมายเด่นชัดในเชิงเนื้อหา
    thai_stop = set(thai_stopwords())
    custom_stop = {"การ", "ความ", "ของ", "ใน", "และ", "เป็น", "ที่", "มี", "ให้", "ได้", "จะ", "มา", "ลง", "เลย", "สามารถ", "ช่วย", "เรา", "คือ", " ", ""}
    all_stop_words = thai_stop.union(custom_stop)
    
    filtered_words = []
    for w in raw_words:
        w = w.strip()
        if w and w not in all_stop_words:
            if len(w) > 1:
                # กรองเศษพยางค์ที่ระบบตัดคำหั่นพลาด ป้องกันพยัญชนะลอยตัวเดี่ยว
                if re.match(r'^[ก-ฮ]{2,3}$', w) and w not in ["พบ", "คน", "จน", "รบ", "สม", "ดม", "คอม", "บีทีเอส"]:
                    continue
                filtered_words.append(w)
            elif w.isalnum() and not re.match(r'[ก-ฮ]', w): 
                filtered_words.append(w)
            
    return filtered_words

# ====================================================
# [แก้ไขข้อที่ 1, 2, 3] ทฤษฎี TERM WEIGHTING ENGINE FOR DOCUMENT CORPUS
# ====================================================
def calculate_tf_idf_system(docs):
    N = len(docs)
    doc_tokens = [text_pipeline(doc) for doc in docs]
    
    # รวบรวมคลังคำศัพท์ทั้งหมดในระบบ (Vocabulary)
    all_terms = set()
    for tokens in doc_tokens:
        all_terms.update(tokens)
    all_terms = sorted(list(all_terms))
    
    # --- [ทฤษฎีข้อที่ 2: Inverse Document Frequency (IDF)] ---
    # คำนวณหาค่า Document Frequency (DF) หรือจำนวนเอกสารที่พบคำศัพท์นั้น ๆ
    df_counts = {term: 0 for term in all_terms}
    for tokens in doc_tokens:
        for term in set(tokens):
            df_counts[term] += 1
            
    # คำนวณค่าค่าน้ำหนักความเฉพาะเจาะจงของคำ (IDF) 
    # ใช้สูตรมาตรฐานตามทฤษฎี IR: IDF = log10(N / DF) + 1
    idf_weights = {}
    for term, df in df_counts.items():
        idf_weights[term] = math.log10(N / df) + 1
        
    # --- [ทฤษฎีข้อที่ 1 & 3: Term Frequency & TF-IDF ในคลังเอกสาร] ---
    doc_vectors = []
    for doc_id, tokens in enumerate(doc_tokens):
        total_words = len(tokens)
        term_counts = {}
        for token in tokens:
            term_counts[token] = term_counts.get(token, 0) + 1
            
        vector = {}
        for term in all_terms:
            if term in term_counts and total_words > 0:
                # 1. Term Frequency (TF) สำหรับเอกสารขนาดยาวใช้แบบ Relative Frequency (นับคำ / คำทั้งหมด)
                tf = term_counts[term] / total_words
                # 3. ผลคูณสุทธิ TF-IDF Weight สำหรับสร้างสารสนเทศดรรชนีผกผัน
                vector[term] = tf * idf_weights[term]
            else:
                vector[term] = 0.0
        doc_vectors.append(vector)
        
    return all_terms, idf_weights, doc_vectors

# ----------------------------------------------------
# คลังเอกสารตั้งต้นสำหรับการทดสอบระบบ (Document Store)
# ----------------------------------------------------
if 'documents_v6_store' not in st.session_state:
    st.session_state.documents_v6_store = [
        "การเรียนรู้ของเครื่อง (Machine Learning) เป็นส่วนหนึ่งของ AI ที่ช่วยให้ระบบคิดเองได้",
        "ระบบการค้นคืนสารสนเทศ (Information Retrieval) ช่วยให้เราค้นหาเอกสารที่ต้องการได้รวดเร็ว",
        "ภาษา Python เป็นภาษาที่นิยมมากที่สุดในการพัฒนา AI และวิเคราะห์ข้อมูล Data Science",
        "การประมวลผลภาษาธรรมชาติ (NLP) คือการทำให้คอมพิวเตอร์เข้าใจภาษาที่มนุษย์ใช้พูดและเขียน",
        "วิธีการเดินทางไปสยามพารากอน สามารถนั่งรถไฟฟ้า BTS มาลงที่สถานีสยามได้เลย"
    ]

# ประมวลผลสร้างดรรชนีคำศัพท์และคำนวณน้ำหนักตัวแบบคลังข้อมูล
vocab, idf_table, doc_tfidf_vectors = calculate_tf_idf_system(st.session_state.documents_v6_store)

# แยกการจัดวางหน้าจอออกเป็น 2 ฝั่งเพื่อความสวยงามและ scannable 
col_left, col_right = st.columns([1.1, 0.9], gap="large")

# ====================================================
# [ขวา] ส่วนจัดการดรรชนีคลังข้อมูล และแสดงตาราง IDF Table
# ====================================================
with col_right:
    with st.container(border=True):
        st.markdown("### 📁 Document Store & Weighting Model")
        st.caption("ส่วนควบคุมการจัดเก็บคลังข้อมูลสารสนเทศดิบและการคำนวณค่าดรรชนีผกผัน")
        
        new_doc = st.text_input("➕ เพิ่มระเบียบข้อมูลใหม่เข้าสู่คลัง (Ingestion):", placeholder="พิมพ์ประโยคภาษาไทยที่ต้องการเพิ่ม...")
        if st.button("ดำเนินการจัดทำดรรชนี (Index Document)"):
            if new_doc.strip():
                st.session_state.documents_v6_store.append(new_doc.strip())
                st.toast("อัปเดตคลังข้อมูลใหม่เรียบร้อย!", icon="💾")
                st.rerun()
        
        st.markdown("**ตารางค่าน้ำหนักคำและดรรชนีผกผัน (IDF Table):**")
        st.caption("💡 หลักทฤษฎี: คำทั่วไปที่พบบ่อยค่า IDF จะต่ำ ส่วนคำเฉพาะทางค่า IDF จะสูง")
        
        for term in vocab:
            st.markdown(f"""
            <div style="background-color: #f1f5f9; padding: 6px 12px; border-radius: 6px; margin-bottom: 4px; border-left: 4px solid #0f172a; display: flex; justify-content: space-between;">
                <span style="font-weight: bold; color: #0f172a;">{term}</span>
                <span style="color: #2563eb; font-size: 13px; font-weight: bold;">ค่า IDF: {idf_table[term]:.4f}</span>
            </div>
            """, unsafe_allow_html=True)

# ====================================================
# [ซ้าย] ส่วนรับข้อมูลสืบค้น การประมวลผลคำค้นหา และการจับคู่ผลลัพธ์
# ====================================================
with col_left:
    st.markdown("### 👤 User Query Interface")
    query = st.text_input("🔍 ป้อนคำค้นหาเพื่อเข้าสู่ระบบ Pipeline (Query):", placeholder="เช่น เรียนรู้และวิเคราะห์ข้อมูล AI")
    
    if query:
        query_tokens = text_pipeline(query)
        
        if query_tokens:
            query_vector = {term: 0.0 for term in vocab}
            
            q_counts = {}
            for t in query_tokens:
                if t in vocab:
                    q_counts[t] = q_counts.get(t, 0) + 1
            
            # --- [แก้ไขข้อที่ 1: การให้น้ำหนักคำตามความถี่ (TF) ฝั่งคำค้นหา] ---
            # ตามทฤษฎีคำค้นหา (Query) มักสั้นมาก การหารด้วยความยาวประโยคจะทำให้ค่าน้ำหนักเพี้ยน 
            # จึงปรับใช้เกณฑ์ Raw TF / Binary TF แทน (ถ้าพบคำในคำค้นหา ให้มีค่าความถี่ตั้งต้นเท่ากับ 1)
            # แล้วคูณด้วยค่า IDF ของคำศัพท์นั้นโดยตรง ช่วยแก้บั๊กตัวเลขทศนิยมสลับตำแหน่งกันได้อย่างสมบูรณ์
            for t in vocab:
                if t in q_counts:
                    # สูตรน้ำหนักฝั่ง Query: TF (ซึ่งมีค่าเท่ากับ 1) * IDF ของคำนั้นๆ
                    query_vector[t] = 1.0 * idf_table[t]

            with st.expander("🛠️ ระบบตรวจสอบสเตตัสการประมวลผล (Pipeline Term Weighting Logs)", expanded=True):
                st.markdown(f"**คำค้นหาหลังคัดกรอง (Tokenized):** `{query_tokens}`")
                active_weights = {k: f"{v:.4f}" for k, v in query_vector.items() if float(v) > 0}
                st.json({"ค่าน้ำหนักคีย์เวิร์ดในคำค้นหา (Query TF-IDF Vector)": active_weights})
                
            st.markdown("---")
            st.markdown("### 🎯 ผลลัพธ์เอกสารผ่านการจัดอันดับ (Ranked Results)")
            
            # คำนวณหาคะแนนความคล้ายคลึงของเวกเตอร์ (Vector Space Dot Product Match)
            scores = []
            for doc_id in range(len(st.session_state.documents_v6_store)):
                score = 0.0
                doc_vec = doc_tfidf_vectors[doc_id]
                for term in vocab:
                    score += query_vector[term] * doc_vec[term]
                scores.append(score)
            
            # เรียงลำดับเอกสารตามคะแนนความคล้ายคลึงจากมากไปหาน้อย
            ranked_indices = np.argsort(scores)[::-1]
            has_results = False
            for idx in ranked_indices:
                if scores[idx] > 0:
                    has_results = True
                    st.markdown(f"""
                    <div style="background-color: #f8fafc; padding: 16px; border-radius: 8px; margin-bottom: 12px; border: 1px solid #e2e8f0; border-left: 5px solid #2563eb;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="background-color: #dbeafe; color: #2563eb; font-size: 11px; padding: 2px 8px; border-radius: 4px; font-weight: bold;">MATCH ID: {idx}</span>
                            <span style="color: #2563eb; font-weight: 700; font-size: 13px;">คะแนนน้ำหนักสุทธิ (TF-IDF Score): {scores[idx]:.4f}</span>
                        </div>
                        <div style="color: #1e293b; font-size: 15px; line-height: 1.5;">{st.session_state.documents_v6_store[idx]}</div>
                    </div>
                    """, unsafe_allow_html=True)
            if not has_results:
                st.warning("ไม่พบเอกสารที่มีค่าน้ำหนักตรงกับคำค้นหาในระบบ")
        else:
            st.info("กรุณาป้อนคำค้นหาที่ไม่ใช่คำหยุด (Stop Words)")