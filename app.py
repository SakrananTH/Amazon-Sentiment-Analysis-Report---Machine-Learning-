import streamlit as st
import joblib
import time
import os
import pandas as pd
import pythainlp

# --- Custom Tokenizer (Required for joblib to load the model) ---
def custom_tokenizer(text):
    return pythainlp.word_tokenize(text, engine='newmm', keep_whitespace=False)
# ---------------------------------------------------------------

# --- Page Config ---
st.set_page_config(
    page_title="เครื่องมือวิเคราะห์ความรู้สึกจากรีวิว",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for White/Clean Theme ---
st.markdown("""
    <style>
    /* Force white background */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Clean Cards */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    /* Input areas */
    .stTextArea textarea {
        background-color: #fcfcfc;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #ffffff;
        color: #333;
        border: 1px solid #ddd;
        border-radius: 8px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        border-color: #007bff;
        color: #007bff;
        background-color: #f0f8ff;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Success/Error messages */
    .stSuccess, .stError {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Model Function ---
@st.cache_resource
def load_model():
    model_path = 'sentiment_model.pkl'
    if not os.path.exists(model_path):
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        return None

# --- Load Data Function ---
@st.cache_data
def load_data():
    csv_path = '9.synthetic_amazon_like_reviews_subset_5000.csv'
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
        except:
            return None
    return None

# --- Main App ---
def main():
    col_main, col_padding = st.columns([3, 1])
    
    with col_main:
        st.title("🛍️ เครื่องมือวิเคราะห์ความรู้สึกจากรีวิว")
        st.caption("ระบบปัญญาประดิษฐ์เพื่อจำแนกความคิดเห็นเชิงบวกและเชิงลบ")

    # Sidebar for Data Preview
    with st.sidebar:
        st.header("📂 ข้อมูลชุดสอน (Dataset)")
        st.write("ตัวอย่างข้อมูลที่ใช้สอนโมเดล:")
        df = load_data()
        if df is not None:
            st.dataframe(df[['text', 'label']].head(10), hide_index=True)
            st.caption(f"จำนวนข้อมูลทั้งหมด: {len(df):,} รายการ")
        else:
            st.warning("ไม่พบไฟล์ข้อมูล CSV")
        
        st.divider()
        st.info("💡 **Tips:** โมเดลนี้ถูกสอนด้วยรีวิวสินค้า Amazon ภาษาอังกฤษ")

    st.divider()

    # Load the trained model
    pipeline = load_model()

    if pipeline is None:
        st.error("⚠️ ไม่พบไฟล์โมเดล ('sentiment_model.pkl')!")
        st.info("โปรดรัน `python train_model.py` ก่อนเพื่อสร้างโมเดล")
        st.stop()

    # --- Main Analysis Section ---
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("📝 พื้นที่วิเคราะห์ข้อความ")
        user_input = st.text_area(
            "พิมพ์รีวิวสินค้า (ภาษาอังกฤษ) ที่นี่:", 
            height=200,
            placeholder="Ex: This product is amazing! I love it so much."
        )

    with col2:
        st.subheader("⚡ ตัวอย่างด่วน")
        st.info("คลิกเพื่อลองข้อความตัวอย่าง:")
        
        examples = [
            "Great product! Works perfectly.", 
            "Terrible quality. Waste of money.",
            "It's okay, but delivery was late.",
            "Absolutely loved it, will buy again!"
        ]
        
        for ex in examples:
            if st.button(f"👉 {ex[:30]}...", key=ex):
                user_input = ex
                # Use session state or rerun would be better, but for simplicity:
                st.write(f"*(คัดลอก: '{ex}' ไปยังช่องว่างแล้ว กดวิเคราะห์ได้เลย)*")


    # Analyze Action
    st.markdown("###")
    analyze_btn = st.button("🔍 วิเคราะห์ความรู้สึก (Analyze Sentiment)", type="primary", use_container_width=True)

    if analyze_btn or (user_input and len(user_input) > 5):
        if not user_input.strip():
            st.warning("⚠️ กรุณาพิมพ์ข้อความก่อนกดวิเคราะห์")
        else:
            # --- Inference ---
            with st.spinner('กำลังประมวลผล...'):
                start_time = time.time()
                try:
                    # Predict
                    prediction = pipeline.predict([user_input])[0]
                    confidence = pipeline.predict_proba([user_input]).max()
                    
                    end_time = time.time()
                    latency = (end_time - start_time) * 1000
                    
                    # --- Display Results ---
                    st.markdown("---")
                    
                    r_col1, r_col2 = st.columns([1, 2])
                    
                    with r_col1:
                        if prediction == "Positive":
                            st.success(f"### ผลลัพธ์: เชิงบวก 😃")
                        else:
                            st.error(f"### ผลลัพธ์: เชิงลบ 😡")
                    
                    with r_col2:
                         st.write(f"**ความมั่นใจของการทำนาย:** {confidence*100:.1f}%")
                         st.write(f"**เวลาในการประมวลผล:** {latency:.2f} มิลลิวินาที")
                         st.progress(confidence)

                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาด: {e}")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #aaa; font-size: 12px;'>MLDS Project • Powered by Streamlit & Scikit-learn</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()