import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pythainlp

def custom_tokenizer(text):
    # Use PyThaiNLP for word segmentation
    # It handles both Thai and English (English words are kept as tokens)
    return pythainlp.word_tokenize(text, engine='newmm', keep_whitespace=False)

def train_and_save_model():
    # --- Configuration ---
    csv_filename_en = '9.synthetic_amazon_like_reviews_subset_5000.csv'
    csv_filename_th = '9.synthetic_amazon_reviews_thai.csv'
    model_filename = 'sentiment_model.pkl'

    print(f"🔄 Starting Process...")
    
    # 1. Load Datasets
    dfs = []
    
    # Load English
    if os.path.exists(csv_filename_en):
        print(f"📂 Loading English dataset: {csv_filename_en}")
        df_en = pd.read_csv(csv_filename_en)
        # Map labels to consistent format if needed
        # Assuming format is 'Positive'/'Negative' or similar
        dfs.append(df_en)
    else:
        print(f"⚠️ Warning: English dataset '{csv_filename_en}' not found.")

    # Load Thai
    if os.path.exists(csv_filename_th):
        print(f"📂 Loading Thai dataset: {csv_filename_th}")
        df_th = pd.read_csv(csv_filename_th)
        # The Thai dataset has columns: review_id,text,label,...
        # IMPORTANT: Map Thai labels to English labels for consistency
        label_map = {'บวก': 'Positive', 'ลบ': 'Negative'}
        df_th['label'] = df_th['label'].map(label_map).fillna(df_th['label'])
        dfs.append(df_th)
    else:
        print(f"⚠️ Warning: Thai dataset '{csv_filename_th}' not found.")

    if not dfs:
        print("❌ Error: No datasets found!")
        return

    # Combine
    df = pd.concat(dfs, ignore_index=True)
    print(f"✅ Combined Datasets: {len(df)} total reviews.")

    # 2. Prepare Data
    # Ensure no missing values in text column
    df = df.dropna(subset=['text', 'label'])
    
    X = df['text']
    y = df['label']

    print(f"📊 Splitting data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Create Pipeline (TF-IDF + Logistic Regression)
    # Use custom tokenizer for Thai support
    # ngram_range=(1,2) helps with negation for both languages
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=custom_tokenizer, 
            lowercase=True, 
            ngram_range=(1, 2)
        )), 
        ('clf', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    ])

    # 4. Train Model
    print("🧠 Training model (this might take a few seconds)...")
    pipeline.fit(X_train, y_train)
    print("✅ Training complete.")

    # 5. Evaluate
    print("📉 Evaluating model performance...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Test Accuracy: {acc:.4f}")
    print("-" * 30)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 30)

    # 6. Save Model
    joblib.dump(pipeline, model_filename)
    print(f"💾 Model saved to: {os.path.abspath(model_filename)}")
    print("\n✅ READY FOR NEXT STEP:")
    print("   Run 'streamlit run app.py' to launch the website.")

if __name__ == "__main__":
    train_and_save_model()