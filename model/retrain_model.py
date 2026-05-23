import pandas as pd
import sqlite3
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from xgboost import XGBClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
import re
from datetime import datetime

nltk.download('stopwords')

ps = PorterStemmer()

def text_processing(text):
    text = str(text)
    token = WordPunctTokenizer()
    stop_words = set(stopwords.words("english"))

    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"[0-9]", " ", text)

    text = [ps.stem(word) for word in token.tokenize(text.lower()) if word not in stop_words]

    return " ".join(text)

class FeedbackModelRetrainer:
    def __init__(self, db_path, original_model_path, original_vectorizer_path):
        self.db_path = db_path
        self.original_model_path = original_model_path
        self.original_vectorizer_path = original_vectorizer_path

        self.tfidf = pickle.load(open(original_vectorizer_path, "rb"))
        self.model = pickle.load(open(original_model_path, "rb"))

    def extract_feedback_data(self, min_confidence=3):
        conn = sqlite3.connect(self.db_path)

        query = f"""
        SELECT claim_text, correct_classification, confidence_level, prediction_result
        FROM user_feedback
        WHERE correct_classification IN ('Genuine', 'Fake')
        AND confidence_level >= {min_confidence}
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            print("No feedback data available for retraining.")
            return None

        df['class'] = df['correct_classification'].map({'Genuine': 1, 'Fake': 0})

        df['weight'] = df['confidence_level'] / 5

        feedback_texts = df['claim_text'].tolist()
        feedback_classes = df['class'].tolist()
        feedback_weights = df['weight'].tolist()

        print(f"Extracted {len(feedback_texts)} feedback entries for retraining.")

        return {
            'texts': feedback_texts,
            'classes': feedback_classes,
            'weights': feedback_weights,
            'raw_data': df
        }

    def analyze_model_accuracy(self, feedback_data):
        df = feedback_data['raw_data']

        df['model_correct'] = df.apply(
            lambda row: (row['prediction_result'] == 'Genuine' and row['class'] == 1) or
                       (row['prediction_result'] == 'Fake' and row['class'] == 0),
            axis=1
        )

        agreement_rate = df['model_correct'].mean() * 100

        confidence_agreement = df.groupby('confidence_level')['model_correct'].agg(['mean', 'count'])
        confidence_agreement['mean'] = confidence_agreement['mean'] * 100  # Convert to percentage

        return {
            'overall_agreement_rate': agreement_rate,
            'confidence_agreement': confidence_agreement,
            'total_feedback_entries': len(df)
        }

    def retrain_model(self, original_data_path=None, output_dir='./model_output', blend_ratio=0.5):

        os.makedirs(output_dir, exist_ok=True)

        feedback_data = self.extract_feedback_data()
        if not feedback_data:
            print("No feedback data available. Skipping retraining.")
            return False

        processed_feedback_texts = [text_processing(text) for text in feedback_data['texts']]

        if original_data_path:
            original_df = pd.read_csv(original_data_path)
            original_df = original_df.dropna(subset=['title_text'])
            processed_original_texts = [text_processing(text) for text in original_df['title_text']]
            original_classes = original_df['class'].tolist()
            n_feedback = len(processed_feedback_texts)
            n_original = min(int(n_feedback * (blend_ratio / (1 - blend_ratio))), len(processed_original_texts))

            indices = np.random.choice(len(processed_original_texts), size=n_original, replace=False)
            sampled_original_texts = [processed_original_texts[i] for i in indices]
            sampled_original_classes = [original_classes[i] for i in indices]

            all_texts = processed_feedback_texts + sampled_original_texts
            all_classes = feedback_data['classes'] + sampled_original_classes

            sample_weights = feedback_data['weights'] + [0.5] * len(sampled_original_texts)

            print(f"Combined {n_feedback} feedback entries with {n_original} original data entries.")
        else:
            all_texts = processed_feedback_texts
            all_classes = feedback_data['classes']
            sample_weights = feedback_data['weights']

            print(f"Using only {len(all_texts)} feedback entries for retraining.")

        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            all_texts, all_classes, sample_weights, test_size=0.2, random_state=42
        )
        X_train_vec = self.tfidf.transform(X_train)
        X_val_vec = self.tfidf.transform(X_val)

        xgb = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic'
        )

        xgb.fit(X_train_vec, y_train, sample_weight=w_train)

        val_accuracy = xgb.score(X_val_vec, y_val)
        print(f"Validation accuracy: {val_accuracy:.4f}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(output_dir, f"model_retrained_{timestamp}.pkl")
        vectorizer_path = os.path.join(output_dir, f"vectorizer_retrained_{timestamp}.pkl")

        pickle.dump(xgb, open(model_path, "wb"))
        pickle.dump(self.tfidf, open(vectorizer_path, "wb"))

        print(f"Saved retrained model to {model_path}")
        print(f"Saved retrained vectorizer to {vectorizer_path}")

        return {
            'model_path': model_path,
            'vectorizer_path': vectorizer_path,
            'validation_accuracy': val_accuracy,
            'num_feedback_entries': len(processed_feedback_texts),
            'timestamp': timestamp
        }

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR, "feedback_database.db")
    MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
    VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
    ORIGINAL_DATA_PATH = os.path.join(BASE_DIR, "../processed/dataset.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "model_output")

    retrainer = FeedbackModelRetrainer(DB_PATH, MODEL_PATH, VECTORIZER_PATH)

    feedback_data = retrainer.extract_feedback_data(min_confidence=3)

    if feedback_data:
        accuracy_analysis = retrainer.analyze_model_accuracy(feedback_data)
        print(f"Model-user agreement rate: {accuracy_analysis['overall_agreement_rate']:.2f}%")
        print(f"Total feedback entries: {accuracy_analysis['total_feedback_entries']}")

        retraining_results = retrainer.retrain_model(
            original_data_path=ORIGINAL_DATA_PATH,
            output_dir=OUTPUT_DIR,
            blend_ratio=0.7
        )

        if retraining_results:
            print("Retraining completed successfully!")
            print(f"New model validation accuracy: {retraining_results['validation_accuracy']:.4f}")
    else:
        print("Not enough feedback data to retrain model. Please collect more feedback.")