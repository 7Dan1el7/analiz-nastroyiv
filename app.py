import streamlit as st
from transformers.pipelines import pipeline
import pandas as pd
import plotly.express as px

@st.cache_resource
def load_classifier():
    return pipeline("sentiment-analysis",
                    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

classifier = load_classifier()

st.title("Аналіз настроїв тексту")

st.header("Введіть текст для аналізу")
user_text = st.text_area("Текст (кілька рядків для масового аналізу):")

if st.button("Аналізувати"):
    if user_text:
        texts = [t.strip() for t in user_text.split("\n") if t.strip()]
        results = classifier(texts)

        df = pd.DataFrame({
            "Текст": texts,
            "Настрій": [r['label'] for r in results],
            "Ймовірність": [round(r['score'], 4) for r in results]
        })

        st.subheader("📊 Результати аналізу")
        st.dataframe(df)

        st.subheader("📈 Візуалізація")
        fig = px.pie(df, names="Настрій", values="Ймовірність", title="Розподіл настроїв")
        st.plotly_chart(fig)
