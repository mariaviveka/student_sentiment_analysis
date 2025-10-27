import streamlit as st
import pandas as pd
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import json

# ----------------- PAGE CONFIG -----------------
st.set_page_config(layout="wide", page_title="ICTAK Student Feedback Dashboard")

# ----------------- DATA LOAD -----------------
@st.cache_data
def load_data():
    return pd.read_csv("Data/ICTAK_Student_Feedback_Dataset.csv")

df = load_data()

# Combine feedback & suggestions
df["text_full"] = (df["Feedback_Comment"].fillna("") + " " + df["Suggestions"].fillna("")).str.strip()

# ----------------- SENTIMENT ANALYSIS -----------------
st.sidebar.title("🔍 Analysis Settings")
analyzer = SentimentIntensityAnalyzer()

df["sentiment_score"] = df["text_full"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])


def label_sentiment(score):
    if score >= 0.3:
        return "Positive"
    elif score <= -0.3:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment_Label"] = df["sentiment_score"].apply(label_sentiment)

# ----------------- TOPIC MODELING -----------------
st.sidebar.subheader("Topic Modeling")
n_topics = st.sidebar.slider("Number of Topics", 3, 10, 6)

vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words="english", ngram_range=(1,2))
X = vectorizer.fit_transform(df["text_full"])

lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
lda_fit = lda.fit(X)
df["dominant_topic"] = lda_fit.transform(X).argmax(axis=1)

def get_top_words(model, feature_names, n_top_words=10):
    topics = {}
    for i, topic in enumerate(model.components_):
        top_words = [feature_names[j] for j in topic.argsort()[:-n_top_words-1:-1]]
        topics[f"Topic_{i}"] = top_words
    return topics

topics = get_top_words(lda_fit, vectorizer.get_feature_names_out(), 10)

# ----------------- DASHBOARD -----------------
st.title("📊 ICTAK Student Feedback — Sentiment & Topic Dashboard")
st.caption("Automatically analyzes student feedback to reveal sentiment trends and key discussion themes.")

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Responses", len(df))
col2.metric("Positive %", f"{100 * (df['Sentiment_Label']=='Positive').mean():.1f}%")
col3.metric("Negative %", f"{100 * (df['Sentiment_Label']=='Negative').mean():.1f}%")

# Sentiment distribution
st.subheader("Sentiment Distribution")
sent_counts = df["Sentiment_Label"].value_counts().reset_index()
sent_counts.columns = ["Sentiment", "Count"]
chart = alt.Chart(sent_counts).mark_bar().encode(
    x=alt.X("Sentiment:N", sort=["Positive", "Neutral", "Negative"]),
    y="Count:Q",
    color="Sentiment:N",
    tooltip=["Sentiment", "Count"]
)
st.altair_chart(chart, use_container_width=True)

# Sentiment by Course
st.subheader("Sentiment by Course")
course_sent = df.groupby(["Course_Name", "Sentiment_Label"]).size().reset_index(name="count")
chart2 = alt.Chart(course_sent).mark_bar().encode(
    x=alt.X("Course_Name:N", sort=None),
    y="count:Q",
    color="Sentiment_Label:N",
    tooltip=["Course_Name", "Sentiment_Label", "count"]
).properties(height=400)
st.altair_chart(chart2, use_container_width=True)

# Topic Overview
st.subheader("Topic Overview")
topic_counts = df["dominant_topic"].value_counts().sort_index().reset_index()
topic_counts.columns = ["Topic", "Count"]
topic_counts["Topic"] = topic_counts["Topic"].apply(lambda x: f"Topic_{x}")
st.bar_chart(topic_counts.set_index("Topic"))

st.write("### 🗂 Top Words per Topic")
for t, words in topics.items():
    st.write(f"**{t}:** {', '.join(words)}")

# Explore responses
st.subheader("Explore Individual Responses")
course_options = ["All"] + sorted(df["Course_Name"].unique().tolist())
selected_course = st.selectbox("Filter by Course", course_options)
sentiment_options = ["All", "Positive", "Neutral", "Negative"]
selected_sentiment = st.selectbox("Filter by Sentiment", sentiment_options)

filtered = df.copy()
if selected_course != "All":
    filtered = filtered[filtered["Course_Name"] == selected_course]
if selected_sentiment != "All":
    filtered = filtered[filtered["Sentiment_Label"] == selected_sentiment]

st.dataframe(
    filtered[
        ["Student_ID","Student_Name","Course_Name","Sentiment_Label","sentiment_score","dominant_topic","Feedback_Comment","Suggestions"]
    ].reset_index(drop=True)
)

# Download analyzed CSV
st.download_button(
    "📥 Download Analyzed CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="ICTAK_Student_Feedback_Analyzed.csv",
    mime="text/csv"
)

st.success("✅ Analysis complete — sentiment and topic insights ready.")

