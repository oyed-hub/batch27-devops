import streamlit as st
import pickle
import string


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = word_tokenize(text)

    cleaned_words = []

    for word in words:
        if word.isalpha():
            if word not in stop_words:
                cleaned_words.append(
                    ps.stem(word)
                )

    return " ".join(cleaned_words)

model = pickle.load(
    open("models/spam_model.pkl", "rb")
)

tfidf = pickle.load(
    open("models/tfidf.pkl", "rb")
)

st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📧 AI Spam Email Detector")

st.markdown("""
Detect whether an **Email** or **SMS** is **Spam** or **Ham**
using Machine Learning and Natural Language Processing.
""")

st.divider()

st.write(
    "Detect whether an Email or SMS is Spam or Ham."
)

email = st.text_area(
    "📩 Enter Email or SMS",
    height=220,
    placeholder="Paste your email here..."
)

if st.button("Predict"):

    processed_email = preprocess_text(email)

    email_vector = tfidf.transform(
        [processed_email]
    )

    prediction = model.predict(
        email_vector
    )

    probability = model.predict_proba(
        email_vector
    )

    if prediction[0] == 1:

        st.error("🚨 Spam Email Detected")

    else:

        st.success("✅ Legitimate Email")

    spam_probability = probability[0][1] * 100
    ham_probability = probability[0][0] * 100
    

    st.subheader("Prediction Probability")
    st.write(f"Spam : {spam_probability:.2f}%")
    st.progress(spam_probability/100)
    st.write(f"Ham : {ham_probability:.2f}%")
    st.progress(ham_probability/100)


    col1,col2,col3 = st.columns(3)
    col1.metric(
    "Characters",
    len(email)
    )
    col2.metric(
    "Words",
    len(email.split())
    )
    col3.metric(
    "Sentences",
    email.count(".")+email.count("!")+email.count("?")
    )

st.sidebar.header("Project")

st.sidebar.write("""
Spam Email Detector

Machine Learning + NLP

Python

Scikit-Learn

TF-IDF

Logistic Regression
""")

st.sidebar.success("Version 1.0")
st.sidebar.subheader("Try These")

st.sidebar.info("""
Spam

Congratulations!

You won ₹50,000.

Click here.

--------------------

Ham

Hi Madhu,

Meeting at 10 AM tomorrow.
""")
st.divider()

st.caption(
"Built using Python • Streamlit • Machine Learning • NLP"
)


