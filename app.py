import re, nltk, numpy as np, streamlit as st
from collections import Counter
from pypdf import PdfReader
from nltk import word_tokenize, pos_tag, ne_chunk, bigrams
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from google import genai

MODEL = "gemini-3.7-flash"
st.set_page_config(page_title="NLP Chat with PDF", layout="wide")
st.title("📄 NLP-Based Chat with Any PDF")

key = st.sidebar.text_input("GOOGLE_API_KEY", type="password")
pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
k = st.sidebar.slider("Top-K chunks", 1, 10, 4)
if not key or pdf is None:
    st.info("Enter your Google API key and upload a PDF in the sidebar.")
    st.stop()

# ---- Unit I: preprocessing tools ----
@st.cache_resource
def setup():
    for p in ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger",
              "averaged_perceptron_tagger_eng", "maxent_ne_chunker", "maxent_ne_chunker_tab", "words"]:
        nltk.download(p, quiet=True)
    return PorterStemmer(), WordNetLemmatizer(), set(stopwords.words("english")), \
        SentenceTransformer("all-MiniLM-L6-v2")
stemmer, lemmatizer, STOP, model = setup()

def preprocess(text):
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())          # regex + normalization
    toks = [t for t in word_tokenize(text) if t not in STOP]  # tokenization + stop-words
    return toks, [stemmer.stem(t) for t in toks], [lemmatizer.lemmatize(t) for t in toks]

# ---- Load PDF -> chunks -> embeddings (Unit IV) ----
@st.cache_data(show_spinner="Reading PDF...")
def load_chunks(data, name):
    import io
    text = " ".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    text = re.sub(r"\s+", " ", text)
    return [text[i:i + 1000] for i in range(0, len(text), 800)]  # 1000 chars, 200 overlap

@st.cache_data(show_spinner="Building embeddings...")
def embed(chunks, name):
    return model.encode(chunks, normalize_embeddings=True)

chunks = load_chunks(pdf.getvalue(), pdf.name)
if not chunks or not "".join(chunks).strip():
    st.error("No text found in this PDF (it may be a scanned image).")
    st.stop()
vecs = embed(chunks, pdf.name)

def retrieve(q):
    sims = vecs @ model.encode([q], normalize_embeddings=True)[0]
    return [chunks[i] for i in np.argsort(-sims)[:k]]

def ask(question):
    context = "\n\n".join(retrieve(question))
    prompt = f"""You are a helpful document assistant. Using the document context below, answer the question in simple, clear language.
Quote relevant facts, names, numbers or values directly from the document. If the question asks about a term or concept
mentioned in the document, explain it using the document first, and add brief general background only if it helps.
If the document doesn't contain the needed information, say "I don't know based on the document."

Context:
{context}

Question: {question}"""
    client = genai.Client(api_key=key)
    for c in client.models.generate_content_stream(model=MODEL, contents=prompt):
        if c.text:
            yield c.text

tab1, tab2 = st.tabs(["💬 Chat with PDF", "🔬 NLP Analysis"])

with tab1:
    q = st.text_input("Ask anything about your PDF")
    c1, c2 = st.columns(2)
    try:
        if c1.button("Get Answer") and q:
            st.caption(f"Query keywords: {preprocess(q)[2]}")
            st.write_stream(ask(q))
        if c2.button("Summarize PDF"):
            st.write_stream(ask("Summarize this document: main points and key details."))
    except Exception as e:
        st.error(f"Gemini error: {e}")

with tab2:
    sample = " ".join(chunks)[:3000]
    toks, stems, lems = preprocess(sample)
    st.write("**Tokens:**", toks[:25])
    st.write("**Stemmed (Porter):**", stems[:25])
    st.write("**Lemmatized:**", lems[:25])
    try:
        tagged = pos_tag(word_tokenize(sample))
        st.write("**POS tags:**", tagged[:20])
        ents = [(" ".join(w for w, _ in t), t.label()) for t in ne_chunk(tagged) if hasattr(t, "label")]
        st.write("**Named Entities (NER):**", ents[:15])
    except Exception as e:
        st.warning(f"POS/NER unavailable: {e}")
    st.write("**Top bigrams (n-grams):**", [" ".join(b) for b, _ in Counter(bigrams(toks)).most_common(8)])
    try:
        tf = TfidfVectorizer(stop_words="english", max_features=500)
        sc = tf.fit_transform(chunks).sum(axis=0).A1
        st.write("**Top TF-IDF keywords:**", [w for w, _ in sorted(zip(tf.get_feature_names_out(), sc), key=lambda x: -x[1])[:10]])
    except Exception as e:
        st.warning(f"TF-IDF unavailable: {e}")
