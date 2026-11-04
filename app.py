import os, re, nltk, streamlit as st
from collections import Counter
from nltk import word_tokenize, pos_tag, ne_chunk, bigrams
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="CHAT WITH PDF", layout="wide")
st.title("CHAT WITH PDF🗣️")

# ---------- Sidebar: API key, PDF, top-k ----------
key = st.sidebar.text_input("GOOGLE_API_KEY", type="password")
pdf = st.sidebar.file_uploader("Upload your PDF", type=["pdf"])
k = st.sidebar.slider("Top-K chunks", 1, 10, 4)
if not key or pdf is None:
    st.info("Enter your Google API key and upload a PDF in the sidebar.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = key

# ---------- Unit I: Text preprocessing tools ----------
@st.cache_resource
def setup_nltk():
    for p in ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger_eng",
              "maxent_ne_chunker_tab", "words"]:
        nltk.download(p, quiet=True)
setup_nltk()
stemmer, lemmatizer, STOP = PorterStemmer(), WordNetLemmatizer(), set(stopwords.words("english"))

def preprocess(text):
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())            # regex + normalization
    toks = word_tokenize(text)                                   # tokenization
    toks = [t for t in toks if t not in STOP]                    # stop-word removal
    return toks, [stemmer.stem(t) for t in toks], [lemmatizer.lemmatize(t) for t in toks]

# ---------- Load PDF -> chunks -> embeddings (Unit IV) -> FAISS ----------
os.makedirs("pdf_files", exist_ok=True)
path = os.path.join("pdf_files", pdf.name)
open(path, "wb").write(pdf.getbuffer())

@st.cache_data(show_spinner="Reading PDF...")
def load_chunks(p):
    docs = PyPDFLoader(p).load()
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

@st.cache_resource(show_spinner="Building index...")
def build_index(p):
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")   # word/sentence embeddings
    return FAISS.from_documents(load_chunks(p), emb)

chunks, vs = load_chunks(path), build_index(path)
full_text = " ".join(c.page_content for c in chunks)

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["💬 Chat with PDF", "🔬 NLP Analysis"])

with tab2:
    st.caption("NLP techniques from the syllabus applied to your report")
    sample = full_text[:3000]
    toks, stems, lems = preprocess(sample)
    st.write("**Tokens (after normalization + stop-word removal):**", toks[:25])
    st.write("**Stemmed (Porter):**", stems[:25])
    st.write("**Lemmatized:**", lems[:25])                                   # Unit I
    st.write("**POS tags:**", pos_tag(word_tokenize(sample))[:20])           # Unit II
    ents = [(" ".join(w for w, _ in t), t.label())
            for t in ne_chunk(pos_tag(word_tokenize(sample))) if hasattr(t, "label")]
    st.write("**Named Entities (NER):**", ents[:15])                         # Unit II
    st.write("**Top bigrams (n-grams):**",
             [" ".join(b) for b, _ in Counter(bigrams(toks)).most_common(8)])  # Unit II
    tf = TfidfVectorizer(stop_words="english", max_features=500)             # Unit III
    scores = tf.fit_transform([c.page_content for c in chunks]).sum(axis=0).A1
    top = sorted(zip(tf.get_feature_names_out(), scores), key=lambda x: -x[1])[:10]
    st.write("**Top TF-IDF keywords:**", [w for w, _ in top])

# ---------- Unit IV: Transformer LLM for question answering / summarization ----------
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
prompt = ChatPromptTemplate.from_template("""
You are a medical report assistant. Using the report context below, answer the question in simple language.
Quote the patient's actual values from the report. If a medical term (e.g. bilirubin) is asked about,
explain what it is, its normal range, and what high/low values may indicate.
If the report doesn't contain the needed info, say "I don't know based on the document."
End with: "This is not medical advice; consult a doctor."

Context:
{context}

Question: {question}
""")
fmt = lambda docs: "\n\n".join(d.page_content for d in docs)
chain = ({"context": vs.as_retriever(search_kwargs={"k": k}) | fmt, "question": RunnablePassthrough()}
         | prompt | llm | StrOutputParser())

with tab1:
    q = st.text_area("Ask about your report (e.g. 'What does my bilirubin level mean?')")
    c1, c2 = st.columns(2)
    if c1.button("Get Answer") and q:
        st.caption(f"Query keywords: {preprocess(q)[2]}")
        st.write_stream(chain.stream(q))
    if c2.button("Summarize Report"):
        st.write_stream(chain.stream("Summarize this report: key findings and abnormal values."))
