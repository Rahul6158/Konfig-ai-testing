import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq

# ===============================
# CONFIG
# ===============================
FAISS_DB_PATH = "/content/faiss_index"  # folder containing index.faiss & index.pkl
GROQ_API_KEY = "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"
GROQ_MODEL = "moonshotai/kimi-k2-instruct-0905"

# ===============================
# LOAD FAISS DB
# ===============================
@st.cache_resource
def load_faiss():
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={"device": "cpu", "trust_remote_code": True}
    )
    return FAISS.load_local(
        FAISS_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

faiss_db = load_faiss()

# ===============================
# INIT GROQ CLIENT
# ===============================
client = Groq(api_key=GROQ_API_KEY)

# ===============================
# JSON GENERATION FUNCTION
# ===============================
def generate_json(user_input: str):
    # Step 1: Retrieve top relevant widgets from FAISS
    docs = faiss_db.similarity_search(user_input, k=5)
    context_text = "\n".join([doc.page_content for doc in docs])

    # Step 2: Construct instruction-focused prompt
    prompt = f"""
You are an expert UI builder assistant.

Task:
1. Using the following TypeScript widget schemas, generate a JSON page structure for the user's request.
2. Only include widgets that are relevant to the request.
3. If a required widget is not present in the retrieved schemas, invent it following the style and structure of the provided widgets.
4. Preserve layout, hierarchy, columns, modals, and other structure hints from the retrieved widgets.
5. Output valid JSON only. Do not include explanations or extra text.

Available widgets:
{context_text}

User request:
{user_input}
"""

    # Step 3: Call Groq API
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_completion_tokens=4096,
        top_p=1,
        stream=False
    )

    return docs, completion.choices[0].message.content

# ===============================
# STREAMLIT APP UI
# ===============================
st.set_page_config(page_title="FAISS + Groq JSON Generator", layout="wide")
st.title("🔧 JSON Page Builder Assistant")

# Input box
user_input = st.text_area("Enter your page description:", placeholder="e.g., A login page with email and password fields...")

if st.button("Generate JSON") and user_input.strip():
    with st.spinner("Generating JSON..."):
        try:
            docs, json_output = generate_json(user_input)

            # Two-column layout
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📑 Retrieved Widget Schemas")
                for i, doc in enumerate(docs, start=1):
                    st.markdown(f"**Widget {i}:**")
                    st.code(doc.page_content, language="typescript")

            with col2:
                st.subheader("🛠️ Generated JSON Structure")
                st.code(json_output, language="json")

        except Exception as e:
            st.error(f"Error: {e}")
