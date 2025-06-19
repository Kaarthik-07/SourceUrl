import os
import pickle
import streamlit as st
import psutil  # To check system memory
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# App UI Setup
st.set_page_config(page_title="SourceURL QA", layout="wide")
st.title("🔗 SourceURL QA with Debug")
st.sidebar.title("📄 Article URL Input")

# ⚙️ Debug Mode
debug = st.sidebar.checkbox("🛠️ Debug Mode", value=False)

# 🔍 Memory Check
mem = psutil.virtual_memory()
if mem.available < 2 * 1024 * 1024 * 1024:  # < 2 GB
    st.sidebar.warning(f"⚠️ Low free memory: {mem.available / (1024**2):.0f} MB. Consider closing apps.")

# 1. URL Inputs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i+1}")
    if url:
        urls.append(url)

process_url = st.sidebar.button("Process URL")
file_path = "hugging_embeddings.pkl"
main_placeholder = st.empty()

# 2. Process URLs
if process_url:
    if not urls:
        st.warning("⚠️ Please enter at least one URL.")
    else:
        main_placeholder.text("🔄 Loading URLs ...")
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        main_placeholder.text("✂️ Splitting content into chunks ...")
        splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=800,
            chunk_overlap=100
        )
        docs = splitter.split_documents(data)

        main_placeholder.text("🧠 Creating embeddings ...")
        embed_model = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
        vect_store = FAISS.from_documents(docs, embed_model)

        with open(file_path, "wb") as f:
            pickle.dump(vect_store, f)

        main_placeholder.success("✅ URLs processed and embeddings saved!")

# 3. Query Input
query = st.text_input("💬 Ask a question based on the URLs above")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vector_store = pickle.load(f)

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        llm = OllamaLLM(model="llama3.2:1b")

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

        try:
            res = chain.invoke({"question": query})

            # ✅ Show everything if debug is on
            if debug:
                st.markdown("### 🔍 Raw Response")
                st.json(res)

            # Show answer
            st.header("📌 Answer")
            answer = res.get("answer", "").strip()
            if answer:
                st.write(answer)
            else:
                st.warning("⚠️ Model returned an empty answer.")

            # Show sources
            st.markdown("#### 📚 Sources:")
            sources = res.get("sources", "")
            if sources:
                for source in sources.split(", "):
                    st.markdown(f"- {source}")
            else:
                st.markdown("No sources returned.")

        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.warning("⚠️ Please process the URLs first.")

# 🔁 Raw LLM Test (Optional)
if st.sidebar.button("🔁 Test LLM (Raw Mode)"):
    st.markdown("### 🧪 Raw LLM Output:")
    llm = OllamaLLM(model="llama3.2:1b")
    try:
        output = llm.invoke("What is ReactJS and who created it?")
        st.write(output if output else "⚠️ No output received.")
    except Exception as e:
        st.error(f"❌ Error: {e}")
