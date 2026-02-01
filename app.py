import streamlit as st

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate


# =====================================================
# CONFIGURAÇÃO STREAMLIT
# =====================================================
st.set_page_config(
    page_title="Oráculo ECD",
    page_icon="📘",
    layout="wide"
)

st.title("📘 Guru SPED Contábil")
st.caption("Baseado exclusivamente no Manual Oficial da ECD")


# =====================================================
# EMBEDDINGS + VECTOR STORE (DEVE SER IGUAL AO INDEXADOR)
# =====================================================
@st.cache_resource
def load_vectorstore():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"  # 🔴 TEM que ser o mesmo do indexador
    )

    vectorstore = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

    return vectorstore


vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# =====================================================
# MODELO LLM (LEVE PARA CODESPACES)
# =====================================================
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# =====================================================
# PROMPT DO ORÁCULO
# =====================================================
PROMPT_TEMPLATE = """
Você é um especialista sênior em Escrituração Contábil Digital (ECD),
com profundo conhecimento do Manual da ECD e das regras do SPED Contábil.

REGRAS OBRIGATÓRIAS:
- Responda SOMENTE com base no contexto fornecido.
- Cite explicitamente Bloco, Registro e seção quando aplicável.
- Se a informação não existir no contexto, diga claramente:
  "Essa informação não consta no Manual da ECD fornecido."
- Seja técnico, objetivo e preciso.
- Não invente regras.
- Quando possível, traduza a regra para lógica computacional.

CONTEXTO:
{context}

PERGUNTA:
{question}

RESPOSTA:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE
)


# =====================================================
# INTERFACE
# =====================================================
question = st.text_input(
    "Digite sua dúvida sobre a ECD:",
    placeholder="Ex: Para que serve o Bloco 0?"
)

if question:
    with st.spinner("🔎 Consultando o Manual da ECD..."):

        # ---------------------------
        # BUSCA SEMÂNTICA
        # ---------------------------
        docs = retriever.invoke(question)

        if not docs:
            st.warning("Nenhum trecho relevante encontrado no manual.")
        else:
            context = "\n\n".join(doc.page_content for doc in docs)

            final_prompt = prompt.format(
                context=context,
                question=question
            )

            # ---------------------------
            # GERAÇÃO DA RESPOSTA
            # ---------------------------
            try:
                response = llm.invoke(final_prompt)

                st.subheader("📖 Resposta do Oráculo")
                st.write(response.content)

            except Exception as e:
                st.error("❌ Erro ao gerar resposta com o modelo.")
                st.exception(e)
