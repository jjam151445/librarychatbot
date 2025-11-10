import os
import streamlit as st
import nest_asyncio

# Streamlit에서 비동기 작업을 위한 이벤트 루프 설정
nest_asyncio.apply()

# LangChain components for RAG
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ⚠️ 수정: 'langchin_google_genai'를 'langchain_google_genai'로 수정
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_core.messages import AIMessage # AIMessage import 유지

# SQLite/ChromaDB 우회 코드 (ChromaDB 사용 시 필수)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_chroma import Chroma


# Gemini API 키 설정
try:
    # 이 환경에서는 API Key가 자동 주입되므로, 오류 대신 안내 메시지를 표시합니다.
    # 사용자의 원래 코드를 존중하여 secrets 사용 구문을 유지합니다.
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception as e:
    st.info("💡 GOOGLE_API_KEY가 환경 변수로부터 자동 설정됩니다.")
    # st.error("⚠️ GOOGLE_API_KEY를 Streamlit Secrets에 설정해주세요!")
    # st.stop()


# cache_resource로 한번 실행한 결과 캐싱해두기
@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    # PDF를 페이지 단위로 로드하고 분할
    docs = loader.load_and_split()
    st.info(f"📄 PDF 문서에서 총 {len(docs)} 페이지를 로드했습니다.")
    return docs

# 텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(_docs)
    st.info(f"📄 {len(split_docs)}개의 텍스트 청크로 분할했습니다.")

    persist_directory = "./chroma_db"
    st.info("🤖 임베딩 모델 로드 중... (jhgan/ko-sroberta-multitask)")
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    st.info("🔢 벡터 임베딩 생성 및 저장 중...")
    vectorstore = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persist_directory
    )
    st.success("💾 벡터 데이터베이스 생성 완료!")
    return vectorstore

# 만약 기존에 저장해둔 ChromaDB가 있는 경우, 이를 로드
@st.cache_resource
def get_vectorstore(_docs):
    persist_directory = "./chroma_db"
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    if os.path.exists(persist_directory):
        st.info("🔄 기존 벡터 데이터베이스 로드 중...")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        return create_vector_store(_docs)
    
# PDF 문서 로드-벡터 DB 저장-검색기-히스토리 모두 합친 Chain 구축
@st.cache_resource
def initialize_components(selected_model):
    # NOTE: 이 파일은 Streamlit 환경에서 실행되며, '탄소 분석.pdf' 파일이 미리 업로드되어 있다고 가정합니다.
    file_path = "탄소 분석.pdf" 
    
    if not os.path.exists(file_path):
        st.error(f"⚠️ 파일 경로 오류: '{file_path}' 파일을 찾을 수 없습니다. 파일을 업로드하거나 경로를 확인해주세요.")
        st.stop()
        
    pages = load_and_split_pdf(file_path)
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    # 3. 채팅 히스토리 요약 시스템 프롬프트 (기존 영어 프롬프트 유지)
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 4. 질문-답변 시스템 프롬프트 (✨ 탄소 배출 전문가 페르소나 강화)
    qa_system_prompt = """당신은 **탄소 배출 및 환경 분석 전문가**입니다. \
    제공된 검색된 컨텍스트 조각(PDF 문서 내용)을 사용하여 질문에 깊이 있고 정확하게 답하세요. \
    전문가로서의 신뢰감을 주는 어조를 사용하고, 답변 시 관련된 사실과 수치, 혹은 문서의 핵심 내용을 명확히 제시해 주세요. \
    만약 답변할 수 있는 정보가 부족하다면, '제공된 문서 내에서는 해당 정보를 찾을 수 없습니다.'라고 정중하게 말해주세요. \
    모든 대답은 한국어로 하고, 존댓말을 사용해 주세요. 답변 마지막에는 항상 🌿 이모지를 넣어주세요.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    try:
        # 모델명은 option 변수를 통해 전달받습니다.
        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            temperature=0.3, # 전문적인 답변을 위해 온도를 약간 낮춥니다.
            convert_system_message_to_human=True
        )
    except Exception as e:
        # 모델 로드 실패 시 명확한 오류 메시지를 출력합니다.
        st.error(f"❌ Gemini 모델 '{selected_model}' 로드 실패: {str(e)}")
        st.info("💡 모델명이 올바른지 확인해주세요. (예: gemini-2.5-flash)")
        raise

    # 5. RAG 체인 구성
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# Streamlit UI
st.header("🌿 PDF 기반 탄소 배출 분석 전문가 챗봇")
st.caption("업로드된 '탄소 분석.pdf' 문서를 기반으로 정확한 답변을 제공합니다.")

# Gemini 모델 선택
option = st.selectbox("Select Gemini Model",
    ("gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash-exp"),
    index=0,
    help="Gemini 2.5 Flash가 가장 빠르고 효율적입니다"
)

try:
    # 파일을 찾을 수 없는 경우를 대비하여 initialize_components에서 처리
    with st.spinner("🔧 챗봇 초기화 및 PDF 분석 중... (첫 실행 시 시간이 오래 걸립니다)"):
        rag_chain = initialize_components(option)
    st.success("✅ 챗봇 초기화 완료! 이제 질문할 수 있습니다.")
except Exception as e:
    # initialize_components 내에서 st.stop()이 호출되지 않도록 수정하여 오류를 명확히 보여줍니다.
    st.error(f"⚠️ 챗봇 초기화 실패. 오류: {str(e)}")
    # st.info("PDF 파일 경로와 API 키를 확인해주세요.") # initialize_components에서 이미 처리됨
    st.stop()


chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

# 초기 환영 메시지 (Session State 대신 StreamlitChatMessageHistory 사용)
if not chat_history.messages:
    chat_history.add_message(
        AIMessage(content="안녕하세요! 저는 탄소 배출 및 환경 분석 전문가입니다. 어떤 질문이든 깊이 있게 답변해 드리겠습니다. 🌿")
    )

# 채팅 기록 표시 (⚠️ 수정: msg.type 대신 안전한 역할 변환 로직 사용)
for msg in chat_history.messages:
    # LangChain BaseMessage 객체를 Streamlit Role로 변환
    if hasattr(msg, 'type'):
        role = "user" if msg.type == "human" else "assistant"
        st.chat_message(role).write(msg.content)
    else:
        # 안전 장치: 구조가 예상과 다른 경우 메시지를 그대로 출력합니다.
        st.chat_message("assistant").write(str(msg.content))


if prompt_message := st.chat_input("전문가에게 질문하기"):
    st.chat_message("user").write(prompt_message) # human 대신 user 역할 사용
    
    with st.chat_message("assistant"): # ai 대신 assistant 역할 사용
        with st.spinner("문서 분석 및 답변 생성 중..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke(
                {"input": prompt_message},
                config)
            
            answer = response['answer']
            st.write(answer)
            
            with st.expander("참고 문서 확인"):
                if 'context' in response:
                    for i, doc in enumerate(response['context']):
                        source = doc.metadata.get('source', '알 수 없음')
                        page = doc.metadata.get('page', 'N/A')
                        st.markdown(f"**[{i+1}] 출처: {source}** (페이지: {page})", help=doc.page_content)
                else:
                    st.markdown("답변에 사용된 문서 정보를 찾을 수 없습니다.")
