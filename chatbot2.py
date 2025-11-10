import os
import streamlit as st
import nest_asyncio

# Streamlit에서 비동기 작업을 위한 이벤트 루프 설정
nest_asyncio.apply()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_chroma import Chroma


#Gemini API 키 설정
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception as e:
    st.error("⚠️ GOOGLE_API_KEY를 Streamlit Secrets에 설정해주세요!")
    st.stop()

#cache_resource로 한번 실행한 결과 캐싱해두기
@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

#텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(_docs)
    st.info(f"📄 {len(split_docs)}개의 텍스트 청크로 분할했습니다.")

    persist_directory = "./chroma_db"
    st.info("🤖 임베딩 모델 로드 중... (첫 실행 시 모델 다운로드)")
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

#만약 기존에 저장해둔 ChromaDB가 있는 경우, 이를 로드
@st.cache_resource
def get_vectorstore(_docs):
    persist_directory = "./chroma_db"
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        return create_vector_store(_docs)
    
# PDF 문서 로드-벡터 DB 저장-검색기-히스토리 모두 합친 Chain 구축
@st.cache_resource
def initialize_components(selected_model):
    file_path = "탄소.pdf"
    pages = load_and_split_pdf(file_path)
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    # 채팅 히스토리 요약 시스템 프롬프트
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

    # 질문-답변 시스템 프롬프트
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer perfect. please use imogi with the answer.
    대답은 한국어로 하고, 존댓말을 써줘.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            temperature=0.7,
            convert_system_message_to_human=True
        )
    except Exception as e:
        st.error(f"❌ Gemini 모델 '{selected_model}' 로드 실패: {str(e)}")
        st.info("💡 'gemini-pro' 모델을 사용해보세요.")
        raise
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# Streamlit UI
st.header("탄소 배출 분석 Q&A 챗봇 💬 📚")

# 첫 실행 안내 메시지
if not os.path.exists("./chroma_db"):
    st.info("🔄 첫 실행입니다. 임베딩 모델 다운로드 및 PDF 처리 중... (약 5-7분 소요)")
    st.info("💡 이후 실행에서는 10-15초만 걸립니다!")

# Gemini 모델 선택 - 최신 2.x 모델 사용
option = st.selectbox("Select Gemini Model",
    ("gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash-exp"),
    index=0,
    help="Gemini 2.5 Flash가 가장 빠르고 효율적입니다"
)

try:
    with st.spinner("🔧 챗봇 초기화 중... 잠시만 기다려주세요"):
        rag_chain = initialize_components(option)
    st.success("✅ 챗봇이 준비되었습니다!")
except Exception as e:
    st.error(f"⚠️ 초기화 중 오류 발생: {str(e)}")
    st.info("PDF 파일 경로와 API 키를 확인해주세요.")
    st.stop()

chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", 
                                     "content": "탄소 배출에 대해 무엇이든 물어보세요!!!!!"}]

for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)


if prompt_message := st.chat_input("Your question"):
    st.chat_message("human").write(prompt_message)
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke(
                {"input": prompt_message},
                config)
            
            answer = response['answer']
            st.write(answer)
            with st.expander("참고 문서 확인"):
                for doc in response['context']:
                    st.markdown(doc.metadata['source'], help=doc.page_content)

import google.generativeai as genai
import os

def main():
    """
    Gemini API를 사용하여 파일 업로드(이미지, PDF 등)가 가능한
    대화형 챗봇을 실행합니다.
    """
    # 1. API 키 설정 (이전과 동일)
    try:
        api_key = os.environ["GEMINI_API_KEY"]
    except KeyError:
        print("환경 변수 'GEMINI_API_KEY'가 설정되지 않았습니다.")
        api_key = my_gemini_api_key  # <--- 이 부분을 실제 API 키로 수정하세요

    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"[오류] API 키 설정에 실패했습니다: {e}")
        return

    # 2. 모델 초기화 (이전과 동일)
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        print(f"[오류] 모델 로딩에 실패했습니다: {e}")
        return

    # 3. 대화 세션 시작 (이전과 동일)
    chat = model.start_chat(history=[])

    print("--- 🤖 Gemini 챗봇 (파일 업로드 가능) ---")
    print("대화를 시작합니다. '그만'을 입력하면 종료됩니다.")
    print("파일을 업로드하려면, 질문 전에 파일 경로를 먼저 입력하세요.")
    print("텍스트만 질문하려면, 파일 경로 입력란에서 Enter를 누르세요.")
    print("-" * 20)

    # 4. 대화 루프
    while True:
        try:
            # === [ 변경점 1: 파일 업로드 ] ===
            uploaded_file = None # 매 턴마다 초기화
            file_path = input("📎 업로드할 파일 경로 (없으면 Enter): ").strip()

            if file_path:
                print(f"파일 업로드 중... ({file_path})")
                try:
                    # 파일을 API에 업로드하고 파일 객체를 받습니다.
                    uploaded_file = genai.upload_file(path=file_path)
                    print(f"✅ 파일 업로드 성공!")
                except FileNotFoundError:
                    print(f"[오류] 파일을 찾을 수 없습니다: {file_path}")
                    continue # 다음 루프로 이동
                except Exception as e:
                    print(f"[오류] 파일 업로드에 실패했습니다: {e}")
                    print("지원되는 파일 형식(JPG, PNG, PDF 등)인지 확인하세요.")
                    continue # 다음 루프로 이동

            # 4-1. 사용자 텍스트 입력 받기
            if uploaded_file:
                user_input = input("You (파일에 대해 질문): ")
            else:
                user_input = input("You (텍스트로 질문): ")

            # 4-2. 종료 조건 확인
            if user_input.lower() == '그만':
                print("Gemini: 🤖 대화를 종료합니다. 이용해주셔서 감사합니다.")
                break

            if not user_input.strip(): # 빈 입력은 무시
                continue

            # === [ 변경점 2: 파일과 텍스트를 함께 전송 ] ===

            # 보낼 콘텐츠를 리스트로 구성합니다.
            content_to_send = []

            # 텍스트 프롬프트를 리스트에 추가합니다.
            content_to_send.append(user_input)

            # (중요) 이번 턴에 업로드된 파일이 있다면, 리스트에 추가합니다.
            if uploaded_file:
                content_to_send.append(uploaded_file)

            # 4-3. (수정) 채팅 세션에 [텍스트] 또는 [텍스트, 파일] 리스트 전송
            response_stream = chat.send_message(content_to_send, stream=True)
            print("Gemini: 🤖 ", end="")

            # 4-4. 스트리밍 응답 출력 (이전과 동일)
            for chunk in response_stream:
                print(chunk.text, end="", flush=True)

            print() # 응답 완료 후 줄바꿈

        except Exception as e:
            print(f"\n\n[오류 발생]: {e}")
            print("API 요청 중 문제가 발생했습니다. 입력을 다시 시도해주세요.")

if __name__ == "__main__":
    main()
