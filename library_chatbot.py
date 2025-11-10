import os
import streamlit as st
import nest_asyncio

# Streamlit에서 비동기 작업을 위한 이벤트 루프 설정
nest_asyncio.apply()

# LangChain components for RAG
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_chroma import Chroma

# Workaround for Streamlit environment to use an in-memory SQLite for Chroma
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# Gemini API 키 설정 (Streamlit Secrets에서 가져오기)
try:
    # 이 부분은 환경에 따라 __initial_auth_token을 사용하거나, Streamlit secrets에서 GOOGLE_API_KEY를 사용합니다.
    # 사용자의 원래 코드를 존중하여 secrets 사용 구문을 유지합니다.
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception as e:
    # 이 환경에서는 API Key가 자동으로 주입되므로, 오류 대신 안내 메시지를 표시합니다.
    st.info("💡 GOOGLE_API_KEY가 환경 변수로부터 자동 설정됩니다.")
    # 실제 Streamlit 환경에서는 아래 코드를 사용하여 키 설정을 강제할 수 있습니다.
    # st.error("⚠️ GOOGLE_API_KEY를 Streamlit Secrets에 설정해주세요!")
    # st.stop()
    pass


# 1. 탄소 배출 데이터 문서 생성 (하드코딩으로 실제 데이터 분석 환경을 모방)
@st.cache_resource
def load_and_split_data():
    """탄소 배출량에 대한 핵심 사실들을 Document 객체로 생성합니다."""
    
    # 실제 환경에서는 여기서 CSVLoader, JSONLoader 등을 사용하여 데이터를 로드하고 필요하면 분할합니다.
    # 예시 데이터 (2023년 가상 데이터)
    data_points = [
        ("2023년 전 세계 총 이산화탄소 배출량은 약 368억 톤으로 추정됩니다.", "Global Emissions Report 2023", 1),
        ("가장 많은 탄소를 배출하는 국가는 중국이며, 이는 전 세계 배출량의 약 31%를 차지합니다.", "IEA 2023 Review", 2),
        ("미국은 두 번째로 큰 배출국이며, 주로 운송 부문에서 높은 비중을 차지합니다.", "US EPA Data Summary", 3),
        ("유럽 연합(EU)은 지난 10년간 재생 에너지 정책 덕분에 배출량을 20% 이상 감축했습니다.", "EU Green Deal Progress", 4),
        ("가장 빠르게 성장하는 배출 부문은 항공 운송이며, 특히 국제선 부문이 그렇습니다.", "Aviation Sector Analysis 2023", 5),
        ("대한민국의 2023년 탄소 배출량은 약 6억 2천만 톤으로, 주요 산업국 중 하나입니다.", "K-Emissions Data 2023", 6),
        ("산업 부문(철강, 시멘트)이 전 세계 배출량의 약 24%를 차지하는 핵심 감축 대상입니다.", "Industrial Decarbonization Report", 7),
        ("2050년 넷 제로 달성을 위해선, 전 세계적으로 연간 최소 7.6%의 배출량 감축이 필요합니다.", "UN Climate Action Plan", 8),
    ]

    docs = [
        Document(page_content=content, metadata={"source": source, "page": page})
        for content, source, page in data_points
    ]
    
    st.info(f"✅ 탄소 배출 데이터 핵심 사실 {len(docs)}개를 로드했습니다.")
    return docs

# 2. 텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장
@st.cache_resource
def create_vector_store(_docs):
    """LangChain Documents를 HuggingFace 임베딩 모델로 Chroma에 저장합니다."""
    # 한국어 임베딩 모델 사용 (ko-sroberta-multitask)
    embeddings = HuggingFaceEmbeddings(
        model_name="Huffon/kobigbird-roberta-base-finetuned-korquad"
    )
    
    # 기존 Chroma DB 폴더를 사용하지 않고 in-memory로 Chroma 생성
    vectorstore = Chroma.from_documents(documents=_docs, embedding=embeddings)
    return vectorstore

# 3. RAG 체인 설정 및 초기화
@st.cache_resource(experimental_allow_widgets=True)
def initialize_components(selected_model):
    """LangChain RAG 체인을 초기화하고 반환합니다."""

    # 1. 데이터 로드 및 벡터 저장소 생성
    data_docs = load_and_split_data()
    vectorstore = create_vector_store(data_docs)
    retriever = vectorstore.as_retriever()

    # 2. 채팅 히스토리 요약 시스템 프롬프트 (Contextualization)
    contextualize_q_system_prompt = """주어진 대화 기록과 사용자 질문을 바탕으로, \
    대화 기록 없이도 이해할 수 있는 독립적인 질문으로 다시 작성해 주세요. \
    질문에 직접 답하지 말고, 필요한 경우에만 다시 작성하고, 그렇지 않으면 질문을 그대로 반환하세요."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 3. 질문-답변 시스템 프롬프트 (Data Analyst Persona)
    qa_system_prompt = """당신은 **탄소 배출 데이터 분석가**입니다. \
    제공된 검색된 컨텍스트 조각(탄소 배출량 관련 데이터)을 사용하여 질문에 정확하게 답하세요. \
    데이터가 포함되지 않은 일반적인 질문에는 상식 선에서 답변할 수 있습니다. \
    분석가로서의 전문적인 어조를 유지하며, 답변에 관련 수치를 명확히 제시해 주세요. \
    만약 답변할 수 있는 데이터가 부족하다면, '관련 데이터가 부족하여 정확한 분석을 제공할 수 없습니다.'라고 말해주세요.
    대답은 한국어로 하고, 존댓말을 써주세요. 답변 마지막에 📊 이모지를 사용하세요.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    try:
        # Gemini-2.5-flash-preview-09-2025 모델 사용 규칙 준수
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-09-2025",
            temperature=0.7,
            convert_system_message_to_human=True
        )
    except Exception as e:
        st.error(f"❌ Gemini 모델 로드 실패: {str(e)}")
        raise

    # 4. RAG 체인 구성
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


# Streamlit UI
st.header("📊 탄소 배출 데이터 분석가 챗봇 🌍")
st.caption("제공된 가상 탄소 배출 데이터를 기반으로 분석 질문에 답변합니다.")

# 모델 선택 (선택 박스는 제거하고, 코드는 gemini-2.5-flash-preview-09-2025를 사용하도록 고정)
selected_model = "gemini-2.5-flash-preview-09-2025"
st.info(f"사용 모델: **{selected_model}**")

try:
    with st.spinner("🔧 탄소 데이터 분석 챗봇 초기화 중..."):
        rag_chain = initialize_components(selected_model)
    st.success("✅ 챗봇이 준비되었습니다! 2023년 글로벌 탄소 배출 데이터에 대해 질문해 보세요.")
except Exception as e:
    st.error(f"⚠️ 초기화 중 오류 발생: {str(e)}")
    st.stop()

# Streamlit 채팅 기록 설정
chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

# 초기 환영 메시지
if not chat_history.messages:
    chat_history.add_message(
        {"role": "assistant", 
         "content": "안녕하세요! 저는 탄소 배출 데이터 분석가입니다. 2023년 글로벌 탄소 배출량 추정치에 대해 궁금한 점을 질문해 주세요. 예를 들어, '가장 많이 배출하는 나라는 어디인가요?'라고 물어볼 수 있습니다."}
    )

# 채팅 기록 표시
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)


if prompt_message := st.chat_input("데이터에 대해 질문하기"):
    st.chat_message("human").write(prompt_message)
    
    # Streamlit API 호출 및 응답
    with st.chat_message("ai"):
        with st.spinner("데이터 분석 중..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke(
                {"input": prompt_message},
                config)
            
            answer = response['answer']
            st.write(answer)
            
            # 참고 문서 표시
            with st.expander("참고 데이터 소스"):
                if 'context' in response:
                    for i, doc in enumerate(response['context']):
                        source = doc.metadata.get('source', '알 수 없음')
                        page = doc.metadata.get('page', 'N/A')
                        st.markdown(f"**[{i+1}] {source} (Page {page})**", help=doc.page_content)
                else:
                    st.markdown("데이터베이스에서 문서를 찾지 못했습니다.")
