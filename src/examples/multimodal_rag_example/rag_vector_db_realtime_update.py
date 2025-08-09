# 필요한 모듈 불러오기
import openai  # [포인트] OpenAI의 GPT 모델을 사용하기 위한 공식 라이브러리입니다.
import os      # [포인트] 환경 변수 설정을 위해 사용하는 파이썬 내장 모듈입니다. API 키를 숨길 수 있어 보안상 유리합니다.
# API 키를 환경 변수에 저장하여 보안 강화
# [예시] 자물쇠로 잠긴 금고에 API 키를 보관하는 느낌입니다. 키를 코드에 직접 쓰지 않아 보안이 향상됩니다.
os.environ['OPENAI_API_KEY'] = ""  # [포인트] 실제 API 키를 환경 변수 OPENAI_API_KEY에 저장합니다.
openai.api_key = os.getenv('OPENAI_API_KEY')  # [포인트] 환경 변수에서 API 키를 불러와 openai 모듈에 등록합니다.

# ✅ 모듈 불러오기
import openai  # OpenAI API 사용을 위한 모듈
import os  # 시스템 환경 설정용
import pandas as pd  # 엑셀 등 표 형식 데이터 처리
from langchain.docstore.document import Document  # 텍스트 문서를 담는 LangChain 문서 객체
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # OpenAI의 임베딩 및 챗 모델 로드
from langchain.vectorstores import Chroma  # 벡터 DB로 Chroma 사용
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate  # 대화형 프롬프트 구성 도구
from langchain.schema.runnable import RunnablePassthrough  # 입력 그대로 전달하는 중간 노드
from langchain.schema.output_parser import StrOutputParser  # 모델 응답을 문자열로 파싱

# ✅ 엑셀 파일 로드
df = pd.read_excel('/content/Sample.xlsx')  # 엑셀 파일을 판다스로 불러옴

# ✅ 문서 생성 (요약을 가장 먼저 배치)
docs = []
for i, row in df.iterrows():  # [포인트] 각 행(row)을 하나의 문서로 처리
    summary = f"요약: {row['설명 및 소개'] if pd.notnull(row['설명 및 소개']) else ''}"  # [포인트] 설명 컬럼을 요약 형태로 가장 앞에 배치
    detail = "\n".join([
        f"{col}: {row[col]}" for col in df.columns
        if col != '설명 및 소개' and pd.notnull(row[col])
    ])  # [포인트] 나머지 컬럼을 줄 단위로 정리
    full_text = f"{summary}\n{detail}"  # 요약 + 상세 내용을 하나의 문자열로 합침
    docs.append(Document(page_content=full_text, metadata={"row": i}))  # [포인트] row 번호를 메타데이터로 저장

# ✅ 확인용 출력
for i, doc in enumerate(docs[:3]):  # 앞의 3개 문서만 출력하여 확인
    print(f"📄 문서 {i+1}")
    print(doc.page_content)
    print("-" * 80)

# ✅ 벡터 DB 구축
embeddings = OpenAIEmbeddings()  # [포인트] 문서를 숫자 벡터로 변환하는 OpenAI 임베딩 사용
db = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./chroma_db")  # [포인트] 변환된 문서를 Chroma DB에 저장
retriever = db.as_retriever(search_kwargs={"k": 5})  # [포인트] 관련 문서 5개를 검색할 수 있도록 설정


# ✅ 벡터 DB 업데이트 함수 (동일한 요약 강조 포함)
def update_vectordb_from_excel(file_path, db):  # [포인트] 새 엑셀 데이터를 벡터 DB에 추가하는 함수
    new_df = pd.read_excel(file_path)  # 새 엑셀 파일 로드
    new_docs = []
    for i, row in new_df.iterrows():  # 새 행마다 문서 구성
        summary = f"요약: {row['설명 및 소개'] if pd.notnull(row['설명 및 소개']) else ''}"  # 요약 먼저
        detail = "\n".join([
            f"{col}: {row[col]}" for col in new_df.columns
            if col != '설명 및 소개' and pd.notnull(row[col])
        ])
        full_text = f"{summary}\n{detail}"
        new_docs.append(Document(page_content=full_text, metadata={"row": i}))
    db.add_documents(new_docs)  # 벡터 DB에 새 문서 추가
    db.persist()  # DB 저장
    print(f"✅ {len(new_docs)}개의 문서가 벡터 DB에 추가되었습니다.")  # 결과 출력

# ✅ 프롬프트 설정
system_prompt = """당신은 사용자 맞춤형 상품 정보 서비스플랫폼의 AI 비서입니다.
다음은 사용자의 질문에 답하기 위해 참고할 수 있는 데이터 문서입니다.

당신의 목표는:
- 사용자의 자연어 질의에 대해,
- 아래 문서들을 바탕으로 가장 관련도 높은 정보를 찾고,
- 명확하고 실용적인 한국어로 답변을 제공하며,
- 필요 시 제품명, 특징, 가격등을 제시하고,
- 해당 정보가 어느 행(row)에 있는지도 함께 출력하는 것입니다.

질문에 대해 아래의 순서로 답변하세요:
1. 문서 내용이 질문과 얼마나 관련이 있는지 짧게 평가하세요.
2. 질문에 대한 정확한 답변을 제공하세요.
3. 출처 정보를 `row id`로 표시하세요. (예: row: 17)

---
{summaries}
---
Answer:"""  # [포인트] AI 비서의 역할과 응답 방식 정의, row 번호까지 출처로 출력

messages = [
    SystemMessagePromptTemplate.from_template(system_prompt),  # 시스템 메시지 삽입 (AI 역할 안내)
    HumanMessagePromptTemplate.from_template("질문: {question}")  # 사용자의 질문 포맷 정의
]
prompt = ChatPromptTemplate.from_messages(messages)  # 전체 프롬프트 구조 결합

# ✅ LLM 체인 구성
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1, max_tokens=4096)  #
chain = (
    {"summaries": retriever, "question": RunnablePassthrough()}  # [포인트] 질문과 관련 문서를 함께 넘김
    | prompt  # 프롬프트 형식에 맞게 입력 구성
    | llm  # GPT에 전달
    | StrOutputParser()  # [포인트] GPT 응답을 텍스트로 파싱
)

# ✅ 테스트 쿼리 실행
query = "펫피더 스마트에 대해 설명해줘"
result = chain.invoke(query)
print(result)

update_vectordb_from_excel("/content/Sample_updated.xlsx", db)

# ✅ 테스트 쿼리 실행
query = "클린플랜트에 대해 설명해줘"
result = chain.invoke(query)
print(result)
