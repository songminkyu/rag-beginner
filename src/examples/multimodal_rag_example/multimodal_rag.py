# ✅ 필수 라이브러리 불러오기
import os, io, base64, logging, re  # 🔹포인트: 파일 처리, 문자열, 로그 등 기본 도구
from pathlib import Path  # 🔹포인트: 경로를 문자열이 아닌 객체처럼 다룰 수 있음
from PIL import Image  # 🔹포인트: 이미지 열기/저장용 라이브러리
from tqdm import tqdm  # 🔹포인트: 작업 진행 상황을 시각적으로 보여주는 프로그레스 바

# 🔹포인트: LangChain의 핵심 컴포넌트 (텍스트 분할, 문서 처리 등)
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 긴 텍스트를 자연스럽게 쪼갬
from langchain_core.documents import Document  # 텍스트 덩어리를 문서 단위로 관리
from langchain_core.output_parsers import StrOutputParser  # 모델 출력값을 문자열로 파싱
from langchain_core.runnables import RunnablePassthrough  # 값 그대로 전달하는 중간 노드
from langchain.prompts import PromptTemplate  # 프롬프트 양식 지정

# 🔹포인트: 로컬 벡터 DB와 OpenAI 연동
from langchain_community.vectorstores import Chroma  # 로컬에서 벡터 저장소 관리
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # GPT-4o 호출, 텍스트 임베딩 처리

# 🔹포인트: Docling으로 PDF에서 이미지/표 구조 추출
from docling_core.types.doc import PictureItem, TableItem  # 그림, 표 형태 정의
from docling.datamodel.base_models import InputFormat  # 입력 파일 형식 정의용
from docling.datamodel.pipeline_options import PdfPipelineOptions  # PDF 처리 옵션 정의
from docling.document_converter import DocumentConverter, PdfFormatOption  # PDF → 내부 구조로 변환

from IPython.display import display as ipy_display  # 노트북에서 이미지 시각 출력

# ✅ 환경 설정
os.environ['OPENAI_API_KEY'] = ""  # 🔹포인트: OpenAI API 키 등록 (유출 금지!)
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)  # GPT-4o 모델 지정 (창의성 낮게)
embeddings = OpenAIEmbeddings()  # 텍스트를 숫자 벡터로 변환하는 객체

# ✅ 이미지 → base64 인코딩 함수
def encode_image_to_base64(image_path):
    image = Image.open(image_path).convert("RGB")  # 이미지 열고 RGB로 변환 (색상 오류 방지)
    buffered = io.BytesIO()  # 메모리 버퍼 객체 생성
    image.save(buffered, format="PNG")  # 버퍼에 PNG 형식으로 저장
    return base64.b64encode(buffered.getvalue()).decode("utf-8")  # base64 문자열로 변환

# ✅ 이미지 설명 생성 함수
def generate_caption(image_path):
    img_b64 = encode_image_to_base64(image_path)  # 🔹포인트: 이미지 → base64로 변환
    prompt = "이 이미지를 한국어로 간결하게 설명해줘."  # GPT에게 줄 질문
    response = llm.invoke([  # 🔹포인트: 이미지 + 텍스트로 GPT에 질의
        {"role": "system", "content": "당신은 이미지를 분석해 설명하는 AI입니다."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]
        }
    ])
    return response.content  # GPT-4o가 생성한 이미지 설명 반환

# ✅ PDF 파싱 및 이미지 추출
def parse_pdf_and_extract_images(pdf_path, output_dir="docling_output"):
    logging.basicConfig(level=logging.INFO)  # 로그 레벨 설정 (디버깅용)
    output_dir = Path(output_dir)
    IMAGE_RESOLUTION_SCALE = 2.0  # 🔹포인트: 이미지 해상도를 2배로 향상

    # PDF 파싱을 위한 설정값 지정
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    # PDF를 내부 구조로 변환 (텍스트, 이미지 포함)
    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    conv_res = doc_converter.convert(Path(pdf_path))  # 변환 실행
    doc_filename = conv_res.input.file.stem
    output_dir.mkdir(parents=True, exist_ok=True)  # 출력 폴더 생성

    # 🔹포인트: 각 페이지 이미지로 저장
    image_paths = []
    for page_no, page in conv_res.document.pages.items():
        page_img = output_dir / f"{doc_filename}-page-{page_no}.png"
        page.image.pil_image.save(page_img, format="PNG")
        image_paths.append(str(page_img))

    # 🔹포인트: 그림/표 항목 별도 저장
    pic_id = 0
    for element, _ in conv_res.document.iterate_items():
        if isinstance(element, (PictureItem, TableItem)):  # 그림 또는 표일 경우
            pic_id += 1
            pic_img = output_dir / f"{doc_filename}-element-{pic_id}.png"
            element.get_image(conv_res.document).save(pic_img, "PNG")
            image_paths.append(str(pic_img))

    # 🔹포인트: 페이지별 텍스트 추출
    text_pages = []
    for page in conv_res.document.pages.values():
        if hasattr(page, "text") and page.text:  # 텍스트 속성이 있으면
            text_pages.append(page.text)
        elif hasattr(page, "get_text") and callable(page.get_text):  # get_text 메서드 있을 경우
            try:
                text_pages.append(page.get_text())
            except:
                text_pages.append("")
        else:
            text_pages.append("")

    return text_pages, image_paths  # 텍스트 리스트, 이미지 경로 리스트 반환

# ✅ 텍스트 + 이미지 설명을 문서 형태로 구성
def prepare_documents(text_pages, image_paths):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # 🔹포인트: 텍스트를 중첩되게 쪼개 정보 손실 방지

    for i, page in enumerate(text_pages):  # 각 페이지별 텍스트 쪼개기
        chunks = splitter.split_text(page)
        for j, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={"source": f"PDF Page {i+1}", "chunk": j+1}))

    for img_path in tqdm(image_paths, desc="이미지 설명 생성 중"):  # 🔹포인트: 이미지마다 설명 생성
        caption = generate_caption(img_path)
        docs.append(Document(page_content=caption, metadata={"image_path": img_path}))

    return docs  # 문서 객체 리스트 반환

# ✅ Step 1: 텍스트/이미지를 벡터로 변환 후 저장
def preprocess_and_store(pdf_path, persist_path="chroma_store"):
    text_pages, image_paths = parse_pdf_and_extract_images(pdf_path)  # 🔹포인트: PDF에서 텍스트/이미지 추출
    docs = prepare_documents(text_pages, image_paths)  # 🔹포인트: 문서 객체로 구성
    vectorstore = Chroma.from_documents(  # 🔹포인트: 벡터 DB에 문서 저장
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_path
    )
    vectorstore.persist()  # 디스크에 벡터 저장소 저장
    print(f"✅ 벡터 DB 저장 완료: {persist_path}")

# ✅ Step 2: RAG 체인 구성
def load_rag_chain(persist_path="chroma_store"):
    vectorstore = Chroma(  # 저장된 벡터 DB 로드
        embedding_function=embeddings,
        persist_directory=persist_path
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # 🔹포인트: 가장 관련도 높은 5개 문서 반환

    prompt = PromptTemplate(  # 🔹포인트: 프롬프트 양식 (문맥 + 질문)
        input_variables=["context", "question"],
        template=(
            "당신은 문서와 이미지 설명을 참고해 질문에 답하는 AI입니다.\n"
            "다음 문맥을 바탕으로 질문에 답하십시오.\n"
            "문맥:\n{context}\n\n"
            "질문: {question}\n"
            "답변:"
        )
    )

    # 🔹포인트: 체인 구성 - 검색 → 프롬프트 → LLM → 파싱
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# ✅ 질문이 이미지 관련인지 여부 판별
def is_image_related_question_llm(question: str) -> bool:
    prompt = (
        "질문이 이미지나 사진, 그림, 차트, 도표 등 시각적인 정보와 관련된 것인지 판단해주세요.\n"
        "오직 '예' 또는 '아니오'로만 대답하세요.\n\n"
        f"질문: {question}\n"
        "답변:"
    )
    response = llm.invoke(prompt).content.strip().lower()  # 🔹포인트: GPT-4o의 판단 이용
    return "예" in response  # '예' 포함 여부로 이미지 질문 여부 반환

# ✅ 최종 질의 함수: 텍스트/이미지 판단 후 답변 + 이미지 출력
def ask_conditional_image(chain, retriever, query: str):
    is_image_question = is_image_related_question_llm(query)  # 🔹포인트: 질문이 이미지 관련인지 판별

    response = chain.invoke(query)  # 🔹포인트: GPT-4o로 문서 기반 응답 생성
    print("\n🧠 GPT-4o 응답:")
    print(response)

    if is_image_question:
        print("\n🖼️ 관련 이미지 출력:")
        results = retriever.get_relevant_documents(query)  # 관련 문서 검색
        image_found = False

        for doc in results:
            if "image_path" in doc.metadata:  # 이미지 포함된 문서일 경우
                img_path = doc.metadata["image_path"]
                if Path(img_path).exists():  # 파일 존재 여부 확인
                    print(f"\n이미지 설명: {doc.page_content}")
                    print(f"이미지 경로: {img_path}")
                    try:
                        ipy_display(Image.open(img_path))  # 이미지 노트북에 출력
                        image_found = True
                    except Exception as e:
                        print(f"이미지 표시 실패: {e}")
        if not image_found:
            print("❌ 관련 이미지가 없습니다.")
    else:
        print("\nℹ️ 텍스트 기반 질문입니다. 이미지는 표시하지 않습니다.")

preprocess_and_store("03_sisFramework 교육자료 V5.0.1.pdf")

# ✅ RAG 체인 로드
chain, retriever = load_rag_chain()

# ✅ 질문 실행
while True:
    query = input("질문 > ")
    if query.lower() in ["exit", "q", "quit"]:
        break
    ask_conditional_image(chain, retriever, query)