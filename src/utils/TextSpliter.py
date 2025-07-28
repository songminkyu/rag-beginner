from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# PyPDFLoader를 사용하여 PDF 파일 로드
loader = PyPDFLoader("../Data/2024_KB_부동산_보고서_최종.pdf")
pages = loader.load()

# PDF 파일의 모든 페이지에서 텍스트를 추출하여 총 글자 수 계산
print('총 글자 수:', len(''.join([i.page_content for i in pages])))


# RecursiveCharacterTextSplitter 초기화
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# 문서 분할
texts = text_splitter.split_documents(pages)
print('분할된 청크의 수:', len(texts))


print('첫번째 청크 출력:', texts[1].page_content)