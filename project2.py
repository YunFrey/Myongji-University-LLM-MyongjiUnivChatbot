# Streamlit 패키지 추가
import streamlit as st
# PDF reader
from PyPDF2 import PdfReader
# Langchain 패키지들
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import BedrockEmbeddings #텍스트를 백터로 바꿔주는 것
from langchain.chat_models import BedrockChat #LLM 라이브러리
from langchain.prompts import PromptTemplate
import gradio as gr
import boto3
import os

######################
# 베드락 호출
######################

session = boto3.Session()
bedrock = session.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"
)

###################### 메인 함수 ######################

######################
##### 경로 내 PDF 파일 로드 #####
######################

#폴더 내의 모든 파일 리스트화
arr = os.listdir('./data/')
#신규 리스트 
newarr = [] 

# pdf 확장자 필터링
for file in arr:
   if ".pdf" in file: 
    newarr.append(file)

print('[불러와진 파일 리스트]\n', newarr)

texts = []
for file in newarr:
   fileloc = './data/'+ file
   loader = PyPDFLoader(fileloc)
   texts.extend(loader.load_and_split())
   print('파일 로드중 : '+file)


####################
# 베드락 임베딩 함수 호출
print('베드락 임베딩 함수 호출')
####################

embedding = BedrockEmbeddings(
        region_name='us-east-1',
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        model_id="amazon.titan-embed-text-v1"
    ) 

####################
# 임베딩 이후 벡터화하여 문서들 저장
print('임베딩 이후 벡터화하여 문서들 저장')
####################
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding)

print('벡터 DB 내장된 벡터의 개수 :', vectordb._collection.count())

#####################
# 벡터화된 결과 저장
#####################

folder_path = './data'
file_name = 'embedding.csv'
file_path = os.path.join(folder_path, file_name)

print('벡터 DB\n')
print(vectordb)

#####################
# 체인 프롬포트 작성
#####################

template = """
너는 명지대학교 학생들의 궁금한 점에 대해 답변해 주는 마루라는 챗봇이야.

대답할 때 너가 지켜야 하는 규칙은 아래와 같아.
- 너는 질문에 대해 무조건 반말로 대답해야 해
- 너가 대답하는 정보는 명지대학교에 대한 정보들이야.
- 만약 질문에 대한 대답을 모르면, 반드시 모른다고 얘기해야해. 억지로 대답을 만들 필요는 없어.

너에게 주어진 문서는 아래와 같고, 반드시 여기에 있는 정보를 기반으로 대답해야해. 그 외 정보는 모른다고 답해야 해.

{context}

질문: {question}
대답: 
"""

prompt = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

#####################
# 체인을 활용
# 유사도가 높은 문서 2개만 추출. k = 2
#####################

retriever = vectordb.as_retriever(search_kwargs={"k": 2})

qa_chain = RetrievalQA.from_chain_type(
    llm=BedrockChat(model_kwargs={"temperature": 0, "max_tokens_to_sample" : 2000},
                        model_id="anthropic.claude-v2",
                        client=bedrock
                    ),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    retriever=retriever,
    return_source_documents=True)

######################
# Gradio 기반 채팅 인터페이스 생성.
######################

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="명지대 학칙봇") # 경제금융용어 챗봇 레이블을 좌측 상단에 구성
    msg = gr.Textbox(label="너가 궁금한 건 뭐야?")  # 하단의 채팅창의 레이블
    clear = gr.Button("대화 초기화")  # 대화 초기화 버튼

    # 챗봇의 답변을 처리하는 함수
    def respond(message, chat_history):
      result = qa_chain(message) # 챗봇의 출력을 뽑고
      bot_message = result['result'] # 답변인 것만 저장

      # 채팅 기록에 사용자의 메시지와 봇의 응답을 추가.
      chat_history.append((message, bot_message))
      return "", chat_history

    # 사용자의 입력을 제출(submit)하면 respond 함수가 호출.
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

    # '초기화' 버튼을 클릭하면 채팅 기록을 초기화.
    clear.click(lambda: None, None, chatbot, queue=False)

# 인터페이스 실행.
demo.launch(debug=True)