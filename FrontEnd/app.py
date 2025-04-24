import os
import sys
from io import BytesIO
import tempfile
import requests

from dotenv import load_dotenv
import streamlit as st
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from groq import Groq

# LangChain tools
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Local utility
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from BackEnd.pdf_utils import extract_docs_and_text_from_pdf


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=250)
st.session_state.vectordatabase = FAISS.load_local("SummerEyes_local_db",embeddings=embedding,allow_dangerous_deserialization=True)
st.session_state.retriever = st.session_state.vectordatabase.as_retriever(kwargs={"k":5})
# Load environment variables
load_dotenv()

# Get the API key and model name from the .env file
API_KEY = os.environ.get('GROQ_API_KEY')
MODEL_NAME = os.getenv('GROQ_MODEL')

client = Groq(
    api_key=API_KEY,
)
# Streamlit settings
st.set_page_config(page_title="SummerEyes", layout="wide")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Pacifico&display=swap" rel="stylesheet">
    <style>
    /* Make the container go below the sticky header */
    div.block-container {
        padding-top: 8rem;
    }

    /* Style the sticky header */
    .sticky-title {
        position: fixed;
        top: 3rem;
        width: 100%;
        background-color: #0e1117;
        font-size: 4rem;
        font-weight: 400;
        z-index: 100;
        text-align: left;
        color: #ffffff;
        font-family: "Pacifico", cursive;
    }
    </style>

    <div class="sticky-title">
        SummerEyes
    </div>
""", unsafe_allow_html=True)

class RAG():
    
    def __init__(self) -> None:
        pass

    def search_db(self,question:str,retriever):
        retrievals = retriever.invoke(question)
        
        context = "\n".join([ret.page_content for ret in retrievals])
        return context 

    
    def search_the_web(self,urls:list):

        urls = [url for url in urls]

        docs_scrape = [WebBaseLoader(url).scrape() for url in urls]
        docs_load = [WebBaseLoader(url).load() for url in urls]
        # docs_list = [item for sublist in docs for item in sublist]
        docs_scrape = "\n".join([str.strip(i.getText()) for i in docs_scrape[0].find_all(["p","code","li"])])
        docs_load[0][0].page_content = docs_scrape
        # docs_list = Document(page_content=docs_list,metadata={})
        docs_list = docs_load[0][0]
        
        return [docs_list]
    
    def extract_from_pdfs(self,path:str):
        docs_list = list()
        for i in range(1):
            loader=PyPDFLoader(path)
            pages=loader.load()
            docs_list.extend([page for page in pages])
        return docs_list 
        
    def splitter_func(self,docs,vectordatabase,embedding):
        
        split_docs = splitter.split_documents(docs)
        
        # Add the document chunks to the "vector store"
        
        vectordatabase.add_documents(
            documents=split_docs,embeddings=embedding
        )
        vectordatabase.save_local("SummerEyes_local_db")
        
        retriever = vectordatabase.as_retriever(search_kwargs={'k': 5
                                                               })
        
        return retriever,vectordatabase
    
    def invoke(self,variables:dict):
        answer = llm(variables,True)
        return answer
    

def llm(inp,_stream):
    return client.chat.completions.create(
            #
            # Required parameters
            #
            messages=inp,

            # The language model which will generate the completion.
            model=MODEL_NAME,

            #
            # Optional parameters
            #

            # Controls randomness: lowering results in less random completions.
            # As the temperature approaches zero, the model will become deterministic
            # and repetitive.
            temperature=0.5,

            # The maximum number of tokens to generate. Requests can use up to
            # 32,768 tokens shared between prompt and completion.
            max_completion_tokens=1024,

            # Controls diversity via nucleus sampling: 0.5 means half of all
            # likelihood-weighted options are considered.
            top_p=1,

            # A stop sequence is a predefined or user-specified text string that
            # signals an AI to stop generating content, ensuring its responses
            # remain focused and concise. Examples include punctuation marks and
            # markers like "[end]".
            stop=None,

            # If set, partial message deltas will be sent.
            stream=_stream,
        )

rag = RAG()

system = [{"role": "system", "content": "You are a helpful assistant, your name is SummerEyes, your core functionality is to create summaries when asked and answer user queries, answer based on context if provided, else answer using your own knowledge."}] 
summary_response = None

if "messages" not in st.session_state:
    st.session_state.messages = []
if "audio_text" not in st.session_state:
    st.session_state.audio_text = None
if "audio_processed" not in st.session_state:
    st.session_state.audio_processed = False
if "pdfs" not in st.session_state:
    st.session_state.pdfs = []
if 'db' not in st.session_state:
    st.session_state.db = False
if 'rag' not in st.session_state:
    st.session_state.rag = False
if 'context' not in st.session_state:
    st.session_state.context = ''
if 'get_summary' not in st.session_state:
    st.session_state.get_summary = False 

with st.sidebar:
    uploaded_file = st.audio_input('Speak')
    l,r = st.columns(2)
    if l.button('',icon=':material/description:') and uploaded_file:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        response_new = requests.post("http://localhost:8000/transcribe", files=files)
        result = response_new.json()

        st.session_state.audio_text = result["text"]
        st.session_state.audio_processed = True
        if len(st.session_state.audio_text) <= 100:
            st.write('Transcription')
            st.write(st.session_state.audio_text)
        # st.rerun()
    
    if r.button('clear'):
        st.session_state.messages = []
    if st.checkbox('Get Summary'):
        st.session_state.get_summary = True
    else:
        st.session_state.get_summary = False 
    if st.checkbox('Save to Database'):
        st.session_state.db = True
    else:
        st.session_state.db = False 
    if st.checkbox('Use RAG'):
        st.session_state.rag = True
    else:
        st.session_state.rag = False 
    if st.session_state.pdfs:
        for pdf in st.session_state.pdfs:
            with st.expander(pdf['title']):
                st.write(pdf['text'])

    
# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        l,r = st.columns([12,1])
        l.markdown(message["content"])
        with r:
            st.components.v1.html(f"""
        <button onclick="navigator.clipboard.writeText(`{message['content']}`)"
                style="background:none;border:none;cursor:pointer;font-size:1em;">📋</button>
    """, height=30)


# If we got a transcription from audio, respond to that
if st.session_state.audio_text:
    user_text = st.session_state.audio_text

    # Display user message
    # with st.chat_message("user"):
    #     st.markdown(user_text)
    # st.session_state.messages.append({"role": "user", "content": user_text})

    # Get LLM response
    messages = [{"role": "system", "content": "You are a helpful assistant, Your name is SummerEyes, your core task is to summarise"}] + [{"role": "user", "content": user_text}]
    response = llm(messages, False)
    assistant_reply = response.choices[0].message.content

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

    # Clear processed audio text
    st.session_state.audio_text = None
    
    


if prompt :=st.chat_input("Summerize with SummerEyes",accept_file=True):
    # st.write(prompt)
    if prompt['files'] != []:

        docs,text = extract_docs_and_text_from_pdf(prompt['files'])
        if st.session_state.db:
            st.session_state.retriever,st.session_state.vectordatabase = rag.splitter_func(docs,st.session_state.vectordatabase,embedding)
        if st.session_state.get_summary:
            st.session_state.messages.append({"role": "user", "content":prompt['text']})
            with st.chat_message("user"):
                l,r = st.columns([12,1])
                l.markdown(prompt['text'])
            with r:
                st.components.v1.html(f"""
        <button onclick="navigator.clipboard.writeText(`{prompt['text']}`)"
                style="background:none;border:none;cursor:pointer;font-size:1em;">📋</button>
    """, height=30)
            messages = [{"role": "user", "content": f"Summarize this:\n{text}"}]
            full_prompt = system + messages
            summary_response = llm(full_prompt, False)
            summary_text = summary_response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": summary_text})
    
            pdf_buffer = BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            width, height = letter

            # Word wrap the text
            lines = summary_text.split('\n')
            y = height - 40
            for line in lines:
                wrapped = [line[i:i+100] for i in range(0, len(line), 100)]
                for part in wrapped:
                    c.drawString(40, y, part)
                    y -= 15
                    if y < 40:
                        c.showPage()
                        y = height - 40

            c.save()
            pdf_buffer.seek(0)

            st.success("✅ Summary Generated!")

            messages = [{"role": "user", "content": f"Give a 4-10 word title for this:\n{summary_text}"}]
            full_prompt = system + messages
            title = llm(full_prompt,False)
            # Provide download link
            st.session_state.pdfs.append({'title':title.choices[0].message.content,'text':summary_text})
            with st.markdown(''):
                st.download_button(
                label=f"📥 Download {title.choices[0].message.content}",
                data=pdf_buffer,
                file_name=f"SummerEyesed-{title.choices[0].message.content}.pdf",
                mime="application/pdf")
            with st.chat_message("assistant"):
                l,r = st.columns([12,1])
                l.markdown(summary_text)
            with r:
                st.components.v1.html(f"""
        <button onclick="navigator.clipboard.writeText(`{prompt['text']}`)"
                style="background:none;border:none;cursor:pointer;font-size:1em;">📋</button>
    """, height=30)
            
            
        
    else:
        if st.session_state.rag:
            syst = [{"role": "system", "content": f"You are a Prompt Optimisation assistant, You Re-Structure the given prompt for vector database search so that relevant keywords are included which might be missing from the original prompt"}]
            msg = [{"role": "user", "content": f"original prompt:\n{prompt['text']}"}]
            f_prompt = syst + msg
            opt_prompt = llm(f_prompt, False)
            opt_prompt_text = opt_prompt.choices[0].message.content
            st.session_state.context = rag.search_db(opt_prompt_text,st.session_state.retriever)
        if st.session_state.context:
            contextual_prompt = [{"role": "user", "content":f"context: {st.session_state.context}\nuser query:{prompt['text']}"}]
            st.session_state.messages.append({"role": "user", "content":prompt['text']})
        else:
            st.session_state.messages.append({"role": "user", "content": prompt['text']})
        with st.chat_message("user"):
            l,r = st.columns([12,1])
            l.markdown(prompt['text'])
            with r:
                st.components.v1.html(f"""
        <button onclick="navigator.clipboard.writeText(`{prompt['text']}`)"
                style="background:none;border:none;cursor:pointer;font-size:1em;">📋</button>
    """, height=30)
        with st.chat_message("assistant"):
            l,r = st.columns([12,1])
            if st.session_state.context:
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                input = system + messages + contextual_prompt
            else:
                messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]
                input = system + messages 
            
            stream = llm(input,True)
            
            response = l.write_stream((chunk.choices[0].delta.content for chunk in stream if chunk.choices[0].delta.content))
            with r:
                st.components.v1.html(f"""
        <button onclick="navigator.clipboard.writeText(`{response}`)"
                style="background:none;border:none;cursor:pointer;font-size:1em;">📋</button>
    """, height=30)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.context = ''
