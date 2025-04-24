from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
first_input = Document(page_content="""Hello""",metadata={})
vectorestore1 = FAISS.from_documents(documents=[first_input],embedding=embedding)
vectorestore1.save_local('SummerEyes_local_db')

