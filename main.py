from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")


import logging  # Import the logging module to handle logging in the application
import os       # Import the os module to work with the file system

# Create a folder named 'logs' if it doesn't already exist
os.makedirs("logs", exist_ok=True)

# Create a logger object with a custom name ('file_api_logger')
logger = logging.getLogger("file_api_logger")

# Set the minimum level of messages this logger will handle (INFO and above)
logger.setLevel(logging.INFO)

# Create a file handler that writes log messages to 'logs/app.log'
file_handler = logging.FileHandler("logs/app.log")

# Define the format of log messages:
# - asctime: the date and time of the log
# - api_path: (custom field – you'll need to pass it in the log manually)
# - levelname: the log level (e.g. INFO, ERROR)
# - message: the actual log message
formatter = logging.Formatter(
    "%(asctime)s, %(api_path)s, %(levelname)s, %(message)s",
    datefmt="%Y-%m-%d, %H:%M:%S"
)

# Attach the formatter to the file handler so logs are saved in that format
file_handler.setFormatter(formatter)

# Add the file handler to the logger so it knows where to send log messages
logger.addHandler(file_handler)


#cleans the text
def clean_text(text):
    import re
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


from langchain_community.document_loaders import PyPDFLoader

#file path
pdf_path = "attention.pdf"

#loading the pdf
loader = PyPDFLoader(pdf_path)
logger.info(f"Loading PDF: {pdf_path}", extra={"api_path": "load_pdf"})
documents = loader.load()
logger.info(f"{len(documents)} pages loaded.", extra={"api_path": "load_pdf"})
            
        
#page numbers % content example
print(f"{len(documents)} pages loaded.")
#print(documents[0].page_content[:500]) it worked.

low_text_pages = []

#checking the character numbers of each page(pypdfloader)
for i, doc in enumerate(documents):
    text_length = len(doc.page_content.strip())
    if text_length < 1500: 
        print(f"Page {i + 1} has low text ({text_length} chars)")
        logger.warning(f"Page {i + 1} has low text ({text_length} chars)", extra={"api_path": "check_low_text"})
        low_text_pages.append(i)

#adding metadata to docs
for doc in documents:
    doc.metadata["source"] = "attention.pdf"
    doc.metadata["page_number"] = doc.metadata.get("page", -1)

from pdf2image import convert_from_path
POPPLER_PATH = os.getenv("POPPLER_PATH")

image_dir = "data/images"
os.makedirs(image_dir, exist_ok=True)
    
#Saving the low text pages as png images
for idx, page_num in enumerate(low_text_pages):
    image_path = os.path.join(image_dir, f"page_{page_num+1}.png")
    image = convert_from_path(pdf_path, dpi=300, poppler_path= POPPLER_PATH, first_page=page_num+1, last_page=page_num+2)[0]
    image.save(image_path, "PNG") #Saves the image as a PNG file named "page_num+1.png" in the current directory.
    logger.info(f"Saved page {page_num+1} as image", extra={"api_path": "convert_to_image"})
    print(f"Saved page {page_num+1} as image")

import pytesseract
from PIL import Image
TESSERACT_PATH = os.getenv("TESSERACT_PATH")

#manual configuration
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


import json
#checks to see if the ocr_text exists from before
if os.path.exists("ocr_texts.json"):
    print("ocr_texts.json already exists!")
    with open('ocr_texts.json', 'r') as f:
        ocr_texts = json.load(f)

else:
    #text extracted from low text page imgs with ocr
    ocr_texts = []

    for page_num in low_text_pages:
        img_path = os.path.join(image_dir, f"page_{page_num+1}.png")
        img = Image.open(img_path)
        text = pytesseract.image_to_string(img)
        print(f"OCR text from page {page_num+1}: {len(text)} chars")
        ocr_texts.append({
            "page": page_num,
            "text": text
        })    
    with open("ocr_texts.json","w") as f:
        json.dump(ocr_texts, f)
        print("created ocr_texts.json!")
    
    

#creating docs from the extracted texts with ocr
from langchain_core.documents import Document

ocr_documents = []

for item in ocr_texts:
    page_number = item["page"]
    text = item["text"].strip()
    
    
    if len(text) > 100: #To  delete short texts or low quality images
        doc = Document(
            page_content=text,
            metadata={
                "source": pdf_path,
                "page": page_number,
                "page_number": page_number
            }
        )
        ocr_documents.append(doc)
        print(f"OCR Document created for page {page_number+1}")
    else:
        print(f"Skipped page {page_number+1} due to short OCR text")


#removing low-text pages from original documents
filtered_documents = [doc for i, doc in enumerate(documents) if i not in low_text_pages]

#Combining filtered documents with OCR documents
final_documents = filtered_documents + ocr_documents

# clean  documents
for doc in final_documents:
    doc.page_content = clean_text(doc.page_content)
    
#Sort the final documents by page number
final_documents.sort(key=lambda doc: doc.metadata["page_number"])

print(f"Total final documents after merging: {len(final_documents)}")


from langchain.text_splitter import RecursiveCharacterTextSplitter

#splitting into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

split_docs = text_splitter.split_documents(final_documents)

print(f"Total chunks: {len(split_docs)}")
#print(split_docs[0].page_content[:300]) it worked.
#print("meta:",split_docs[50].metadata)


from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

#memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

from langchain_postgres import PGVector

# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3
collection_name = "my_docs"

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)
vectorstore.add_documents(split_docs)
print("PG vectorestore was created!")

#retrieve
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

from langchain_openai import ChatOpenAI
#from langchain_community.chat_models import ChatOpenAI deprecated
from langchain.chains import RetrievalQA

#chat model
chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

# creating a q&a chain
#qa_chain = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever, memory=memory)
#output_key="result"  saves the result in the memory, it can only save one(result or source_document)

from langchain.chains import ConversationalRetrievalChain

qa_chain = ConversationalRetrievalChain.from_llm(llm=chat, chain_type="stuff", retriever=retriever, memory=memory)


while(True):
    #question
    query = input("**Please enter your question: ")
 
    #invoking the chain
    if query.lower() == 'exit':
        logger.info("User exited the chat", extra={"api_path": "user_query"})
        break
        
    logger.info("User entered a question", extra={"api_path": "user_query"})
    logger.info(f"User query: {query}", extra={"api_path": "user_query"})

    try:
        context = qa_chain.invoke(query, return_only_outputs=False)
        print(context)
        prompt = f"""Answer the following question, 
                    - If the context contains relevant information to answer the question, answer ONLY based on that information.
                    - If the context does NOT contain relevant information, answer the question based on your own knowledge.

                    Context:
                    {context["answer"]}
                    

                    Question:
                    {query}
        """
        response = chat.invoke(prompt)
        from langchain_core.messages import AIMessage, HumanMessage

        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(response.content)
        
        print(prompt)

        logger.info(f"Model response (first 200 chars): {response.content[:200]}", extra={"api_path": "llm_response"})

        print(response.content)

    except Exception as e:
        logger.error(f"Error during chat: {e}", extra={"api_path": "llm_response"})
        print("An error occurred. Please try again.")
    


#print(response["result"])
#print(response["source"])

"""
Page 13 has low text (812 chars)
Page 14 has low text (814 chars)
Page 15 has low text (817 chars)
Saved page 13 as image
Saved page 14 as image
Saved page 15 as image
OCR text from page 13: 845 chars
OCR text from page 14: 771 chars
OCR text from page 15: 779 chars
""" 