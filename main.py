from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

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
documents = loader.load()

#page numbers % content example
print(f"{len(documents)} pages loaded.")
#print(documents[0].page_content[:500]) it worked.

low_text_pages = []

#checking the character numbers of each page(pypdfloader)
for i, doc in enumerate(documents):
    text_length = len(doc.page_content.strip())
    if text_length < 1500: 
        print(f"Page {i + 1} has low text ({text_length} chars)")
        low_text_pages.append(i)

#adding metadata to docs
for doc in documents:
    doc.metadata["source"] = "attention.pdf"
    doc.metadata["page_number"] = doc.metadata.get("page", -1)

from pdf2image import convert_from_path

#Saving the low text pages as png images
POPPLER_PATH = os.getenv("POPPLER_PATH")

for idx, page_num in enumerate(low_text_pages):
    image = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH, first_page=page_num+1, last_page=page_num+2)[0]
    image.save(f"page_{page_num+1}.png", "PNG") #Saves the image as a PNG file named "page_num+1.png" in the current directory.
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
        img_path = f"page_{page_num+1}.png"
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



from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

#checks to see if the vectorstore exists from before
if os.path.exists("faiss_index"):
    print("vectorestore already exists!")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    #allow_dangerous_deserialization=True do this only when you created the file yourslef and nobody else has access
else:
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local("faiss_index")
    print("vectorestore was created!")
    
#retrieve
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

#chat model
chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

# creating a q&a chain
qa_chain = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever)


while(True):
    #question
    query = input("Please enter your question: ")
    #invoking the chain
    if query.lower() == 'exit':
        break
    else:
        response = qa_chain.invoke(query, return_only_outputs=False)
        print(response["result"])




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