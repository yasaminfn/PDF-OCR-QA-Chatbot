# PDF-QA with OCR, LangChain, and PostgreSQL

This project extracts and processes content from PDF files (including OCR for low-text pages) to enable a fully conversational question-answering system using OpenAI, LangChain, and a PostgreSQL vector database (with `pgvector`).

---

## Features

- **Load and parse PDFs** using `PyPDFLoader` from LangChain community.
- **Detect low-text pages** and convert them to images for OCR processing.
- **Perform OCR** on low-text pages using Tesseract to extract missing textual content.
- **Merge OCR-extracted text** with original PDF text.
- **Clean and chunk** the combined text into manageable parts.
- **Store embeddings** for chunks in a PostgreSQL database with `pgvector` extension.
- **Conversational retrieval-based Q&A** using OpenAI GPT-3.5-turbo model.
- **Persistent conversation memory** saved in PostgreSQL to maintain chat context.
- **Structured logging** for tracking the pipeline and user queries.

---

## Prerequisites

- Python 3.8 or higher  
- PostgreSQL with `pgvector` extension enabled  
- Poppler (for PDF to image conversion)  
- Tesseract OCR  
- Docker (optional, for running PostgreSQL with pgvector)  
- OpenAI API key

---

## Project Structure
```
├── data/
│ ├── attention.pdf
│ └── images/ # Images of low-text PDF pages
├── logs/
│ └── app.log # Application logs(not committed)
├── main.py # Main script
├── requirements.txt
├── README.md
└── .env # Environment variables (not committed)
```

