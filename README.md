# Document Extraction & Search API 🔍📄

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)

This FastAPI project provides an API for uploading text files, PDFs, and images to extract text using OCR (Optical Character Recognition) and vector embedding search in MongoDB Atlas. It also integrates with Google Gemini AI to answer questions using a retrieval-augmented generation (RAG) approach.

## 📋 Table of Contents
- [🚀 Getting Started](#-getting-started)
  - [📋 Prerequisites](#-prerequisites)
  - [⚙️ Installation](#️-installation)
  - [🔐 Environment Variables](#-environment-variables)
- [📚 API Documentation](#-api-documentation)
  - [🏠 Root Endpoint](#-root-endpoint)
  - [📥 Extract Text](#-extract-text)
  - [🔎 Search](#-search)
- [🧠 Model and Library Details](#-model-and-library-details)

## 🚀 Getting Started

### 📋 Prerequisites
- Python 3.8+
- MongoDB Atlas account for vector storage
- Google Gemini API key

### ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd <repository-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your `.env` file:
   - Create a `.env` file in the project root with the following variables:

     ```env
     MONGO_URI=<your_mongo_uri>
     GEMINI_API_KEY=<your_gemini_api_key>
     ```

4. Run the application:
   ```bash
   uvicorn app:app --reload
   ```

### 🔐 Environment Variables
| Variable | Description |
|----------|-------------|
| `MONGO_URI` | The MongoDB URI to connect to your MongoDB Atlas cluster |
| `GEMINI_API_KEY` | API key for Google Gemini Generative AI |

## 📚 API Documentation

### 🏠 Root Endpoint
**Endpoint:** `GET /`

- **Description**: A simple endpoint to confirm that the server is running.
- **Response**: `"Hello!"`

### 📥 Extract Text
**Endpoint:** `POST /extract-text`

- **Description**: Upload a file (text, PDF, or image) to extract text and store vector embeddings in MongoDB.
- **Request**:
  - **File**: Upload the file as `file` in the request body.
- **Response**:
  - **Success**:
    ```json
    {
      "extracted_text": "<Extracted text from the file>"
    }
    ```
  - **Error**: Returns an error if the file type is unsupported.

- **Supported File Types**:
  | File Type | Description |
  |-----------|-------------|
  | `.txt` | Plain text files |
  | `.pdf` | PDF documents |
  | Image files | e.g., `.png`, `.jpg` |

### 🔎 Search
**Endpoint:** `POST /search`

- **Description**: Accepts a prompt, retrieves relevant document passages, and uses the Gemini AI model to generate an answer.
- **Request**:
  - **Body**: JSON object with the prompt.
    ```json
    {
      "prompt": "Enter your question here."
    }
    ```
- **Response**:
  - **Result**:
    ```json
    {
      "result": "<Generated answer based on the prompt and retrieved documents>"
    }
    ```
  - **Error**: Returns an error message if processing fails.

## 🧠 Model and Library Details

| Component | Description |
|-----------|-------------|
| Text Extraction | Uses `pytesseract` for OCR on image files |
| Document Storage | Utilizes MongoDB Atlas for vector embeddings of extracted text |
| Language Model | Google Gemini AI (`gemini-1.5-flash`) for answering questions based on retrieved passages |
| Embeddings | Uses `HuggingFaceEmbeddings` for text-to-vector transformation |
| Document Chunking | Text is chunked into paragraphs and processed as individual documents for efficient search and retrieval |

---

📝 **Note**: This project is designed to provide efficient document processing and intelligent search capabilities. For any issues or feature requests, please open an issue in the repository.
