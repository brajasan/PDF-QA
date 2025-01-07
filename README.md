# Streamlit PDF QA Application

This is a Streamlit-based web application that allows users to upload PDF documents and ask questions about their content. The app uses LangChain, an advanced AI framework, to process the text and generate answers interactively, providing a seamless and engaging user experience.

---

## Overview

The **Streamlit PDF QA Application** is designed to make document analysis more efficient by enabling users to interact with uploaded PDFs as though conversing with an expert. Key features of the application include:

### Features:
1. **PDF Upload**: Upload any PDF document to analyze its content.
2. **Vectorstore Creation**: Automatically creates a searchable vectorstore using text embeddings for efficient retrieval.
3. **Question Answering**: Ask questions in natural language, and the application responds with precise answers.
4. **Memory-Enhanced Conversations**: Maintains the context of the conversation for a more human-like interaction.
5. **Real-time Response Streaming**: Displays responses dynamically for an engaging user experience.

### Technical Highlights:
- **LangChain Integration**: Utilizes LangChain's advanced modules, including `RetrievalQA`, `ConversationBufferMemory`, and custom LLM models.
- **Ollama Model**: Powered by the `llama3.2:1b` small model but powerful.
- **Streamlit Interface**: Provides a simple and intuitive web interface for users.

---

## How It Works

1. **Upload PDF**: Drag and drop your PDF file into the interface.
2. **Document Processing**: The app processes the PDF using LangChain tools, creating a vectorstore for efficient search and retrieval.
3. **Interactive QA**: Ask questions related to the uploaded document, and the application will use the context and memory to provide concise answers.

---

## Conclusion

The **Streamlit PDF QA Application** bridges the gap between static documents and interactive learning. By combining LangChain's powerful AI capabilities with Streamlit's user-friendly interface, the app empowers users to extract meaningful insights from their documents effortlessly. Whether you're analyzing reports, studying research papers, or reviewing contracts, this application offers a powerful, conversational approach to document exploration.

Feel free to extend this project by experimenting with additional models, fine-tuning the embeddings, or adding support for other file types.

---

## Installation and Usage

### Prerequisites:
- Python 3.8 or higher
- Required libraries: `streamlit`, `langchain`, `chromadb`

### Setup:
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Preparing Ollama with llama3.2 models:
   CPU only:
   ```bash
   docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
   docker exec -it ollama ollama run llama3.2:1b
   ```
   Check the documentation: https://hub.docker.com/r/ollama/ollama
4. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## Contributing

Contributions are welcome!

If you find a bug or have a suggestion for improvement, please open an issue on the GitHub repository.


---

## License

This project is licensed under the MIT License.