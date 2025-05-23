# Website Q\&A System using OpenAI GPT and RAG

This Streamlit app enables users to ask questions based on the content of a specific website. It uses OpenAI’s GPT-3.5-turbo, Chroma for vector storage, and LangChain for chaining the logic.

##  Features

* Web scraping of article content using `WebBaseLoader`
* Text chunking for better context management
* Embedding generation using `text-embedding-3-small`
* Retrieval-based question answering using OpenAI GPT-3.5
* Simple Streamlit interface for asking questions

## Target Website

This app scrapes and processes content from:

  * [https://lilianweng.github.io/posts/2023-06-23-agent/]

## Setup Instructions

1. **Clone the repository**

```cmd
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. **Install dependencies**

```cmd
pip install -r requirements.txt
```

3. **Create a `.env` file**

```env
OPENAI_API_KEY=your_openai_api_key
```

4. **Run the Streamlit app**

```cmd
streamlit run app.py
```

##  How It Works

1. Scrapes specific classes from the target blog post.
2. Splits the text into manageable chunks using `RecursiveCharacterTextSplitter`.
3. Converts these chunks into embeddings using OpenAI's embedding model.
4. Stores and retrieves relevant documents with Chroma.
5. Uses a prompt template to query GPT-3.5 for concise answers.
6. Displays the result in a simple web interface.

## Example Usage

* Ask: *"What is the role of an agent in this context?"*
* Receive a concise answer pulled from the document context.

## Dependencies

* Streamlit
* LangChain
* OpenAI API
* BeautifulSoup (bs4)
* Chroma
* python-dotenv
