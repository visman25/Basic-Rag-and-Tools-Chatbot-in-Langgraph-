from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph, START,END
from typing import TypedDict,NotRequired,Annotated,Literal,Any, Dict, Optional
from pydantic import BaseModel,Field
import operator
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
import requests,os,tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
# -------------------
# 1. LLM
# -------------------
model = ChatOllama(model="qwen2.5:7b", temperature=0)
embeddings=OllamaEmbeddings(model="qwen3-embedding:0.6b")
# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """Build a FAISS retriever for the uploaded PDF and store it for the thread."""
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        # Drop empty chunks and null characters
        clean_chunks = []
        for c in chunks:
            text = c.page_content.replace("\x00", "").strip()
            if text:
                c.page_content = text
                clean_chunks.append(c)

        if not clean_chunks:
            raise ValueError("No extractable text found. The PDF may be scanned (image-only).")

        # Build FAISS in small batches
        batch_size = 8
        vector_store = FAISS.from_documents(clean_chunks[:batch_size], embeddings)
        for i in range(batch_size, len(clean_chunks), batch_size):
            vector_store.add_documents(clean_chunks[i:i + batch_size])

        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        info = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(clean_chunks),
        }
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = info
        return info
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

# -------------------
# 3. Tools
# -------------------
# Tools
search_tool = DuckDuckGoSearchRun(region="us-en")
@tool
def web_search(query: str) -> str:
    """Search the web. ONLY use for current events or facts you cannot know.
    Never use for greetings or casual chat."""
    return search_tool.run(query)
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform basic arithmetic on two numbers.
    Supported operations: add, sub, mul, div."""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation: {operation}"}

        return {"result": result}

    except Exception as e:
        return {"error": str(e)}

@tool 
def stock(symbol:str)->dict:
        """Fetch latest stock price for a given symbol(eg. 'AAPL','Tesla')
        using Alpha Vantage API key in the URL """
        url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=WUJHA79T4J'
        r = requests.get(url)
        return r.json()

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [web_search,stock, calculator, rag_tool]

model_tools=model.bind_tools(tools)
# -------------------
# 4. State
# -------------------
class chatstate(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]
# -------------------
# 5. Nodes
# -------------------

def chat_node(state: chatstate,config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = model_tools.invoke(messages, config=config)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# -------------------
# 5. Checkpointer
# -------------------
conn = sqlite3.connect(database='chatbot.db',check_same_thread=False)

checkpointer= SqliteSaver(conn=conn)
# -------------------
# 6. Graph
# -------------------
graph = StateGraph(chatstate)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
def retrive_all_threads():

    all_threads=set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)
def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})