# Install dependencies first:
# pip install python-docx "pandas==2.2.3" openpyxl PyPDF2 sentence-transformers faiss-cpu langchain-text-splitters groq gradio
# pip install -q langgraph langchain-groq langchain-chroma langchain-huggingface networkx matplotlib chroma-hnswlib sentence-transformers

import os
import re
import sys
import subprocess
import uuid
from collections import Counter

# Automated Dependency Installation Check
required_pkgs = [
    "langchain-groq", "langchain-chroma", "langchain-huggingface",
    "langgraph", "networkx", "plotly", "pypdf", "python-docx", "openpyxl"
]

try:
    import langchain_groq
    import pypdf
    import docx
    import openpyxl
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + required_pkgs)
    import pypdf
    import docx
    import openpyxl

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from typing import Dict, TypedDict, List, Annotated, Sequence

import gradio as gr

# LangChain & LangGraph Core Setup
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------------------------------------------------
# 1. Global Vector Database & Graph Memory
# ---------------------------------------------------------------------------
VECTOR_STORE = None
DOCUMENT_GRAPH = nx.DiGraph()

# ---------------------------------------------------------------------------
# 2. State Definition
# ---------------------------------------------------------------------------
class RAGState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: List[Document]
    requires_human: bool
    escalation_reason: str
    api_key: str
    model_name: str

# ---------------------------------------------------------------------------
# 3. RAG Node Implementations
# ---------------------------------------------------------------------------
def retrieve_node(state: RAGState) -> Dict:
    messages = state.get("messages", [])
    if not messages:
        return {"context": []}

    user_query = messages[-1].content

    if VECTOR_STORE is None:
        return {"context": []}

    retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(user_query)
    return {"context": docs}

def generate_node(state: RAGState) -> Dict:
    groq_api_key = state.get("api_key", "")
    model_name = state.get("model_name", "llama-3.3-70b-versatile")

    if not groq_api_key or not groq_api_key.strip():
        return {"messages": [AIMessage(content="⚠️ Error: Groq API Key is missing. Please enter it on the left panel.")]}

    user_msg = state["messages"][-1].content if state["messages"] else ""

    # Human Escalation Trigger
    if any(keyword in user_msg.lower() for keyword in ["refund", "cancel", "escalate", "supervisor"]):
        return {
            "requires_human": True,
            "escalation_reason": "Escalation requested (Refund/Cancellation)",
            "messages": [AIMessage(content="🔄 Routing request to a human supervisor for manual approval...")]
        }

    docs = state.get("context", [])
    if docs:
        context_text = "\n\n".join([f"Source ({d.metadata.get('source', 'Doc')}): {d.page_content}" for d in docs])
    else:
        context_text = "No document uploaded or matching context found in vector storage."

    try:
        llm = ChatGroq(model_name=model_name, groq_api_key=groq_api_key, temperature=0.1)
        system_instruction = SystemMessage(
            content=(
                "You are an accurate Knowledge-Graph RAG Assistant. Answer the user's question "
                f"STRICTLY using the document context below. If context is missing, inform the user.\n\nCONTEXT:\n{context_text}"
            )
        )
        response = llm.invoke([system_instruction] + list(state["messages"]))
        return {"messages": [response], "requires_human": False}
    except Exception as e:
        return {"messages": [AIMessage(content=f"❌ Groq LLM Error: {str(e)}")]}

def human_review_node(state: RAGState) -> Dict:
    return {"requires_human": False}

# ---------------------------------------------------------------------------
# 4. Graph Construction & Compilation
# ---------------------------------------------------------------------------
def route_after_generate(state: RAGState) -> str:
    if state.get("requires_human", False):
        return "human_review_node"
    return END

builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.add_node("human_review_node", human_review_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_conditional_edges(
    "generate",
    route_after_generate,
    {"human_review_node": "human_review_node", END: END}
)
builder.add_edge("human_review_node", "generate")

memory = MemorySaver()
compiled_app = builder.compile(checkpointer=memory, interrupt_before=["human_review_node"])

# ---------------------------------------------------------------------------
# 5. File Extractor & Keyword Engine for Graph Plotting
# ---------------------------------------------------------------------------
def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    elif ext == ".pdf":
        reader = pypdf.PdfReader(file_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

    elif ext in [".docx", ".doc"]:
        doc = docx.Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs if p.text])

    elif ext in [".xlsx", ".xls"]:
        wb = openpyxl.load_workbook(file_path, data_only=True)
        excel_texts = []
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            excel_texts.append(f"--- Sheet: {sheet_name} ---")
            for row in sheet.iter_rows(values_only=True):
                row_vals = [str(cell) for cell in row if cell is not None]
                if row_vals:
                    excel_texts.append(" | ".join(row_vals))
        text = "\n".join(excel_texts)

    return text

STOPWORDS = {"this", "that", "with", "from", "have", "more", "your", "which", "there", "about", "their", "would", "none", "sheet"}

def extract_top_keywords(text: str, top_n: int = 3) -> List[str]:
    words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
    filtered = [w.capitalize() for w in words if w not in STOPWORDS]
    return [kw for kw, _ in Counter(filtered).most_common(top_n)]

def chunk_terms(text: str) -> set:
    return {w.capitalize() for w in re.findall(r'\b[A-Za-z]{4,}\b', text.lower()) if w not in STOPWORDS}

# ---------------------------------------------------------------------------
# Graph scaling: the number of spheres grows linearly with the volume of data
# (one sphere per CHUNKS_PER_NODE chunks) until MAX_GROUP_NODES is reached.
# Past that cap each sphere simply represents more chunks, so the 3D view stays
# responsive for very large uploads.
# ---------------------------------------------------------------------------
CHUNKS_PER_NODE = 10
MAX_GROUP_NODES = 800
GROUPS_PER_TOPIC = 8          # one topic sphere per 8 group spheres
MIN_TOPICS, MAX_TOPICS = 6, 120
LINKS_PER_GROUP = 3           # topic links drawn from each group sphere
GENERIC_TERM_RATIO = 0.09     # on large corpora ignore words found in >9% of chunks (boilerplate)

_SPHERE_CACHE = {}

def unit_sphere(n_u: int, n_v: int):
    """Vertices and triangle indices of a unit UV sphere (cached per resolution)."""
    if (n_u, n_v) not in _SPHERE_CACHE:
        u = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
        v = np.linspace(0, np.pi, n_v)
        uu, vv = np.meshgrid(u, v)
        verts = np.stack([(np.cos(uu) * np.sin(vv)).ravel(),
                          (np.sin(uu) * np.sin(vv)).ravel(),
                          np.cos(vv).ravel()], axis=1)
        r, c = np.meshgrid(np.arange(n_v - 1), np.arange(n_u), indexing="ij")
        r, c = r.ravel(), c.ravel()
        a = r * n_u + c
        b = r * n_u + (c + 1) % n_u
        d = (r + 1) * n_u + c
        e = (r + 1) * n_u + (c + 1) % n_u
        faces = np.concatenate([np.stack([a, d, b], axis=1), np.stack([b, e, d], axis=1)])
        _SPHERE_CACHE[(n_u, n_v)] = (verts, faces)
    return _SPHERE_CACHE[(n_u, n_v)]

def sphere_trace(nodes, pos, radii, color, name, resolution):
    """Merge many spheres into a single Mesh3d trace."""
    verts, faces = unit_sphere(*resolution)
    all_v = np.concatenate([verts * radii[n] + pos[n] for n in nodes])
    all_f = np.concatenate([faces + i * len(verts) for i in range(len(nodes))])
    return go.Mesh3d(
        x=all_v[:, 0], y=all_v[:, 1], z=all_v[:, 2],
        i=all_f[:, 0], j=all_f[:, 1], k=all_f[:, 2],
        color=color, flatshading=False, name=name, hoverinfo="skip",
        lighting=dict(ambient=0.45, diffuse=0.9, specular=0.6, roughness=0.3),
        lightposition=dict(x=100, y=200, z=300),
    )

def build_document_graph(sources: Dict[str, List[Document]]):
    """Fill DOCUMENT_GRAPH from {filename: chunks} and draw it as 3D spheres.

    Returns (figure, stats). Node types: doc (one per file), group (a run of
    consecutive chunks), topic (a keyword shared by many groups).
    """
    DOCUMENT_GRAPH.clear()
    total = sum(len(docs) for docs in sources.values())
    terms = {name: [chunk_terms(d.page_content) for d in docs] for name, docs in sources.items()}

    # Topic spheres: widest-spread non-generic keywords, count scaled to data volume
    doc_freq = Counter(t for per_file in terms.values() for s in per_file for t in s)
    ratio = GENERIC_TERM_RATIO if total >= 100 else 0.5
    min_df = 2 if total >= 20 else 1
    group_size = max(CHUNKS_PER_NODE, -(-total // MAX_GROUP_NODES))
    n_groups = sum(-(-len(docs) // group_size) for docs in sources.values())
    n_topics = min(MAX_TOPICS, max(MIN_TOPICS, n_groups // GROUPS_PER_TOPIC))
    topics = [t for t, c in doc_freq.most_common() if min_df <= c <= ratio * total][:n_topics]
    topic_set = set(topics)
    for t in topics:
        DOCUMENT_GRAPH.add_node(t, node_type="topic", info=f"Topic: {t}")

    for name, docs in sources.items():
        doc_node = f"DOC: {name}"
        DOCUMENT_GRAPH.add_node(doc_node, node_type="doc", info=f"{name} ({len(docs)} chunks)")
        file_topics = Counter()
        previous = doc_node
        for g, start in enumerate(range(0, len(docs), group_size)):
            end = min(start + group_size, len(docs))
            counts = Counter(t for s in terms[name][start:end] for t in s if t in topic_set)
            linked = [t for t, _ in counts.most_common(LINKS_PER_GROUP)]
            group_node = f"{name[:18]} #{g}"
            DOCUMENT_GRAPH.add_node(group_node, node_type="group", size=end - start,
                                    info=f"{name} chunks {start}-{end - 1}<br>{', '.join(linked) or 'no shared topics'}")
            DOCUMENT_GRAPH.add_edge(previous, group_node, relation="NEXT")
            previous = group_node
            for t in linked:
                DOCUMENT_GRAPH.add_edge(group_node, t, relation="MENTIONS")
            file_topics.update(counts)
        for t, _ in file_topics.most_common(5):
            DOCUMENT_GRAPH.add_edge(doc_node, t, relation="HAS_TOPIC")

    n_nodes = DOCUMENT_GRAPH.number_of_nodes()
    pos = nx.spring_layout(DOCUMENT_GRAPH, dim=3, seed=42, iterations=100 if n_nodes <= 300 else 50)
    pos = {n: np.array(p) for n, p in pos.items()}

    degree = dict(DOCUMENT_GRAPH.degree())
    max_deg = max(degree.values()) or 1
    radii = {}
    for n, data in DOCUMENT_GRAPH.nodes(data=True):
        if data["node_type"] == "doc":
            radii[n] = 0.12
        elif data["node_type"] == "topic":
            radii[n] = 0.035 + 0.06 * degree[n] / max_deg
        else:
            radii[n] = 0.02 + 0.03 * data["size"] / group_size

    resolution = (16, 10) if n_nodes <= 150 else (10, 7) if n_nodes <= 400 else (8, 6)

    ex, ey, ez = [], [], []
    for a, b in DOCUMENT_GRAPH.edges():
        ex += [pos[a][0], pos[b][0], None]
        ey += [pos[a][1], pos[b][1], None]
        ez += [pos[a][2], pos[b][2], None]
    traces = [go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", hoverinfo="skip",
                           line=dict(color="#90A4AE", width=2), name="edges")]

    groups = {"doc": ("#FF5722", "Document"), "topic": ("#2196F3", "Topics"), "group": ("#4CAF50", "Chunk groups")}
    for node_type, (color, label) in groups.items():
        nodes = [n for n, d in DOCUMENT_GRAPH.nodes(data=True) if d["node_type"] == node_type]
        if not nodes:
            continue
        traces.append(sphere_trace(nodes, pos, radii, color, label, resolution))
        labelled = node_type != "group"
        traces.append(go.Scatter3d(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], z=[pos[n][2] for n in nodes],
            mode="text" if labelled else "markers",
            text=[n.replace("DOC: ", "") for n in nodes] if labelled else None,
            textposition="top center", textfont=dict(size=10),
            marker=dict(size=6, opacity=0.02, color=color),
            hovertext=[f"{DOCUMENT_GRAPH.nodes[n]['info']}<br>links: {degree[n]}" for n in nodes],
            hoverinfo="text", name=label, showlegend=False,
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Knowledge Graph: {n_nodes} spheres for {total:,} chunks (drag to rotate, scroll to zoom)",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0), height=560, showlegend=False,
    )
    stats = {"nodes": n_nodes, "edges": DOCUMENT_GRAPH.number_of_edges(), "groups": n_groups,
             "topics": len(topics), "chunks_per_group": group_size}
    return fig, stats

_EMBEDDINGS = None

def get_embeddings():
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"batch_size": 64},
        )
    return _EMBEDDINGS

EMBED_BATCH = 256

def process_files_and_build_graph(files, progress=gr.Progress()):
    global VECTOR_STORE
    if not files:
        return "No file uploaded.", None
    if not isinstance(files, list):
        files = [files]

    splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    sources, warnings = {}, []
    for n, f in enumerate(files):
        path = f if isinstance(f, str) else f.name
        filename = os.path.basename(path)
        progress(n / len(files), desc=f"Reading {filename}")
        text_content = extract_text_from_file(path)
        if not text_content.strip():
            warnings.append(f"⚠️ Could not extract readable text from {filename}.")
            continue
        chunks = splitter.split_text(text_content)
        sources[filename] = [
            Document(page_content=chunk, metadata={"source": filename, "chunk_id": i})
            for i, chunk in enumerate(chunks)
        ]

    if not sources:
        return "\n\n".join(warnings), None

    # Fresh Chroma collection; embed in batches so large uploads report progress
    if VECTOR_STORE is not None:
        VECTOR_STORE.delete_collection()
    VECTOR_STORE = Chroma(collection_name=f"kb_{uuid.uuid4().hex[:12]}", embedding_function=get_embeddings())
    all_docs = [d for docs in sources.values() for d in docs]
    for start in range(0, len(all_docs), EMBED_BATCH):
        progress(start / len(all_docs), desc=f"Embedding chunks {start:,}/{len(all_docs):,}")
        VECTOR_STORE.add_documents(all_docs[start:start + EMBED_BATCH])

    progress(1.0, desc="Rendering 3D graph")
    fig, stats = build_document_graph(sources)

    status = (f"✅ Processed **{len(sources)}** file(s): **{len(all_docs):,}** chunks → "
              f"**{stats['nodes']}** spheres ({stats['groups']} chunk groups of ~{stats['chunks_per_group']} chunks, "
              f"{stats['topics']} topics) and {stats['edges']} edges.")
    if warnings:
        status += "\n\n" + "\n\n".join(warnings)
    return status, fig

# ---------------------------------------------------------------------------
# 6. UI Handlers
# ---------------------------------------------------------------------------
thread_id = "rag_session_1"

def handle_user_message(api_key, model_choice, user_msg, history):
    history = history or []

    if not user_msg or not user_msg.strip():
        return "", history, gr.update(visible=False)

    if not api_key or not api_key.strip():
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": "⚠️ Please enter your Groq API Key in the left sidebar."})
        return "", history, gr.update(visible=False)

    config = {"configurable": {"thread_id": thread_id}}

    try:
        events = compiled_app.stream(
            {
                "messages": [HumanMessage(content=user_msg)],
                "api_key": api_key,
                "model_name": model_choice
            },
            config,
            stream_mode="values"
        )

        bot_msg = "No response generated."
        requires_human = False

        for event in events:
            if "messages" in event and event["messages"]:
                bot_msg = event["messages"][-1].content
            requires_human = event.get("requires_human", False)

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": bot_msg})

        return "", history, gr.update(visible=requires_human)

    except Exception as error:
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": f"❌ Execution Failure: {str(error)}"})
        return "", history, gr.update(visible=False)

def handle_human_approval(api_key, model_choice, action, notes, history):
    history = history or []
    config = {"configurable": {"thread_id": thread_id}}
    resolution_msg = f"🛡️ **Supervisor Action**: {action} | Notes: {notes}"

    try:
        compiled_app.update_state(
            config,
            {
                "requires_human": False,
                "messages": [AIMessage(content=resolution_msg)],
                "api_key": api_key,
                "model_name": model_choice
            },
            as_node="human_review_node"
        )

        events = compiled_app.stream(None, config, stream_mode="values")
        bot_msg = ""
        for event in events:
            if "messages" in event and event["messages"]:
                bot_msg = event["messages"][-1].content

        history.append({"role": "user", "content": f"Directive Submitted: {action}"})
        history.append({"role": "assistant", "content": bot_msg})

        return history, gr.update(visible=False)
    except Exception as e:
        history.append({"role": "assistant", "content": f"❌ Resolution error: {str(e)}"})
        return history, gr.update(visible=False)

# ---------------------------------------------------------------------------
# 7. Gradio Application Interface
# ---------------------------------------------------------------------------
with gr.Blocks(title="LangGraph Knowledge-Graph RAG Agent") as demo:
    gr.Markdown("# 🕸️ LangGraph Knowledge-Graph RAG Agent")

    with gr.Row():
        with gr.Column(scale=1):
            groq_key = gr.Textbox(label="Groq API Key", placeholder="gsk_...", type="password")
            model_select = gr.Dropdown(
                choices=[
                    "llama-3.3-70b-versatile",
                    "qwen-2.5-32b",
                    "mixtral-8x7b-32768",
                    "openai/gpt-oss-120b"
                ],
                value="llama-3.3-70b-versatile",
                label="Groq Model Engine"
            )

            gr.Markdown("### 📄 Knowledge Base Ingestion")
            kb_file = gr.File(
                label="Upload Documents (.txt, .pdf, .docx, .xlsx) - multiple large files supported",
                file_types=[".txt", ".pdf", ".docx", ".xlsx", ".xls"],
                file_count="multiple"
            )
            file_status = gr.Markdown("No document ingested yet.")
            graph_visual = gr.Plot(label="Extracted Keyword & Title Graph")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="RAG Assistant Window")
            msg_input = gr.Textbox(label="Query Input", placeholder="Ask a question about the uploaded knowledge base...")
            send_btn = gr.Button("Submit Question", variant="primary")

            with gr.Group(visible=False) as human_panel:
                gr.Markdown("### 🛡️ Human Intervention Approval Desk")
                action_choice = gr.Radio(["Approve Request", "Reject & Redirect"], value="Approve Request", label="Decision")
                admin_notes = gr.Textbox(label="Notes", placeholder="Reason for approval/denial...")
                human_submit = gr.Button("Resume Execution Flow", variant="stop")

    # Wire Trigger Events
    kb_file.change(
        fn=process_files_and_build_graph,
        inputs=[kb_file],
        outputs=[file_status, graph_visual]
    )

    send_btn.click(
        fn=handle_user_message,
        inputs=[groq_key, model_select, msg_input, chatbot],
        outputs=[msg_input, chatbot, human_panel]
    )

    human_submit.click(
        fn=handle_human_approval,
        inputs=[groq_key, model_select, action_choice, admin_notes, chatbot],
        outputs=[chatbot, human_panel]
    )

demo.launch(theme=gr.themes.Soft(), share=True, debug=True, max_file_size="1gb")
