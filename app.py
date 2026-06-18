"""
app.py

Personal Knowledge AI Assistant

功能：
1. 上傳文件
2. 建立知識庫
3. RAG檢索
4. Ollama回答
5. 顯示引用來源

Author: Team RAG
"""

import os
import html
import json
import streamlit as st
import streamlit.components.v1 as components

from parser import DocumentParser
from chunker import TextChunker
from embedding import EmbeddingService
from vector_store import VectorStore
from retriever import Retriever
from prompt_builder import PromptBuilder
from llm_service import LLMService


UPLOAD_DIR = "uploads"
LLM_MODEL_NAME = "gemma3"

os.makedirs(UPLOAD_DIR, exist_ok=True)


def extract_markdown_code(text):
    """
    從 LLM 回答中取出 Markdown 內容
    """

    for fence in ("```markdown", "```md", "```"):

        if fence in text:
            return (
                text
                .split(fence, 1)[1]
                .split("```", 1)[0]
                .strip()
            )

    return text.strip()


def clean_markmap_markdown(markdown, source):
    """
    清理 LLM 常見的 Markdown 輸出格式問題
    """

    markdown = html.unescape(markdown)
    markdown = markdown.replace("\r\n", "\n")
    markdown = markdown.replace("```markdown", "")
    markdown = markdown.replace("```md", "")
    markdown = markdown.replace("```", "")

    lines = []

    for line in markdown.splitlines():
        line = line.rstrip()

        if not line:
            continue

        if line.strip().lower() in ("markdown", "markmap"):
            continue

        lines.append(line)

    if not lines:
        return f"# {source}"

    if not lines[0].lstrip().startswith("#"):
        lines.insert(0, f"# {source}")

    return "\n".join(lines)


def limit_markmap_tree(
    node,
    max_depth=4,
    max_children_by_depth=None,
    depth=0
):
    """
    限制心智圖節點數量，避免畫面過度切碎
    """

    if max_children_by_depth is None:
        max_children_by_depth = {
            0: 20,
            1: 10,
            2: 8
        }

    if depth >= max_depth - 1:
        node["children"] = []
        return node

    children = node.get("children", [])
    max_children = max_children_by_depth.get(
        depth,
        2
    )
    node["children"] = children[:max_children]

    for child in node["children"]:
        limit_markmap_tree(
            child,
            max_depth=max_depth,
            max_children_by_depth=max_children_by_depth,
            depth=depth + 1
        )

    return node


def build_markmap_tree(markdown):
    """
    將 Markmap Markdown 轉成可互動 SVG 使用的樹狀資料
    """

    root = None
    stack = []
    current_heading_level = 1

    for raw_line in markdown.splitlines():

        if not raw_line.strip():
            continue

        expanded_line = raw_line.replace("\t", "  ")
        stripped = expanded_line.strip()

        if stripped.startswith("#"):

            hashes = len(stripped) - len(
                stripped.lstrip("#")
            )

            label = stripped[hashes:].strip()
            level = max(1, hashes)
            current_heading_level = level

        elif (
            stripped.startswith("- ")
            or stripped.startswith("* ")
        ):

            indent = len(expanded_line) - len(
                expanded_line.lstrip(" ")
            )

            label = stripped[2:].strip()
            level = (
                current_heading_level +
                1 +
                indent // 2
            )

        else:
            continue

        if not label:
            continue

        node = {
            "label": label,
            "children": []
        }

        if root is None:
            root = node
            stack = [(level, node)]
            continue

        while stack and stack[-1][0] >= level:
            stack.pop()

        parent = stack[-1][1] if stack else root
        parent["children"].append(node)
        stack.append((level, node))

    if root is None:
        root = {
            "label": "心智圖",
            "children": []
        }

    return limit_markmap_tree(root)


def render_markmap(markdown):
    """
    在 Streamlit 中顯示 Markmap 心智圖
    """

    tree_json = json.dumps(
        build_markmap_tree(markdown),
        ensure_ascii=False
    )

    components.html(
        f"""
        <style>
            html,
            body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
            }}

            #mindmap-stage {{
                width: 100%;
                height: 760px;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                background: #ffffff;
                overflow: hidden;
                cursor: grab;
            }}

            #mindmap-stage:active {{
                cursor: grabbing;
            }}

            #mindmap {{
                width: 100%;
                height: 100%;
            }}

            .link {{
                fill: none;
                stroke: #cbd5e1;
                stroke-width: 2;
            }}

            .node rect {{
                fill: #ffffff;
                stroke: #64748b;
                stroke-width: 1.4;
                rx: 8;
            }}

            .node.depth-0 rect {{
                fill: #0f172a;
                stroke: #0f172a;
            }}

            .node text {{
                fill: #0f172a;
                font-family: "Microsoft JhengHei", system-ui, sans-serif;
                font-size: 14px;
                dominant-baseline: middle;
                pointer-events: none;
            }}

            .node.depth-0 text {{
                fill: #ffffff;
                font-weight: 700;
                font-size: 16px;
            }}

            .collapse-dot {{
                fill: #2563eb;
                stroke: #ffffff;
                stroke-width: 2;
                cursor: pointer;
            }}
        </style>

        <div id="mindmap-stage">
            <svg id="mindmap"></svg>
        </div>
        <script>
            const tree = {tree_json};
            const svg = document.querySelector("#mindmap");
            const namespace = "http://www.w3.org/2000/svg";
            let scale = 1;
            let offsetX = 40;
            let offsetY = 40;
            let dragging = false;
            let startX = 0;
            let startY = 0;
            let startOffsetX = 0;
            let startOffsetY = 0;

            const group = document.createElementNS(namespace, "g");
            svg.appendChild(group);

            function createSvg(tag, attrs = {{}}) {{
                const element = document.createElementNS(namespace, tag);

                Object.entries(attrs).forEach(([key, value]) => {{
                    element.setAttribute(key, value);
                }});

                return element;
            }}

            function wrapText(text, maxChars = 18) {{
                const words = String(text).split("");
                const lines = [];
                let current = "";

                words.forEach((char) => {{
                    current += char;

                    if (current.length >= maxChars) {{
                        lines.push(current);
                        current = "";
                    }}
                }});

                if (current) lines.push(current);

                return lines.slice(0, 4);
            }}

            function collectVisible(node, depth = 0, rows = []) {{
                rows.push({{ node, depth }});

                if (!node.collapsed) {{
                    (node.children || []).forEach((child) => {{
                        collectVisible(child, depth + 1, rows);
                    }});
                }}

                return rows;
            }}

            function layout(root) {{
                const rows = collectVisible(root);

                rows.forEach((item, index) => {{
                    item.node.x = item.depth * 300;
                    item.node.y = index * 86;
                    item.node.depth = item.depth;
                }});
            }}

            function drawLinks(node) {{
                if (node.collapsed) return;

                (node.children || []).forEach((child) => {{
                    const path = createSvg("path", {{
                        class: "link",
                        d: `
                            M ${{node.x + 210}} ${{node.y + 28}}
                            C ${{node.x + 255}} ${{node.y + 28}},
                              ${{child.x - 45}} ${{child.y + 28}},
                              ${{child.x}} ${{child.y + 28}}
                        `
                    }});

                    group.appendChild(path);
                    drawLinks(child);
                }});
            }}

            function drawNode(node) {{
                const labelLines = wrapText(node.label);
                const width = node.depth === 0 ? 240 : 230;
                const height = Math.max(56, 26 + labelLines.length * 18);

                const nodeGroup = createSvg("g", {{
                    class: `node depth-${{node.depth}}`,
                    transform: `translate(${{node.x}}, ${{node.y}})`
                }});

                const rect = createSvg("rect", {{
                    width,
                    height
                }});

                nodeGroup.appendChild(rect);

                labelLines.forEach((line, index) => {{
                    const text = createSvg("text", {{
                        x: 14,
                        y: height / 2 + (index - (labelLines.length - 1) / 2) * 18
                    }});

                    text.textContent = line;
                    nodeGroup.appendChild(text);
                }});

                if ((node.children || []).length) {{
                    const dot = createSvg("circle", {{
                        class: "collapse-dot",
                        cx: width - 14,
                        cy: height / 2,
                        r: 7
                    }});

                    dot.addEventListener("click", (event) => {{
                        event.stopPropagation();
                        node.collapsed = !node.collapsed;
                        render();
                    }});

                    nodeGroup.appendChild(dot);
                }}

                group.appendChild(nodeGroup);

                if (!node.collapsed) {{
                    (node.children || []).forEach(drawNode);
                }}
            }}

            function render() {{
                group.innerHTML = "";
                layout(tree);
                drawLinks(tree);
                drawNode(tree);
                group.setAttribute(
                    "transform",
                    `translate(${{offsetX}}, ${{offsetY}}) scale(${{scale}})`
                );
            }}

            svg.addEventListener("wheel", (event) => {{
                event.preventDefault();
                const direction = event.deltaY > 0 ? -1 : 1;
                const factor = direction > 0 ? 1.08 : 0.92;
                scale = Math.min(2.6, Math.max(0.25, scale * factor));
                render();
            }}, {{ passive: false }});

            svg.addEventListener("pointerdown", (event) => {{
                dragging = true;
                startX = event.clientX;
                startY = event.clientY;
                startOffsetX = offsetX;
                startOffsetY = offsetY;
            }});

            svg.addEventListener("pointermove", (event) => {{
                if (!dragging) return;
                offsetX = startOffsetX + event.clientX - startX;
                offsetY = startOffsetY + event.clientY - startY;
                render();
            }});

            window.addEventListener("pointerup", () => {{
                dragging = false;
            }});

            render();
        </script>
        """,
        height=790,
        scrolling=True
    )


def build_mindmap_prompt(source, source_text):
    """
    建立心智圖 Prompt
    """

    return f"""
你是一位擅長整理文件重點的知識管理助理。

請根據下方文件內容，為檔案「{source}」製作一張心智圖。

要求：
1. 只輸出 Markmap 可讀取的 Markdown，不要輸出解釋文字。
2. 第一行必須是 # {source}。
3. 使用 Markdown 標題與項目符號建立階層。
4. 使用繁體中文整理內容。
5. 先閱讀整份內容，再自行判斷最適合的主要主題，不要照頁面順序機械切割。
6. 建議 4 到 20 個主要主題；如果文件真的很複雜，最多 20 個。
7. 每個主要主題若有小主題、方法、概念、用途、限制或結論，請加入 2 到 10 個子節點。
8. 子節點下面若有必要的關鍵重點，可再加入 1 到 8 個第三層節點。
9. 不要為了減少節點而只留一層；有明確小主題或重點時要放入子節點。
10. 不要把細節、例子、程式碼、頁碼、重複定義都做成節點；請合併相近概念。
11. 每個節點文字要短，盡量 20 個中文字以內。
12. 需要涵蓋文件前段、中段、後段的重要內容，不要只整理前半部。
13. 不要加入文件中沒有根據的內容。
14. 不要使用 Mermaid、HTML、表格或程式碼區塊。

請嚴格使用以下格式：
# {source}
## 主題一
- 小主題
  - 關鍵重點
- 小主題
## 主題二
- 小主題
  - 關鍵重點
- 小主題

文件內容：
{source_text}
"""


# ---------------------
# Streamlit Config
# ---------------------

st.set_page_config(
    page_title="Personal Knowledge AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("個人知識庫 AI 助理")

st.markdown("---")


# ---------------------
# Session State
# ---------------------

if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "show_db_files" not in st.session_state:
    st.session_state.show_db_files = False

if "pending_delete_source" not in st.session_state:
    st.session_state.pending_delete_source = None

if "show_mindmap_files" not in st.session_state:
    st.session_state.show_mindmap_files = False

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "build_success_message" not in st.session_state:
    st.session_state.build_success_message = None


# ---------------------
# Sidebar
# ---------------------

with st.sidebar:

    st.header("文件管理")

    uploaded_files = st.file_uploader(
        "上傳文件",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True,
        key=f"uploaded_files_{st.session_state.uploader_key}"
    )

    build_button = st.button(
        "建立資料庫"
    )

    st.markdown("---")

    if st.button("顯示資料庫檔案"):
        st.session_state.show_db_files = True
        st.session_state.pending_delete_source = None

    if st.session_state.show_db_files:

        try:

            store = VectorStore()
            store.load()
            db_files = store.list_sources()

        except FileNotFoundError:

            db_files = []
            st.info("目前尚未建立資料庫")

        except Exception as e:

            db_files = []
            st.error(
                f"讀取資料庫失敗: {str(e)}"
            )

        if db_files:

            selected_source = st.selectbox(
                "選擇要管理的檔案",
                ["請選擇檔案"] + db_files
            )

            selected_source = (
                selected_source
                if selected_source != "請選擇檔案"
                else None
            )

            delete_button = st.button(
                "刪除檔案",
                disabled=selected_source is None
            )

            if delete_button:
                st.session_state.pending_delete_source = selected_source

            if st.session_state.pending_delete_source:

                st.warning(
                    f"確定要從資料庫刪除 "
                    f"{st.session_state.pending_delete_source} 嗎？"
                )

                confirm_delete = st.button(
                    "確認刪除"
                )

                cancel_delete = st.button(
                    "取消"
                )

                if confirm_delete:

                    try:

                        store = VectorStore()
                        store.load()
                        deleted_count = store.delete_source(
                            st.session_state.pending_delete_source
                        )

                        st.session_state.pending_delete_source = None
                        st.session_state.chat_history = []

                        st.success(
                            f"已刪除 {deleted_count} 個 Chunk"
                        )

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"刪除失敗: {str(e)}"
                        )

                if cancel_delete:
                    st.session_state.pending_delete_source = None
                    st.rerun()

        elif "db_files" in locals():

            st.info("資料庫目前沒有檔案")

    st.markdown("---")

    if st.button("繪製檔案心智圖"):
        st.session_state.show_mindmap_files = True

    if st.session_state.show_mindmap_files:

        try:

            store = VectorStore()
            store.load()
            mindmap_files = store.list_sources()

        except FileNotFoundError:

            mindmap_files = []
            st.info("目前尚未建立資料庫")

        except Exception as e:

            mindmap_files = []
            st.error(
                f"讀取資料庫失敗: {str(e)}"
            )

        if mindmap_files:

            selected_mindmap_source = st.selectbox(
                "選擇要繪製心智圖的檔案",
                ["請選擇檔案"] + mindmap_files,
                key="mindmap_source"
            )

            selected_mindmap_source = (
                selected_mindmap_source
                if selected_mindmap_source != "請選擇檔案"
                else None
            )

            draw_mindmap_button = st.button(
                "產生心智圖",
                disabled=selected_mindmap_source is None
            )

            if draw_mindmap_button:

                try:

                    with st.spinner("AI 正在繪製心智圖..."):

                        store = VectorStore()
                        store.load()
                        source_text = store.get_source_text(
                            selected_mindmap_source
                        )

                        if not source_text:
                            st.warning("找不到該檔案的內容")
                        else:
                            prompt = build_mindmap_prompt(
                                selected_mindmap_source,
                                source_text
                            )

                            llm = LLMService(
                                model_name=LLM_MODEL_NAME
                            )

                            mindmap = llm.generate(prompt)

                            st.session_state.chat_history.append(
                                {
                                    "question":
                                    f"請繪製 {selected_mindmap_source} 的心智圖",
                                    "answer": mindmap,
                                    "citation": selected_mindmap_source,
                                    "type": "mindmap"
                                }
                            )

                            st.success("心智圖已加入聊天紀錄")
                            st.rerun()

                except Exception as e:

                    st.error(
                        f"產生心智圖失敗: {str(e)}"
                    )

        elif "mindmap_files" in locals():

            st.info("資料庫目前沒有檔案")


# ---------------------
# 建立知識庫
# ---------------------

if st.session_state.build_success_message:

    st.success(
        st.session_state.build_success_message
    )

    st.session_state.build_success_message = None


if build_button:

    if not uploaded_files:

        st.warning("請先上傳文件")

    else:

        with st.spinner("建立資料庫中..."):

            all_docs = []

            # 1. 儲存檔案
            for file in uploaded_files:

                file_path = os.path.join(
                    UPLOAD_DIR,
                    file.name
                )

                with open(
                    file_path,
                    "wb"
                ) as f:

                    f.write(
                        file.getbuffer()
                    )

                # 2. Parser
                docs = (
                    DocumentParser.parse(
                        file_path
                    )
                )

                all_docs.extend(docs)

            # 3. Chunk
            chunker = TextChunker(
                chunk_size=500,
                chunk_overlap=100
            )

            chunks = (
                chunker.chunk_documents(
                    all_docs
                )
            )

            # 4. Embedding
            embedder = (
                EmbeddingService()
            )

            embedded_chunks = (
                embedder.embed_documents(
                    chunks
                )
            )

            # 5. FAISS
            store = VectorStore()

            try:

                store.load()

            except FileNotFoundError:

                pass

            added_count = store.add_documents(
                embedded_chunks,
                replace_existing=True
            )

            store.save()

            st.session_state.vector_ready = True

        st.session_state.build_success_message = (
            f"資料庫更新完成"
        )

        st.session_state.uploader_key += 1

        st.rerun()


st.markdown("---")


# ---------------------
# 問答區
# ---------------------

st.header("💬 問答")

question = st.text_input(
    "請輸入問題"
)

ask_button = st.button(
    "送出"
)


if ask_button:

    if not question:

        st.warning("請輸入問題")

    else:

        try:

            with st.spinner("AI思考中..."):

                # Embedding
                embedder = (
                    EmbeddingService()
                )

                # Vector DB
                store = VectorStore()

                store.load()

                # Retriever
                retriever = Retriever(
                    vector_store=store,
                    embedding_service=embedder
                )

                results = (
                    retriever.search(
                        question,
                        top_k=30
                    )
                )

                # Prompt
                builder = (
                    PromptBuilder()
                )

                prompt = (
                    builder.build_prompt(
                        question,
                        results
                    )
                )

                # LLM
                llm = LLMService(
                    model_name=LLM_MODEL_NAME
                )

                answer = (
                    llm.generate(
                        prompt
                    )
                )

                citations = (
                    builder.build_citation(
                        results
                    )
                )

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "citation": citations,
                        "type": "qa"
                    }
                )

        except Exception as e:

            st.error(
                f"錯誤: {str(e)}"
            )


# ---------------------
# 顯示聊天紀錄
# ---------------------

if st.session_state.chat_history:

    st.markdown("---")

    mindmap_chats = [
        chat
        for chat in st.session_state.chat_history
        if chat.get("type") == "mindmap"
    ]

    chat_tab, mindmap_tab = st.tabs(
        [
            "聊天紀錄",
            "Markmap 心智圖"
        ]
    )

    with chat_tab:

        st.header("聊天紀錄")

        for chat in reversed(
            st.session_state.chat_history
        ):

            with st.container():

                st.markdown(
                    f"### 🙋 問題\n{chat['question']}"
                )

                if chat.get("type") == "mindmap":

                    st.markdown(
                        "### 🧠 心智圖\n"
                        "已產生 Markmap 心智圖，請切換到「Markmap 心智圖」分頁查看。"
                    )

                else:

                    st.markdown(
                        f"### 🤖 回答\n{chat['answer']}"
                    )

                st.markdown(
                    "### 📚 來源"
                )

                st.code(
                    chat["citation"]
                )

                st.markdown("---")

    with mindmap_tab:

        st.header("Markmap 心智圖")

        if not mindmap_chats:

            st.info("目前尚未產生心智圖")

        else:

            mindmap_options = [
                f"{len(mindmap_chats) - index}. {chat['citation']}"
                for index, chat in enumerate(
                    reversed(mindmap_chats)
                )
            ]

            selected_mindmap_label = st.selectbox(
                "選擇心智圖",
                mindmap_options
            )

            selected_mindmap_index = mindmap_options.index(
                selected_mindmap_label
            )

            selected_mindmap_chat = list(
                reversed(mindmap_chats)
            )[selected_mindmap_index]

            markmap_markdown = extract_markdown_code(
                selected_mindmap_chat["answer"]
            )

            markmap_markdown = clean_markmap_markdown(
                markmap_markdown,
                selected_mindmap_chat["citation"]
            )

            render_markmap(
                markmap_markdown
            )

            with st.expander("Markmap Markdown"):

                st.code(
                    markmap_markdown,
                    language="markdown"
                )
