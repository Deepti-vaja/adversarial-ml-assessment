"""
Executive Presentation Web UI for Antigravity RAG Chatbot (Deliverable M9).

Pure presentation layer built using standard Python HTTP server (ThreadingHTTPServer).
Designed with modular architecture: CSS design tokens, client-side JavaScript bundle,
and HTML layout structure are separated into discrete, maintainable builder functions
rather than monolithic string literals.

Strictly preserves existing backend orchestration by delegating queries directly to
`rag.chatbot_cli.answer_query()`.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List

# Resolve imports cleanly whether executed via `-m rag.gradio_app` or direct script
try:
    from rag.chatbot_cli import answer_query
    from rag.retriever import VectorRetriever
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rag.chatbot_cli import answer_query
    from rag.retriever import VectorRetriever


# =====================================================================
# PRESENTATION ASSETS: MODULAR CSS DESIGN SYSTEM
# =====================================================================

def get_design_system_css() -> str:
    """Return organized CSS styling grouped by design tokens and component layers."""
    return """
    /* --- 1. Design Tokens & Theme Variables --- */
    :root {
        --bg-page: #0f172a;
        --bg-header: rgba(15, 23, 42, 0.85);
        --bg-card: #1e293b;
        --bg-bubble-user: #3b82f6;
        --bg-bubble-bot: #1e293b;
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
        --accent: #38bdf8;
        --border: #334155;
        --shadow: rgba(0, 0, 0, 0.35);
        --code-bg: #0b1120;
        --score-badge: #10b981;
    }
    [data-theme="light"] {
        --bg-page: #f8fafc;
        --bg-header: rgba(248, 250, 252, 0.85);
        --bg-card: #ffffff;
        --bg-bubble-user: #2563eb;
        --bg-bubble-bot: #ffffff;
        --text-main: #0f172a;
        --text-muted: #64748b;
        --accent: #0284c7;
        --border: #e2e8f0;
        --shadow: rgba(0, 0, 0, 0.08);
        --code-bg: #f1f5f9;
        --score-badge: #059669;
    }

    /* --- 2. Core Layout & Reset --- */
    * {
        box-sizing: border-box;
        margin: 0;
        padding: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        transition: background-color 0.25s ease, border-color 0.25s ease, color 0.25s ease;
    }
    body {
        background-color: var(--bg-page);
        color: var(--text-main);
        display: flex;
        flex-direction: column;
        height: 100vh;
        overflow: hidden;
    }
    main {
        display: flex;
        flex: 1;
        overflow: hidden;
    }

    /* --- 3. Header & Navigation Bar --- */
    header {
        background-color: var(--bg-header);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid var(--border);
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        z-index: 10;
    }
    .logo-area h1 { font-size: 1.25rem; font-weight: 700; color: var(--text-main); }
    .logo-area p { font-size: 0.8rem; color: var(--text-muted); margin-top: 0.15rem; }
    .header-actions { display: flex; gap: 0.75rem; }
    .btn {
        background: var(--bg-card);
        color: var(--text-main);
        border: 1px solid var(--border);
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        cursor: pointer;
        font-size: 0.85rem;
        font-weight: 500;
        box-shadow: 0 2px 4px var(--shadow);
    }
    .btn:hover { border-color: var(--accent); color: var(--accent); }

    /* --- 4. Chat Interface & Bubbles --- */
    .chat-section {
        flex: 3;
        display: flex;
        flex-direction: column;
        border-right: 1px solid var(--border);
        height: 100%;
    }
    .chat-history {
        flex: 1;
        overflow-y: auto;
        padding: 1.5rem 2rem;
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
    }
    .message { display: flex; flex-direction: column; max-width: 80%; animation: fadeIn 0.3s ease; }
    .message.user { align-self: flex-end; }
    .message.bot { align-self: flex-start; width: 85%; max-width: 85%; }
    .bubble {
        padding: 1rem 1.25rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px var(--shadow);
        line-height: 1.6;
        font-size: 0.95rem;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .message.user .bubble { background-color: var(--bg-bubble-user); color: #ffffff; }
    .message.bot .bubble { background-color: var(--bg-bubble-bot); border: 1px solid var(--border); }
    .bot-actions { display: flex; gap: 0.5rem; margin-top: 0.4rem; }
    .btn-copy {
        background: transparent; border: none; color: var(--text-muted);
        font-size: 0.75rem; cursor: pointer; padding: 0.2rem 0.5rem; border-radius: 0.3rem;
    }
    .btn-copy:hover { color: var(--accent); background: rgba(56, 189, 248, 0.1); }

    /* --- 5. Controls & Prompt Input Panel --- */
    .input-panel {
        padding: 1rem 2rem;
        background-color: var(--bg-header);
        border-top: 1px solid var(--border);
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }
    .sample-prompts { display: flex; flex-wrap: wrap; gap: 0.5rem; }
    .chip {
        background: var(--bg-card); border: 1px solid var(--border); color: var(--text-muted);
        padding: 0.3rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; cursor: pointer;
        max-width: 260px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .chip:hover { border-color: var(--accent); color: var(--text-main); }
    .input-box { display: flex; gap: 0.75rem; }
    textarea {
        flex: 1; background-color: var(--bg-card); color: var(--text-main);
        border: 1px solid var(--border); border-radius: 0.5rem; padding: 0.75rem 1rem;
        font-size: 0.95rem; resize: none; height: 52px; outline: none;
    }
    textarea:focus { border-color: var(--accent); }
    .btn-send {
        background-color: var(--accent); color: #ffffff; border: none;
        padding: 0 1.5rem; border-radius: 0.5rem; font-weight: 600; cursor: pointer;
    }
    .btn-send:hover { opacity: 0.9; }
    .btn-send:disabled { opacity: 0.5; cursor: not-allowed; }

    /* --- 6. Retrieved Evidence Panel --- */
    .evidence-section {
        flex: 2; background-color: var(--bg-page); padding: 1.5rem;
        overflow-y: auto; display: flex; flex-direction: column; gap: 1rem;
    }
    .evidence-header h2 { font-size: 1rem; font-weight: 600; color: var(--text-main); }
    .evidence-header p { font-size: 0.8rem; color: var(--text-muted); margin-top: 0.2rem; }
    .evidence-card {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 0.5rem; overflow: hidden;
    }
    .evidence-card summary {
        padding: 0.75rem 1rem; font-size: 0.85rem; font-weight: 600;
        cursor: pointer; list-style: none; display: flex; justify-content: space-between;
        background: rgba(0,0,0,0.1);
    }
    .evidence-card summary::-webkit-details-marker { display: none; }
    .score-badge {
        background-color: rgba(16, 185, 129, 0.15); color: var(--score-badge);
        padding: 0.15rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; font-weight: 700;
    }
    .evidence-body {
        padding: 1rem; font-size: 0.8rem; color: var(--text-muted); line-height: 1.5;
        border-top: 1px solid var(--border); background: var(--code-bg); white-space: pre-wrap;
        max-height: 300px; overflow-y: auto;
    }
    .meta-info { display: flex; gap: 1rem; margin-bottom: 0.5rem; font-size: 0.75rem; color: var(--accent); font-weight: 600; }

    /* --- 7. Animations & Loading Indicators --- */
    .loading { display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem; color: var(--text-muted); padding: 0.5rem 0; }
    .spinner {
        width: 16px; height: 16px; border: 2px solid var(--border);
        border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    """


# =====================================================================
# PRESENTATION ASSETS: MODULAR CLIENT JAVASCRIPT BUNDLE
# =====================================================================

def get_client_javascript() -> str:
    """Return structured client-side JavaScript grouped logically by functionality."""
    return """
    // --- 1. Configuration & Constants ---
    const SAMPLE_QUESTIONS = [
        "What clean test accuracy and validation loss did the baseline FraudCNN model achieve?",
        "How did PGD attack perturbation affect accuracy compared to the clean baseline evaluation?",
        "What accuracy trade-offs did adversarial training and feature squeezing demonstrate?",
        "What did PCA decision region analysis reveal regarding adversarial feature space perturbations?",
        "Compare the training overhead and robustness differences between baseline and defense models."
    ];

    // --- 2. DOM Elements & Initialization ---
    const chatHistory = document.getElementById("chatHistory");
    const evidenceList = document.getElementById("evidenceList");
    const queryInput = document.getElementById("queryInput");
    const btnSend = document.getElementById("btnSend");
    const btnClear = document.getElementById("btnClear");
    const btnTheme = document.getElementById("btnTheme");
    const samplePrompts = document.getElementById("samplePrompts");

    SAMPLE_QUESTIONS.forEach(q => {
        const chip = document.createElement("button");
        chip.className = "chip";
        chip.textContent = q;
        chip.title = q;
        chip.onclick = () => { queryInput.value = q; sendQuery(); };
        samplePrompts.appendChild(chip);
    });

    // --- 3. Event Listeners ---
    btnTheme.onclick = () => {
        const body = document.body;
        const next = body.getAttribute("data-theme") === "dark" ? "light" : "dark";
        body.setAttribute("data-theme", next);
        btnTheme.textContent = next === "dark" ? "🌙 Theme" : "☀️ Theme";
    };

    btnClear.onclick = () => {
        chatHistory.innerHTML = `
            <div class="message bot">
                <div class="bubble">Welcome to the Antigravity Executive RAG Assistant. Ask any analytical question regarding our adversarial training, PGD attack evaluations, boundary analysis, or model benchmarks.</div>
            </div>`;
        evidenceList.innerHTML = `<p style="font-size: 0.8rem; color: var(--text-muted);">No evidence retrieved yet.</p>`;
    };

    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuery(); }
    });

    btnSend.onclick = sendQuery;

    // --- 4. DOM Rendering Helpers ---
    function appendMessage(role, text) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${role}`;
        const bubbleDiv = document.createElement("div");
        bubbleDiv.className = "bubble";
        bubbleDiv.textContent = text;
        msgDiv.appendChild(bubbleDiv);

        if (role === "bot") {
            const actionsDiv = document.createElement("div");
            actionsDiv.className = "bot-actions";
            const copyBtn = document.createElement("button");
            copyBtn.className = "btn-copy";
            copyBtn.textContent = "📋 Copy Response";
            copyBtn.onclick = () => {
                navigator.clipboard.writeText(text);
                copyBtn.textContent = "✅ Copied!";
                setTimeout(() => { copyBtn.textContent = "📋 Copy Response"; }, 2000);
            };
            actionsDiv.appendChild(copyBtn);
            msgDiv.appendChild(actionsDiv);
        }
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return msgDiv;
    }

    function renderEvidence(chunks) {
        if (!chunks || chunks.length === 0) {
            evidenceList.innerHTML = `<p style="font-size: 0.8rem; color: var(--text-muted);">No evidence returned.</p>`;
            return;
        }
        evidenceList.innerHTML = "";
        chunks.forEach((chunk, idx) => {
            const meta = chunk.metadata || {};
            const docName = meta.doc || "Unknown Doc";
            const secName = meta.section || "Unknown Section";
            const score = chunk.score !== undefined ? chunk.score : "N/A";

            const card = document.createElement("details");
            card.className = "evidence-card";
            if (idx === 0) card.open = true;

            card.innerHTML = `
                <summary>
                    <span>[Rank #${idx + 1}] ${docName}</span>
                    <span class="score-badge">Retrieval Score: ${score}</span>
                </summary>
                <div class="evidence-body">
                    <div class="meta-info"><span>File: ${docName}</span><span>Section: ${secName}</span></div>
                    <div>${chunk.text}</div>
                </div>`;
            evidenceList.appendChild(card);
        });
    }

    // --- 5. Async API Fetch Client ---
    async function sendQuery() {
        const query = queryInput.value.trim();
        if (!query) return;

        queryInput.value = "";
        appendMessage("user", query);
        btnSend.disabled = true;
        queryInput.disabled = true;

        const loader = document.createElement("div");
        loader.className = "loading";
        loader.innerHTML = `<div class="spinner"></div><span>Synthesizing response via RAG backend...</span>`;
        chatHistory.appendChild(loader);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 60000);

            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: query }),
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            const data = await res.json();
            chatHistory.removeChild(loader);

            if (res.ok) {
                appendMessage("bot", data.answer || "No response generated.");
                renderEvidence(data.retrieved_chunks || []);
            } else {
                appendMessage("bot", `Error (${res.status}): ${data.error || 'Server error'}`);
            }
        } catch (err) {
            if (chatHistory.contains(loader)) chatHistory.removeChild(loader);
            const msg = err.name === "AbortError" ? "Request timed out (60s)." : `Network error: ${err.message}`;
            appendMessage("bot", msg);
        } finally {
            btnSend.disabled = false;
            queryInput.disabled = false;
            queryInput.focus();
        }
    }
    """


# =====================================================================
# PRESENTATION ASSETS: MODULAR HTML BUILDERS
# =====================================================================

def render_header_html() -> str:
    """Build semantic HTML header with title and control actions."""
    return """
    <header>
        <div class="logo-area">
            <h1>🛡️ Antigravity RAG Executive Assistant</h1>
            <p>Adversarial ML Assessment & Knowledge Retrieval Engine</p>
        </div>
        <div class="header-actions">
            <button class="btn" id="btnClear" title="Clear Conversation">🧹 Clear</button>
            <button class="btn" id="btnTheme" title="Toggle Dark/Light Mode">🌙 Theme</button>
        </div>
    </header>
    """


def render_chat_section_html() -> str:
    """Build main chat transcript area and query input controls."""
    return """
    <section class="chat-section">
        <div class="chat-history" id="chatHistory">
            <div class="message bot">
                <div class="bubble">Welcome to the Antigravity Executive RAG Assistant. Ask any analytical question regarding our adversarial training, PGD attack evaluations, boundary analysis, or model benchmarks.</div>
            </div>
        </div>
        <div class="input-panel">
            <div class="sample-prompts" id="samplePrompts"></div>
            <div class="input-box">
                <textarea id="queryInput" placeholder="Type your query or select a benchmark prompt above... (Press Enter to send)" rows="1"></textarea>
                <button class="btn-send" id="btnSend">Send</button>
            </div>
        </div>
    </section>
    """


def render_evidence_aside_html() -> str:
    """Build collapsible retrieved document inspection panel."""
    return """
    <aside class="evidence-section">
        <div class="evidence-header">
            <h2>📑 Retrieved Evidence Panel</h2>
            <p>Real-time semantic chunks & strict similarity metrics</p>
        </div>
        <div id="evidenceList">
            <p style="font-size: 0.8rem; color: var(--text-muted);">No evidence retrieved yet. Submit a query to inspect source chunks and exact retrieval scores.</p>
        </div>
    </aside>
    """


def render_complete_html_page() -> str:
    """Assemble complete single-page application HTML from modular builders."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Antigravity Executive RAG Assistant</title>
    <style>{get_design_system_css()}</style>
</head>
<body data-theme="dark">
    {render_header_html()}
    <main>
        {render_chat_section_html()}
        {render_evidence_aside_html()}
    </main>
    <script>{get_client_javascript()}</script>
</body>
</html>
"""


# =====================================================================
# CLEAN ROUTING HTTP SERVER HANDLER
# =====================================================================

class ExecutiveUIHandler(BaseHTTPRequestHandler):
    """Clean REST router forwarding presentation requests and API invocations."""

    def log_message(self, format: str, *args: Any) -> None:
        """Format server logs cleanly to avoid polluting benchmark output."""
        print(f"[UI Server] {self.client_address[0]} - {format % args}")

    def do_GET(self) -> None:
        """Clean routing table for GET endpoints."""
        if self.path in ["/", "/index.html"]:
            self.serve_html_page()
        else:
            self.serve_json_error(404, "Endpoint not found")

    def do_POST(self) -> None:
        """Clean routing table for POST API endpoints."""
        if self.path == "/api/chat":
            self.handle_api_chat()
        else:
            self.serve_json_error(404, "Endpoint not found")

    def serve_html_page(self) -> None:
        """Send generated HTML single-page application payload."""
        page_content = render_complete_html_page()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(page_content.encode("utf-8"))

    def handle_api_chat(self) -> None:
        """Parse incoming query payload and forward directly to backend CLI entry point."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            query = payload.get("query", "").strip()

            if not query:
                return self.serve_json_error(400, "Empty query text")

            # Forward query and singleton retriever directly to normal CLI public function
            result = answer_query(
                query=query,
                config_path="configs/rag.yaml",
                retriever=self.server.retriever
            )

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        except Exception as e:
            self.serve_json_error(500, str(e))

    def serve_json_error(self, status_code: int, message: str) -> None:
        """Send formatted JSON error message with appropriate HTTP status code."""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message, "status": status_code}).encode("utf-8"))


def run_server(port: int = 7860, config_path: str = "configs/rag.yaml") -> None:
    """Initialize singleton retriever and start ThreadingHTTPServer."""
    print("\n============================================================")
    print("      LAUNCHING ANTIGRAVITY RAG EXECUTIVE WEB UI")
    print("============================================================")
    print(f"[Initializing] Loading config from '{config_path}'...")

    retriever = None
    try:
        retriever = VectorRetriever(vector_store_dir="rag/vector_store")
        print("[Singleton] VectorRetriever loaded successfully into memory.")
    except Exception as e:
        print(f"[Warning] Could not initialize singleton retriever: {e}")

    server_address = ("localhost", port)
    httpd = ThreadingHTTPServer(server_address, ExecutiveUIHandler)
    httpd.allow_reuse_address = True
    httpd.retriever = retriever

    print(f"[Online] Executive Web UI running at: http://localhost:{port}")
    print("[Control] Press Ctrl+C to shutdown server cleanly.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Shutdown] KeyboardInterrupt received. Shutting down UI server...")
    finally:
        httpd.server_close()
        print("[Shutdown] Clean socket release completed. Server stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Antigravity RAG Executive Web UI Server")
    parser.add_argument("--port", type=int, default=7860, help="Port to host UI server on")
    parser.add_argument("--config", type=str, default="configs/rag.yaml", help="Path to config file")
    args = parser.parse_args()

    run_server(port=args.port, config_path=args.config)
