import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

// --- Interfaces for TypeScript ---
interface SourceDocument {
  content: string;
  page: number;
}
interface Message {
  sender: "user" | "ai";
  text: string;
  sourceDocuments?: SourceDocument[];
}

const UploadIcon: React.FC = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
    <polyline points="17 8 12 3 7 8"></polyline>
    <line x1="12" y1="3" x2="12" y2="15"></line>
  </svg>
);

const SendIcon: React.FC = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <line x1="22" y1="2" x2="11" y2="13"></line>
    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
  </svg>
);

const AiAvatar: React.FC = () => (
  <div className="avatar ai-avatar">
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 8V4H8"></path>
      <rect x="4" y="12" width="8" height="8" rx="2"></rect>
      <path d="M8 12v-2a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-4a2 2 0 0 1-2-2v-2"></path>
    </svg>
  </div>
);

const UserAvatar: React.FC = () => (
  <div className="avatar user-avatar">
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
      <circle cx="12" cy="7" r="4"></circle>
    </svg>
  </div>
);

// --- Sources Component with Typed Props ---
const Sources: React.FC<{ documents: SourceDocument[] }> = ({ documents }) => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  if (!documents || documents.length === 0) return null;
  return (
    <div className="sources-container">
      <button onClick={() => setIsOpen(!isOpen)} className="sources-button">
        {isOpen ? "Hide Sources" : `View ${documents.length} Sources`}
      </button>
      {isOpen && (
        <div className="sources-content">
          {documents.map((doc, index) => (
            <div key={index} className="source-item">
              <p>
                <strong>Source {index + 1}</strong> (Page {doc.page + 1})
              </p>
              <p className="source-quote">
                "{doc.content.substring(0, 150)}..."
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// --- Main App Component ---
function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [docId, setDocId] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string>("");
  const [isReady, setIsReady] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentQuery, setCurrentQuery] = useState<string>("");

  const chatEndRef = useRef<HTMLDivElement>(null);
  const API_URL = "http://127.0.0.1:8000";

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
      setStatusMessage(`Selected: ${event.target.files[0].name}`);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    const formData = new FormData();
    formData.append("file", selectedFile);
    setIsLoading(true);
    setStatusMessage(`Processing ${selectedFile.name}...`);
    try {
      const response = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok)
        throw new Error(data.detail || "Failed to upload file.");
      setDocId(data.doc_id);
      setMessages([
        {
          sender: "ai",
          text: `I've finished analyzing **${selectedFile.name}**. What would you like to know?`,
        },
      ]);
      setIsReady(true);
    } catch (error) {
      setStatusMessage(`Error: ${(error as Error).message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuery = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!currentQuery.trim() || isLoading || !docId) return;
    const userMessage: Message = { sender: "user", text: currentQuery };
    setMessages((prev) => [
      ...prev,
      userMessage,
      { sender: "ai", text: "Thinking..." },
    ]);
    const query = currentQuery;
    setCurrentQuery("");
    setIsLoading(true);
    try {
      const response = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_id: docId, question: query }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to get an answer.");
      }
      const result = await response.json();
      const aiMessage: Message = {
        sender: "ai",
        text: result.report.answer,
        sourceDocuments: result.source_documents,
      };
      setMessages((prev) => [...prev.slice(0, -1), aiMessage]);
    } catch (error) {
      setMessages((prev) => [
        ...prev.slice(0, -1),
        {
          sender: "ai",
          text: `Sorry, an error occurred: ${(error as Error).message}`,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewUpload = () => {
    setSelectedFile(null);
    setDocId(null);
    setIsReady(false);
    setMessages([]);
    setStatusMessage("");
  };

  return (
    <div className="app">
      <header className="header">
        <h1>
          <AiAvatar /> Deep Researcher
        </h1>
        {isReady && (
          <button onClick={handleNewUpload} className="new-upload-btn">
            New Document
          </button>
        )}
      </header>

      <main className="mainContent">
        {!isReady ? (
          <div className="uploadContainer">
            <h2>Upload a Document to Analyze</h2>
            <p>
              Your document will be processed locally and will not be stored on
              our servers.
            </p>
            <div className="upload-box">
              <input
                type="file"
                id="file-upload"
                onChange={handleFileChange}
                accept=".pdf"
              />
              <label htmlFor="file-upload" className="file-upload-label">
                <UploadIcon />
                <span>
                  {selectedFile
                    ? selectedFile.name
                    : "Click to choose a PDF file"}
                </span>
              </label>
              <button
                onClick={handleUpload}
                disabled={isLoading || !selectedFile}
                className="uploadButton"
              >
                {isLoading ? "Processing..." : "Start Session"}
              </button>
            </div>
            {statusMessage && <p className="status-message">{statusMessage}</p>}
          </div>
        ) : (
          <div className="chatContainer">
            <div className="messages">
              {messages.map((msg, index) => (
                <div
                  key={index}
                  className={`message-wrapper ${msg.sender}-wrapper`}
                >
                  {msg.sender === "ai" ? <AiAvatar /> : <UserAvatar />}
                  <div className="message-content">
                    <div className={`message ${msg.sender}`}>
                      {msg.text === "Thinking..." ? (
                        <div className="loading-dots">
                          <span></span>
                          <span></span>
                          <span></span>
                        </div>
                      ) : (
                        <div className="markdown-container">
                          <ReactMarkdown>{msg.text}</ReactMarkdown>
                        </div>
                      )}
                    </div>
                    {msg.sender === "ai" &&
                      msg.sourceDocuments &&
                      msg.sourceDocuments.length > 0 && (
                        <Sources documents={msg.sourceDocuments} />
                      )}
                  </div>
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>
            <form onSubmit={handleQuery} className="query-form">
              <input
                type="text"
                value={currentQuery}
                onChange={(e) => setCurrentQuery(e.target.value)}
                placeholder="Ask a follow-up question..."
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !currentQuery.trim()}
              >
                <SendIcon />
              </button>
            </form>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
