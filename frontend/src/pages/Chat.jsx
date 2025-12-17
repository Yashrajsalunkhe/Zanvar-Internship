import React, { useState, useRef, useEffect } from "react";
import Navbar from "../components/Navbar";
import { sendChatMessage } from '../api';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';

const AI_AVATAR = "https://lh3.googleusercontent.com/aida-public/AB6AXuB0wTJWIrhQLopY-NST5feHfVDAW3zt9fBCsOqMH7dKm-alsHLUuB-obdVAvz9OW9yvwXShnjqyZPb3peYkyp_0qmg-JRqwrsXXlIF5QsSKJlMOL5Fwvlq-uEZqhXC1WuviG7Cm9F1vDsxt6qlUw8djKsMXcvB6-dERLimymkLwinEZVAi8UyU6VUxAWxkkeODlqWnUihH0ssmhexrOAhW6FKxh1Ywgc3l4luHOALFzb0_UBMalrmXPiJsrnZUXOGuZEBbt3TBqXgY";
const USER_AVATAR = "https://lh3.googleusercontent.com/aida-public/AB6AXuCowg1gvXl3EKMwpawwVdqdaWLKixCbSbj3seHqsvId_ih8u-jW9o68SBfvHmr_GgkeWydQ4NMVasHaV6a_vmnEGFgC6UeBQs01IfTSXtrQkuAjUDYu2aR7AU01mCFwoUd4sJx0FgMX4bTLLXnfN30fJGKtY-qnwaowq5MAnUMC42DHZ1jpZkSzUGDjRX_NVQte3tUtxPscdaZopJZly8xlDcVtPCXjxg09Jo97XKwNvjcGvp0VaL4ZrRzxRja7mxJm7j5gcKkxURA";

const Chat = () => {
  const [messages, setMessages] = useState([
    {
      sender: "ai",
      name: "AI Assistant",
      avatar: AI_AVATAR,
      text: "Hi there! I'm ready to help you analyze your data. What would you like to know?",
    },
  ]);
  const [input, setInput] = useState("");
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    // On mount, check localStorage for uploaded file info
    const fileInfo = localStorage.getItem('uploadedFile');
    if (fileInfo) {
      setUploadedFile(JSON.parse(fileInfo));
    }
  }, []);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    setError(null);
    const userMsg = {
      sender: "user",
      name: "You",
      avatar: USER_AVATAR,
      text: input,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    try {
      const res = await sendChatMessage(input);
      const aiMessage = {
        sender: "ai",
        name: "AI Assistant",
        avatar: AI_AVATAR,
        text: res.response || res.reply || "(No response from backend)",
      };

      // Include chart data if present
      if (res.chart_data) {
        aiMessage.chartData = res.chart_data;
      }

      setMessages((prev) => [...prev, aiMessage]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          sender: "ai",
          name: "AI Assistant",
          avatar: AI_AVATAR,
          text: "Sorry, there was an error contacting the backend.",
        },
      ]);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative flex h-screen w-full flex-col bg-[#0f172a] text-slate-50 font-[Inter] overflow-hidden selection:bg-indigo-500/30">

      {/* Ambient Background Effects */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-10%] right-[-5%] w-[500px] h-[500px] bg-indigo-500/10 rounded-full blur-[100px] animate-pulse"></div>
        <div className="absolute bottom-[-10%] left-[-10%] w-[600px] h-[600px] bg-blue-600/10 rounded-full blur-[120px]"></div>
      </div>

      {/* Analytics Drawer (Overlay) */}
      {showAnalytics && (
        <div className="fixed inset-0 z-50 flex justify-end bg-black/60 backdrop-blur-sm transition-all duration-300">
          <div className="w-full max-w-md h-full bg-[#1e293b]/95 backdrop-blur-xl border-l border-white/10 shadow-2xl flex flex-col transform transition-transform duration-300 animate-slide-in-right">
            <div className="flex items-center justify-between px-6 py-5 border-b border-white/5">
              <span className="text-white text-lg font-semibold tracking-tight">Data Analytics</span>
              <button
                className="text-slate-400 hover:text-white p-2 rounded-full hover:bg-white/5 transition-all"
                onClick={() => setShowAnalytics(false)}
                aria-label="Close analytics panel"
              >
                <svg width="20" height="20" fill="none" viewBox="0 0 24 24"><path stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
              </button>
            </div>
            <div className="flex-1 flex flex-col items-center justify-center text-slate-300 px-8">
              <div className="w-16 h-16 bg-gradient-to-br from-indigo-500 to-blue-600 rounded-2xl flex items-center justify-center mb-6 shadow-lg shadow-indigo-500/20">
                <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
              </div>
              <span className="text-xl font-medium text-white mb-2">Analytics Pro</span>
              <p className="text-slate-400 text-center leading-relaxed">Advanced visualizations and deep insights are coming soon to this panel.</p>
            </div>
          </div>
          <div className="flex-1" onClick={() => setShowAnalytics(false)} />
        </div>
      )}

      {/* Header */}
      <Navbar activePage="chat" />

      {/* Main Content Area */}
      <main className="relative flex-1 flex flex-col max-w-5xl mx-auto w-full p-4 md:p-6 z-10 overflow-hidden">

        {/* Chat History */}
        <div className="flex-1 overflow-y-auto pr-2 space-y-6 scrollbar-hide py-4">

          {/* Welcome/Empty State if needed, or File Badge */}
          {uploadedFile && (
            <div className="flex justify-center mb-6">
              <div className="flex items-center gap-2 px-4 py-2 bg-indigo-500/10 border border-indigo-500/20 rounded-full text-indigo-300 text-xs font-medium">
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                <span>Analyzing: {uploadedFile.filename}</span>
              </div>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex items-start gap-4 animate-fade-in-up ${msg.sender === "user" ? "flex-row-reverse" : ""}`}
              style={{ animationDelay: `${idx * 0.05}s` }}
            >
              <div className={`shrink-0 w-9 h-9 rounded-full overflow-hidden border border-white/10 shadow-lg ${msg.sender === "ai" ? "bg-indigo-600 p-0.5" : "bg-slate-700"}`}>
                <img src={msg.avatar} alt={msg.name} className="w-full h-full object-cover rounded-full" />
              </div>

              <div className={`flex flex-col gap-1 max-w-[85%] md:max-w-[75%] ${msg.sender === "user" ? "items-end" : "items-start"}`}>
                <span className="text-[11px] font-medium text-slate-400 px-1">{msg.name}</span>

                {msg.sender === "ai" ? (
                  <div className="group relative">
                    <div className="relative bg-[#1e293b]/80 backdrop-blur-md border border-white/5 rounded-2xl rounded-tl-sm px-6 py-4 shadow-sm text-slate-200 text-[15px] leading-relaxed">
                      <ReactMarkdown
                        rehypePlugins={[rehypeRaw]}
                        components={{
                          ul: (props) => <ul className="list-disc pl-5 my-2 space-y-1" {...props} />,
                          ol: (props) => <ol className="list-decimal pl-5 my-2 space-y-1" {...props} />,
                          li: (props) => <li className="pl-1" {...props} />,
                          p: (props) => <p className="mb-2 last:mb-0" {...props} />,
                          a: (props) => <a className="text-indigo-400 hover:text-indigo-300 underline underline-offset-2" {...props} />,
                          code: (props) => <code className="bg-black/30 px-1.5 py-0.5 rounded text-xs font-mono text-indigo-300" {...props} />,
                          img: (props) => <img className="rounded-lg my-3 border border-white/10 max-w-full" {...props} alt="content" />
                        }}
                      >
                        {msg.text}
                      </ReactMarkdown>

                      {/* Chart Visualization */}
                      {msg.chartData && (
                        <div className="mt-4 bg-[#0f172a] rounded-xl border border-white/10 overflow-hidden shadow-inner">
                          <div className="px-4 py-3 border-b border-white/5 bg-white/5 flex items-center justify-between">
                            <h4 className="text-sm font-semibold text-indigo-300 flex items-center gap-2">
                              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" /></svg>
                              {msg.chartData.title || 'Data Visualization'}
                            </h4>
                          </div>
                          <div className="p-5">
                            {msg.chartData.type === 'bar' && (
                              <div className="space-y-4">
                                {msg.chartData.labels?.map((label, i) => {
                                  const value = msg.chartData.datasets?.[0]?.data?.[i] || 0;
                                  const maxVal = Math.max(...(msg.chartData.datasets?.[0]?.data?.map(v => parseFloat(v) || 0) || [1]));
                                  const percentage = (parseFloat(value) / maxVal) * 100;
                                  return (
                                    <div key={i} className="group/bar">
                                      <div className="flex justify-between text-xs mb-1.5">
                                        <span className="text-slate-400 font-medium">{label}</span>
                                        <span className="text-slate-200 font-mono">{value}</span>
                                      </div>
                                      <div className="w-full bg-white/5 rounded-full h-2.5 overflow-hidden">
                                        <div
                                          className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-1000 ease-out group-hover/bar:brightness-110 relative"
                                          style={{ width: `${percentage}%` }}
                                        >
                                          <div className="absolute inset-0 bg-white/20 animate-shimmer" style={{ backgroundSize: '200% 100%' }}></div>
                                        </div>
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            )}
                            {(msg.chartData.type === 'pie' || msg.chartData.type === 'line') && (
                              <div className="text-center py-8">
                                <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-white/5 mb-3">
                                  <svg className="w-6 h-6 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
                                </div>
                                <p className="text-sm text-slate-400">
                                  {msg.chartData.type === 'pie' ? 'Pie' : 'Line'} chart data is available but simplified view is shown.
                                </p>
                                <div className="mt-4 grid grid-cols-2 gap-2 text-left">
                                  {msg.chartData.labels?.slice(0, 6).map((label, i) => (
                                    <div key={i} className="bg-white/5 p-2 rounded border border-white/5 text-xs">
                                      <div className="text-slate-500 truncate">{label}</div>
                                      <div className="text-slate-300 font-mono">{msg.chartData.datasets?.[0]?.data?.[i]}</div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="bg-gradient-to-br from-indigo-600 to-violet-600 text-white rounded-2xl rounded-tr-sm px-5 py-3.5 shadow-md text-[15px] leading-relaxed selection:bg-white/30">
                    {msg.text}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex items-start gap-4 animate-fade-in-up">
              <div className="shrink-0 w-9 h-9 rounded-full bg-indigo-600 p-0.5 border border-white/10 shadow-lg">
                <img src={AI_AVATAR} alt="AI" className="w-full h-full object-cover rounded-full" />
              </div>
              <div className="bg-[#1e293b]/50 backdrop-blur-sm border border-white/5 rounded-2xl rounded-tl-sm px-5 py-4">
                <div className="flex gap-1.5">
                  <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '0s' }}></div>
                  <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '0.15s' }}></div>
                  <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '0.3s' }}></div>
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="flex justify-center animate-fade-in-up">
              <div className="bg-red-500/10 border border-red-500/20 text-red-200 px-4 py-2 rounded-lg text-sm flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                {error}
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area (Floating) */}
        <div className="relative mt-2">
          <form
            onSubmit={handleSend}
            className="group relative flex items-center gap-2 bg-[#1e293b]/80 backdrop-blur-xl border border-white/10 rounded-full p-2 pr-2 shadow-2xl transition-all duration-300 focus-within:border-indigo-500/50 focus-within:shadow-indigo-500/10 focus-within:bg-[#1e293b]"
          >
            <button
              type="button"
              className="p-3 text-slate-400 hover:text-indigo-400 hover:bg-white/5 rounded-full transition-colors"
              onClick={() => setShowAnalytics(true)}
              title="View Analytics"
            >
              <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
            </button>

            <input
              className="flex-1 bg-transparent border-none outline-none text-slate-200 placeholder-slate-500 text-[15px] pl-2 min-h-[44px]"
              placeholder="Ask something about your data..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading}
              autoComplete="off"
            />

            <button
              type="submit"
              disabled={!input.trim() || loading}
              className={`p-3 rounded-full transition-all duration-300 flex items-center justify-center
                ${!input.trim() || loading
                  ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                  : 'bg-indigo-600 text-white hover:bg-indigo-500 shadow-lg shadow-indigo-600/30 hover:scale-105 active:scale-95'
                }`}
            >
              {loading ? (
                <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
              ) : (
                <svg className="w-5 h-5 ml-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M14 5l7 7m0 0l-7 7m7-7H3" /></svg>
              )}
            </button>
          </form>
          <div className="text-center mt-2.5">
            <p className="text-[10px] text-slate-600 font-medium tracking-wide uppercase">AI-Powered Insights • Zanvar Internship</p>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Chat;
