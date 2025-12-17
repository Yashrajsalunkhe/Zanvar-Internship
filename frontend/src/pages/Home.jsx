import React from "react";
import Navbar from "../components/Navbar";

const Home = () => {
  return (
    <div className="relative flex min-h-screen w-full flex-col bg-[#0f172a] text-slate-50 font-[Inter] overflow-x-hidden selection:bg-indigo-500/30">

      {/* Ambient Background */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[-10%] w-[800px] h-[800px] bg-indigo-600/10 rounded-full blur-[120px]"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[600px] h-[600px] bg-blue-600/10 rounded-full blur-[100px]"></div>
      </div>

      {/* Header */}
      <Navbar activePage="home" />

      <main className="relative z-10 flex-1 flex flex-col items-center">

        {/* Hero Section */}
        <section className="relative w-full max-w-7xl mx-auto px-6 py-20 md:py-32 flex flex-col items-center text-center">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[300px] bg-indigo-500/30 blur-[120px] rounded-full pointer-events-none"></div>

          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-xs font-medium mb-8 animate-fade-in-up">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
            </span>
            AI-Powered Analytics v2.0
          </div>

          <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-white mb-6 max-w-4xl leading-tight animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
            Unlock the Power of <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-indigo-400 animate-gradient-x">Your Data</span> with AI
          </h1>

          <p className="text-lg md:text-xl text-slate-400 max-w-2xl mb-10 leading-relaxed animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
            Transform your spreadsheets into actionable insights. Upload your data, ask questions, and visualize results instantly with our advanced intelligent agent.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto animate-fade-in-up" style={{ animationDelay: '0.3s' }}>
            <button
              onClick={() => window.location.href = '/upload'}
              className="h-12 px-8 rounded-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium transition-all shadow-xl shadow-indigo-600/25 hover:scale-105 active:scale-95 flex items-center justify-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>
              Upload File
            </button>
            <button
              onClick={() => window.location.href = '/chat'}
              className="h-12 px-8 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 text-white font-medium transition-all hover:scale-105 active:scale-95 backdrop-blur-sm flex items-center justify-center gap-2"
            >
              <svg className="w-5 h-5 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" /></svg>
              Try Demo Chat
            </button>
          </div>
        </section>

        {/* Features Grid */}
        <section className="w-full max-w-7xl mx-auto px-6 py-20">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Feature 1 */}
            <div className="group p-8 rounded-3xl bg-white/5 border border-white/5 hover:bg-white/10 hover:border-indigo-500/30 transition-all duration-300 hover:-translate-y-1">
              <div className="w-12 h-12 rounded-2xl bg-indigo-500/20 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                <svg className="w-6 h-6 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
              </div>
              <h3 className="text-xl font-semibold text-white mb-3">Instant Analysis</h3>
              <p className="text-slate-400 leading-relaxed">
                Upload large datasets and get immediate processing. No more waiting for manual calculations or complex formulas.
              </p>
            </div>

            {/* Feature 2 */}
            <div className="group p-8 rounded-3xl bg-white/5 border border-white/5 hover:bg-white/10 hover:border-indigo-500/30 transition-all duration-300 hover:-translate-y-1">
              <div className="w-12 h-12 rounded-2xl bg-purple-500/20 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                <svg className="w-6 h-6 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-3.582 8-8 8a8.013 8.013 0 01-5.699-2.356l-3.293 3.292a1 1 0 01-1.414 0l-1.414-1.414a1 1 0 010-1.414l3.292-3.293A8.013 8.013 0 012 12c0-4.418 3.582-8 8-8s8 3.582 8 8z" /></svg>
              </div>
              <h3 className="text-xl font-semibold text-white mb-3">Natural Conversation</h3>
              <p className="text-slate-400 leading-relaxed">
                Chat with your data as if it were a colleague. Ask plain English questions and get accurate, data-backed answers.
              </p>
            </div>

            {/* Feature 3 */}
            <div className="group p-8 rounded-3xl bg-white/5 border border-white/5 hover:bg-white/10 hover:border-indigo-500/30 transition-all duration-300 hover:-translate-y-1">
              <div className="w-12 h-12 rounded-2xl bg-blue-500/20 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                <svg className="w-6 h-6 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
              </div>
              <h3 className="text-xl font-semibold text-white mb-3">Rich Visualizations</h3>
              <p className="text-slate-400 leading-relaxed">
                Automatically generate beautiful bar, line, and pie charts that help you see the trends and outliers clearly.
              </p>
            </div>
          </div>
        </section>

      </main>

      <footer className="border-t border-white/5 bg-[#0f172a] py-8 text-center text-slate-500 text-sm">
        <p>© 2025 Zanvar Data Insights. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default Home;
