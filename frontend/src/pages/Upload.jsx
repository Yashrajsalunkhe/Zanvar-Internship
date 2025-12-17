import React, { useState } from "react";
import Navbar from "../components/Navbar";
import { uploadFile } from '../api';

const Upload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState("");
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  };

  const handleFileSelection = async (file) => {
    setSelectedFile(file);
    setUploadProgress(0);
    setStatus("");
    setUploading(true);
    try {
      const res = await uploadFile(file);
      if (res.error) {
        setStatus(`Upload failed: ${res.error}`);
      } else {
        setUploadProgress(100);
        setStatus("Processing complete!");
        localStorage.setItem('uploadedFile', JSON.stringify({
          filename: res.filename || res.file_info?.Filename || file.name,
          format: (res.filename || file.name).split('.').pop().toUpperCase()
        }));
        setTimeout(() => {
          window.location.href = '/chat';
        }, 1200);
      }
    } catch (err) {
      setStatus("Upload failed: " + err.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="relative flex min-h-screen w-full flex-col bg-[#0f172a] text-slate-50 font-[Inter] overflow-x-hidden selection:bg-indigo-500/30">

      {/* Background Blobs */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-20%] right-[-10%] w-[600px] h-[600px] bg-purple-600/10 rounded-full blur-[100px]"></div>
        <div className="absolute bottom-[-10%] left-[-10%] w-[500px] h-[500px] bg-indigo-600/10 rounded-full blur-[100px]"></div>
      </div>

      {/* Header */}
      <Navbar activePage="upload" />

      <main className="relative z-10 flex-1 flex flex-col items-center justify-center p-6">

        <div className="w-full max-w-2xl animate-fade-in-up">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-white mb-2">Upload Your Dataset</h2>
            <p className="text-slate-400">Supports .xlsx and .csv files. We'll analyze it instantly.</p>
          </div>

          <div
            className={`relative group rounded-3xl border-2 border-dashed transition-all duration-300 p-12 text-center cursor-pointer overflow-hidden
               ${dragActive
                ? 'border-indigo-500 bg-indigo-500/10 scale-[1.02]'
                : 'border-slate-700 bg-slate-800/30 hover:bg-slate-800/50 hover:border-slate-600'
              }
             `}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-upload').click()}
          >
            <input
              type="file"
              id="file-upload"
              accept=".xlsx,.csv"
              className="hidden"
              onChange={handleChange}
            />

            <div className="relative z-10 flex flex-col items-center gap-4">
              <div className={`size-16 rounded-2xl flex items-center justify-center transition-all duration-300 ${dragActive ? 'bg-indigo-500 text-white' : 'bg-slate-700 text-slate-300 group-hover:bg-indigo-600 group-hover:text-white group-hover:scale-110 shadow-lg'}`}>
                <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" /></svg>
              </div>
              <div>
                <p className="text-lg font-semibold text-white group-hover:text-indigo-200 transition-colors">
                  {dragActive ? "Drop file here" : "Click to upload or drag and drop"}
                </p>
                <p className="text-sm text-slate-500 mt-1">
                  Excel or CSV files up to 10MB
                </p>
              </div>
            </div>
          </div>

          {selectedFile && (
            <div className="mt-6 rounded-2xl bg-white/5 border border-white/5 p-4 animate-fade-in-up">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="size-10 rounded-lg bg-green-500/20 text-green-400 flex items-center justify-center">
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-white truncate max-w-[200px]">{selectedFile.name}</p>
                    <p className="text-xs text-slate-400">{(selectedFile.size / 1024).toFixed(1)} KB</p>
                  </div>
                </div>
                {status && (
                  <span className={`text-xs font-semibold px-2 py-1 rounded-md ${status.includes('failed') ? 'bg-red-500/20 text-red-300' : 'bg-green-500/20 text-green-300'}`}>
                    {status}
                  </span>
                )}
              </div>

              <div className="h-2 w-full bg-slate-700/50 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-700 ease-out ${status.includes('failed') ? 'bg-red-500' : 'bg-gradient-to-r from-indigo-500 to-green-400 relative'}`}
                  style={{ width: `${uploadProgress}%` }}
                >
                  {!status && <div className="absolute inset-0 bg-white/30 animate-shimmer" style={{ backgroundSize: '100% 100%' }}></div>}
                </div>
              </div>
            </div>
          )}

        </div>
      </main>
    </div>
  );
};

export default Upload;
