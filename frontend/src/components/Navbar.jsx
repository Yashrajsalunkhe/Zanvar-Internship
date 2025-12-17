import React from 'react';

const Navbar = ({ activePage = '' }) => {
    const PROFILE_AVATAR = "https://lh3.googleusercontent.com/aida-public/AB6AXuDYdyNgyyZJ4noBYbSowbQlmHmOoT39UUwlEC_T057UfEahdu0OnGFRFCxmREtADsrlfZR9KE8G8w0As4FIHwcpS0lJf4WTu3Z8h-g4OzroeQn7u_R18GyuHYiqffgV_Ego8eJ3ON9Z2cBdt1YRrHSWUWkh2_hFmrUczIs6zmWo5sKsTOXmroNtBKycJ3CTJ5_s8KzaCsq7iH00lmHZGqhl9HGn6fEQFRYjBUwGdGifsFqPzptMAhVi4O5TPHtzdIhvYg7XI9esHsA";

    const getLinkClasses = (pageName) => {
        const isActive = activePage === pageName;
        return `px-4 py-1.5 text-sm font-medium rounded-full transition-all ${isActive
                ? 'text-white bg-indigo-500/20 border border-indigo-500/20 shadow-sm cursor-default'
                : 'text-slate-400 hover:text-white hover:bg-white/5'
            }`;
    };

    return (
        <header className="relative z-10 flex items-center justify-between px-6 py-4 border-b border-white/5 bg-[#0f172a]/80 backdrop-blur-md sticky top-0">
            <div className="flex items-center gap-3">
                <div className="size-8 text-indigo-400">
                    <svg viewBox="0 0 48 48" fill="none" className="w-full h-full">
                        <path fillRule="evenodd" clipRule="evenodd" d="M24 18.4228L42 11.475V34.3663C42 34.7796 41.7457 35.1504 41.3601 35.2992L24 42V18.4228Z" fill="currentColor"></path>
                        <path fillRule="evenodd" clipRule="evenodd" d="M24 8.18819L33.4123 11.574L24 15.2071L14.5877 11.574L24 8.18819ZM9 15.8487L21 20.4805V37.6263L9 32.9945V15.8487ZM27 37.6263V20.4805L39 15.8487V32.9945L27 37.6263ZM25.354 2.29885C24.4788 1.98402 23.5212 1.98402 22.646 2.29885L4.98454 8.65208C3.7939 9.08038 3 10.2097 3 11.475V34.3663C3 36.0196 4.01719 37.5026 5.55962 38.098L22.9197 44.7987C23.6149 45.0671 24.3851 45.0671 25.0803 44.7987L42.4404 38.098C43.9828 37.5026 45 36.0196 45 34.3663V11.475C45 10.2097 44.2061 9.08038 43.0155 8.65208L25.354 2.29885Z" fill="currentColor"></path>
                    </svg>
                </div>
                <h1 className="text-white text-lg font-bold tracking-tight">Zanvar <span className="text-indigo-400 font-medium">Data Insights</span></h1>
            </div>

            <nav className="hidden md:flex items-center gap-1 bg-white/5 rounded-full p-1 border border-white/5">
                <a href="/" className={getLinkClasses('home')}>Home</a>
                <a href="/upload" className={getLinkClasses('upload')}>Upload</a>
                <a href="/chat" className={getLinkClasses('chat')}>Chat</a>
            </nav>

            <div className="flex items-center gap-4">
                <button
                    onClick={() => window.location.href = '/upload'}
                    className={`hidden md:flex items-center justify-center h-9 px-4 rounded-full bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium transition-all shadow-lg shadow-indigo-600/20 hover:scale-105 active:scale-95 ${activePage === 'upload' ? 'hidden' : ''}`}
                >
                    Get Started
                </button>
                <button
                    onClick={() => window.location.href = '/profile'}
                    className="size-9 rounded-full bg-gradient-to-br from-slate-700 to-slate-800 border border-white/10 hover:border-indigo-500/50 transition-all overflow-hidden"
                    title="Profile"
                >
                    <img src={PROFILE_AVATAR} alt="Profile" className="w-full h-full object-cover opacity-90 hover:opacity-100 transition-opacity" />
                </button>
            </div>
        </header>
    );
};

export default Navbar;
