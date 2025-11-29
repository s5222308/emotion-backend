import React from 'react';

const ToggleSidebarButton = () => {
    const handleToggle = () => {
        const sidebar = document.getElementById('sidebar');
        if (sidebar) {
            if (sidebar.classList.contains('w-[25rem]')) {
                sidebar.classList.remove('w-[25rem]');
                sidebar.classList.add('w-0', 'overflow-hidden');
            } else {
                sidebar.classList.remove('w-0', 'overflow-hidden');
                sidebar.classList.add('w-[25rem]');
            }
        }
    };
    
    return (
        <button 
            onClick={handleToggle}
            className={`
                z-50 p-2 rounded-full
                transition-transform duration-300 ease-in-out
                bg-blue-600 text-white shadow-lg
                hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
            `}
            aria-label="Toggle Sidebar"
        >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
        </button>
    );
};

export default ToggleSidebarButton