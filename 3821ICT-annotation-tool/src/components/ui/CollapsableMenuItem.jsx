import { useState } from "react";
const CollapsibleMenuItem = ({ title, icon, children, open }) => {
    const [isOpen, setIsOpen] = useState(open == true || false);

    const toggleOpen = () => {
        setIsOpen(!isOpen);
    };

    return (
        <li className="mb-4">
            <button
                onClick={toggleOpen}
                className="flex items-center justify-between w-full text-lg font-medium text-foreground hover:cursor-pointer hover:text-muted-foreground transition-colors duration-200"
            >
                <div className="flex items-center">
                    {icon}
                    <span>{title}</span>
                </div>
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className={`h-5 w-5 ml-2 transition-transform duration-200 ${isOpen ? 'rotate-90' : ''}`}
                    viewBox="0 0 20 20"
                    fill="currentColor"
                >
                    <path
                        fillRule="evenodd"
                        d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"
                        clipRule="evenodd"
                    />
                </svg>
            </button>
            <div className={`overflow-hidden transition-all duration-300 ease-in-out ${isOpen ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'}`}>
                {children}
            </div>
        </li>
    );
};

export default CollapsibleMenuItem