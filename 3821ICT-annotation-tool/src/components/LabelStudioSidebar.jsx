import { motion } from 'framer-motion';

const Sidebar = ({children, bottomChildren}) => {
    return (
        <motion.div
            id="sidebar"
            className={`
                h-screen flex flex-col bg-background text-foreground
                w-0 shadow-lg
            `}
            initial={{ x: -300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.5, type: "spring", stiffness: 100 }}
        >
            <div className="flex gap-4 items-start p-4">
                <h1 className="text-xl font-semibold">LS Control Panel</h1>
                {bottomChildren && (
                    <div className="flex-none">
                        {bottomChildren}
                    </div>
                )}
            </div>
            
            <div className={`p-4 flex-grow overflow-y-auto`}>
                <nav>
                    <ul>
                        {children}
                    </ul>
                </nav>
            </div>
        </motion.div>
    );
};

export default Sidebar;