import React, { useState, useEffect } from 'react';
import { Key, Check, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import Tooltip from './Tooltip';

export function ApiTokenInput({
    placeholder = "Enter your API token here...",
    label = "API Token",
    description = "Your API token will be stored securely and used for authentication."
}) {
    const [token, setToken] = useState('');
    const [isValid, setIsValid] = useState(false);
    const [isSaved, setIsSaved] = useState(false);

    useEffect(() => {
        fetch("http://localhost:9090/get-labelstudio-key")
            .then(res => res.json())
            .then(data => {
                if (data) {
                    setToken(data);
                    setIsSaved(true);
                    setIsValid(true);
                }
            })
            .catch(err => console.error("Error fetching token:", err));
    }, []);


    const truncateToken = (tokenValue) => {
        if (tokenValue.length <= 8) return tokenValue;
        const start = tokenValue.slice(0, 4);
        const end = tokenValue.slice(-4);
        return `${start}...${end}`;
    };

    const handleTokenChange = (e) => {
        const value = e.target.value;
        setToken(value);
        setIsValid(value.length > 0);
        setIsSaved(false);
    };

    const handleSave = async () => {
        if (!token.trim()) return;
        try {
            const res = await fetch("http://localhost:9090/set-labelstudio-key", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ api_key: token.trim() }),
            });

            if (res.ok) {
                setIsSaved(true);
            } else {
                console.error("Failed to save token");
            }
        } catch (err) {
            console.error("Error saving token:", err);
        }
    };


    const handleDelete = () => {
        setToken('');
        setIsValid(false);
        setIsSaved(false);
    };

    const displayValue = token && isSaved ? truncateToken(token) : token;
    
    // Framer Motion variants for animations
    const containerVariants = {
        hidden: { opacity: 0, y: 20 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' } }
    };

    const saveButtonVariants = {
        hidden: { opacity: 0, scale: 0.9 },
        visible: { opacity: 1, scale: 1, transition: { duration: 0.3 } },
        exit: { opacity: 0, scale: 0.9, transition: { duration: 0.2 } }
    };

    const savedStateVariants = {
        hidden: { opacity: 0, y: -10 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.3 } },
        exit: { opacity: 0, y: 10, transition: { duration: 0.2 } }
    };

    return (
        <motion.div
            className="w-full rounded-2xl p-6 bg-accent max-w-md space-y-4"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
        >
            <div className="space-y-2">
                <div className="flex items-center gap-2">
                    <Key className="w-4 h-4 text-muted-foreground" />
                    <label htmlFor="api-token" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                        {label}
                    </label>
                    <Tooltip text={"This token can be found in labelstudio > account settings > personal access token"}/>
                </div>
                <p className="text-sm text-muted-foreground">{description}</p>
            </div>

            <div className="relative">
                <input
                    id="api-token"
                    type="text"
                    value={displayValue}
                    onChange={handleTokenChange}
                    placeholder={placeholder}
                    className="flex bg-background h-10 w-full rounded-md border border-input bg-input-background px-3 py-2 pr-12 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                    readOnly={isSaved}
                />
                <div className="absolute right-0 top-0 h-full flex items-center">
                    {token && (
                        <motion.button
                            type="button"
                            className="px-3 py-2 hover:bg-muted/50 rounded-sm focus:outline-none text-muted-foreground hover:text-destructive transition-colors"
                            onClick={handleDelete}
                            title="Delete token"
                            whileHover={{ scale: 3 }}
                            whileTap={{ scale: 0.9 }}
                        >
                            <X className="w-4 h-4" />
                        </motion.button>
                    )}
                </div>
            </div>

            <AnimatePresence mode="wait">
                {!isSaved ? (
                    <motion.button
                        key="save-button"
                        onClick={handleSave}
                        disabled={!isValid}
                        className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 h-10 px-4 py-2 w-full bg-primary text-primary-foreground hover:bg-primary/90"
                        variants={saveButtonVariants}
                        initial="hidden"
                        animate="visible"
                        exit="exit"
                        whileHover={{ scale: 1.04 }}
                        whileTap={{ scale: 0.98 }}
                    >
                        Save Token
                    </motion.button>
                ) : (
                    <motion.div
                        key="saved-state"
                        className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium h-10 px-4 py-2 w-full bg-secondary text-secondary-foreground"
                        variants={savedStateVariants}
                        initial="hidden"
                        animate="visible"
                        exit="exit"
                    >
                        <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            transition={{ delay: 0.2, type: 'spring', stiffness: 500, damping: 20 }}
                        >
                            <Check className="w-4 h-4 mr-2" />
                        </motion.div>
                        Token Saved
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
}