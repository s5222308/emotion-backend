import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const ProgressMonitor = () => {
    const [progress, setProgress] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchProgress = async () => {
            try {
                const response = await fetch('http://localhost:9090/get_progress');
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                const data = await response.json();
                setProgress(data);
                setError(null); // Clear any previous errors
            } catch (err) {
                setError('Failed to fetch progress. Is the backend running?');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        // Fetch immediately and then every 3 seconds
        fetchProgress();
        const intervalId = setInterval(fetchProgress, 3000);

        // Cleanup function to clear the interval
        return () => clearInterval(intervalId);
    }, []);

    const containerVariants = {
        hidden: { opacity: 0, scale: 0.95 },
        visible: { opacity: 1, scale: 1, transition: { duration: 0.5, ease: 'easeOut' } },
        exit: { opacity: 0, scale: 0.95, transition: { duration: 0.3 } }
    };

    const countVariants = {
        animate: {
            opacity: 1,
            y: 0,
            transition: { type: "spring", stiffness: 300, damping: 24 }
        },
        initial: {
            opacity: 0,
            y: 10
        }
    };
    
    // The loading and error states are also animated for a smooth transition.
    return (
        <AnimatePresence mode="wait">
            {loading && (
                <motion.div
                    key="loading"
                    className="p-2 bg-gray-200 rounded-xl shadow-md text-center"
                    variants={containerVariants}
                    initial="hidden"
                    animate="visible"
                    exit="exit"
                >
                    Loading progress...
                </motion.div>
            )}

            {error && (
                <motion.div
                    key="error"
                    className="p-2 bg-red-200 text-red-800 rounded-xl shadow-md text-center"
                    variants={containerVariants}
                    initial="hidden"
                    animate="visible"
                    exit="exit"
                >
                    {error}
                </motion.div>
            )}

            {progress && !loading && !error && (
                <motion.div
                    key="progress"
                    className="p-3 bg-muted rounded-xl text-foreground"
                    variants={containerVariants}
                    initial="hidden"
                    animate="visible"
                    exit="exit"
                >
                    <h3 className="text-md text-center font-bold text-foreground mb-2">Task Progress</h3>
                    <div className="grid grid-cols-2">
                        <span className="font-semibold">Submitted</span>
                        <motion.span 
                            key={`submitted-${progress.submitted}`}
                            className="text-center"
                            variants={countVariants}
                            initial="initial"
                            animate="animate"
                        >
                            {progress.submitted}
                        </motion.span>
                        <span className="font-semibold">Completed</span>
                        <motion.span 
                            key={`completed-${progress.completed}`}
                            className="text-center"
                            variants={countVariants}
                            initial="initial"
                            animate="animate"
                        >
                            {progress.completed}
                        </motion.span>
                        <span className="font-semibold col-span-2">Status</span>
                        <span className="text-sm col-span-2">{progress.message}</span>
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};

export default ProgressMonitor;