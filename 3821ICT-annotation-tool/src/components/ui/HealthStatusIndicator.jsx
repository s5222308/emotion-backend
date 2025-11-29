import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from 'framer-motion';

export function HealthStatusIndicator({
    endpoint = "http://localhost:9090/health",
    interval = 5000,
}) {
    const [isHealthy, setIsHealthy] = useState(null);

    useEffect(() => {
        const checkHealth = async () => {
            try {
                const res = await fetch(endpoint);
                setIsHealthy(res.ok);
            } catch (err) {
                setIsHealthy(false);
            }
        };

        checkHealth();
        const timer = setInterval(checkHealth, interval);

        return () => clearInterval(timer);
    }, [endpoint, interval]);
    
    // Variants for the status dot animation
    const dotVariants = {
        hidden: { scale: 0, opacity: 0 },
        visible: { scale: 1, opacity: 1, transition: { duration: 0.3 } }
    };

    // Variants for the status text
    const textVariants = {
        hidden: { opacity: 0, x: -10 },
        visible: { opacity: 1, x: 0, transition: { duration: 0.3 } }
    };

    return (
        <div className="flex rounded-2xl p-6 flex-col items-start bg-muted gap-2">
            <h1 className="text-md">Emotion Backend</h1>
            <div className="flex items-center gap-2">
                <AnimatePresence mode="wait">
                    <motion.span
                        key={isHealthy === null ? 'checking' : isHealthy ? 'healthy' : 'unhealthy'}
                        className={`inline-block w-3 h-3 rounded-full ${
                            isHealthy === null
                                ? "bg-gray-400 animate-pulse"
                                : isHealthy
                                    ? "bg-green-500"
                                    : "bg-red-500"
                        }`}
                        variants={dotVariants}
                        initial="hidden"
                        animate="visible"
                        exit="hidden"
                    />
                </AnimatePresence>
                <AnimatePresence mode="wait">
                    <motion.span
                        key={isHealthy === null ? 'checking-text' : isHealthy ? 'healthy-text' : 'unhealthy-text'}
                        className="text-sm"
                        variants={textVariants}
                        initial="hidden"
                        animate="visible"
                        exit="hidden"
                    >
                        {isHealthy === null
                            ? "Checking..."
                            : isHealthy
                                ? "Backend is healthy"
                                : "Backend is unreachable"}
                    </motion.span>
                </AnimatePresence>
            </div>
            <div className="flex flex-col justify-end h-full">
                <span className="text-sm align-bottom text-muted-foreground">Powered by: Ultralytics, Pytorch, Flask</span>
            </div>
        </div>
    );
}