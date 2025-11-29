import { useState } from "react";
import { XCircle, CheckCircle2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const AbortButton = () => {
  const [message, setMessage] = useState(null);
  const [isError, setIsError] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

  const handleAbort = async () => {
    setMessage("Attempting to abort all tasks...");
    setIsError(false);
    try {
      const response = await fetch("http://localhost:9090/abort_all", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (response.ok) {
        setMessage("Successfully aborted all tasks.");
        setIsError(false);
        setIsSuccess(true);

        // reset icon after 2s
        setTimeout(() => setIsSuccess(false), 2000);
      } else {
        const errorText = await response.text();
        setMessage(`Failed to abort tasks: ${errorText || response.statusText}`);
        setIsError(true);
      }
    } catch (error) {
      setMessage(`Error sending abort request: ${error.message}`);
      setIsError(true);
    } finally {
      setTimeout(() => {
        setMessage(null);
      }, 3000);
    }
  };

  const buttonVariants = {
    rest: { scale: 1 },
    hover: { scale: 1.1 },
    tap: { scale: 0.95 },
  };

  const messageVariants = {
    initial: { opacity: 0, y: -10 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: 10 },
  };

  const iconVariants = {
    initial: { opacity: 0, scale: 0.5, rotate: -90 },
    animate: { opacity: 1, scale: 1, rotate: 0 },
    exit: { opacity: 0, scale: 0.5, rotate: 90 },
  };

  return (
    <div className="flex flex-col items-center">
      <div className="relative inline-block group">
        <motion.button
        onClick={handleAbort}
        className="w-8 h-8 hover:cursor-pointer flex items-center justify-center rounded-full text-white font-semibold transition-colors shadow-lg"
        variants={buttonVariants}
        initial="rest"
        whileHover="hover"
        whileTap="tap"
        animate={{
            backgroundColor: isSuccess ? "#16a34a" : "#991b1b", // green-600 vs red-800
        }}
        transition={{ duration: 0.3 }}
        >
            <AnimatePresence mode="wait" initial={false}>
                {isSuccess ? (
                <motion.div
                    key="check"
                    variants={iconVariants}
                    initial="initial"
                    animate="animate"
                    exit="exit"
                >
                    <CheckCircle2 size={20} className="text-white" />
                </motion.div>
                ) : (
                <motion.div
                    key="x"
                    variants={iconVariants}
                    initial="initial"
                    animate="animate"
                    exit="exit"
                >
                    <XCircle size={20} className="text-white" />
                </motion.div>
                )}
            </AnimatePresence>
            </motion.button>
        <div className="absolute top-12 left-1/2 transform -translate-x-1/2 whitespace-nowrap px-2 py-1 bg-gray-800 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity duration-300">
          Abort All Tasks
        </div>
      </div>

      <div className="relative h-6 w-6 mt-2">
        <AnimatePresence>
          {message && (
            <motion.span
              key="message"
              variants={messageVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              className={`text-xs absolute bottom-5 left-10 font-medium ${
                isError ? "text-red-500" : "text-green-500"
              }`}
            >
              {message}
            </motion.span>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default AbortButton;