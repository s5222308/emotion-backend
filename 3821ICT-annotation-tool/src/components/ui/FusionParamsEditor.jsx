import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { RotateCwIcon } from "lucide-react";
import Tooltip from "./Tooltip";

const FusionParamsEditor = () => {
    const [params, setParams] = useState(null);
    const [paramLimits, setParamLimits] = useState(null);
    
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [status, setStatus] = useState("");

    useEffect(() => {
        const fetchParams = async () => {
            setLoading(true);
            setStatus("Fetching fusion parameters...");
            try {
                const res = await fetch("http://localhost:9090/get-fusion_params");
                if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
                const data = await res.json();
                
                console.log(data.params)
                console.log(data.limits)
                setParams(data.params);
                setParamLimits(data.limits);

                setStatus("Parameters loaded successfully.");
            } catch (err) {
                console.error("Failed to fetch fusion params", err);
                setStatus("Error: Failed to fetch parameters.");
            } finally {
                setLoading(false);
            }
        };
        fetchParams();
    }, []);

    const handleChange = (key, value) => {
        setParams((prev) => ({ ...prev, [key]: value }));
    };

    const handleReset = () => {
        setParams({
            audio_bias: 1.0,
            video_bias: 1.0,
            libreface_bias: 1.0,
            beta: 0.8,
            floor_prob: 1e-6,
            fps: 30,
            min_duration: 0.5,
            debug: false
        });
    };

    const handleSave = async () => {
        setSaving(true);
        setStatus("Saving changes...");
        try {
            const res = await fetch("http://localhost:9090/update-fusion_params", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(params),
            });
            if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
            setStatus("Saved successfully!");
        } catch (err) {
            console.error("Failed to update fusion params", err);
            setStatus("Error: Failed to save changes.");
        } finally {
            setSaving(false);
        }
    };

    const containerVariants = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: {
                staggerChildren: 0.1,
            },
        },
    };

    const itemVariants = {
        hidden: { opacity: 0, y: 20 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" } },
    };

    return (
        <div className="p-4 relative bg-accent rounded-xl text-foreground w-full flex flex-col gap-6">
            <h2 className="text-xl font-semibold mb-2">Fusion Parameters</h2>
            <button
                onClick={handleReset}
                disabled={saving}
                className={`absolute flex gap-2 right-0 items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 h-10 px-4 py-2 text-secondary-foreground hover:bg-secondary/90
                    ${saving ? 'cursor-not-allowed' : ''}
                `}
            >
                <RotateCwIcon/>
                <Tooltip text={"Please save after resetting"}></Tooltip>
            </button>
            <AnimatePresence mode="wait">
                {loading && (
                    <motion.p
                        key="loading"
                        className="text-gray-400"
                        initial="hidden"
                        animate="visible"
                        exit="hidden"
                        variants={itemVariants}
                    >
                        {status}
                    </motion.p>
                )}
                {!loading && (!params || !paramLimits) && (
                    <motion.p
                        key="no-params"
                        className="text-red-500"
                        initial="hidden"
                        animate="visible"
                        exit="hidden"
                        variants={itemVariants}
                    >
                        No fusion parameters loaded
                    </motion.p>
                )}
                {!loading && params && paramLimits && (
                    <motion.div
                        key="content"
                        initial="hidden"
                        animate="visible"
                        variants={containerVariants}
                    >
                        <motion.div
                            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
                            variants={containerVariants}
                        >
                            <NumberInput
                                title="Beta"
                                toolTipText="Smoothing factor for video frame probabilities. Higher values make video labels more stable over time."
                                param="beta"
                                value={params.beta}
                                step={0.01}
                                min={paramLimits.beta[0]}
                                max={paramLimits.beta[1]}
                                onChange={handleChange}
                                isSaving={saving}
                            />
                            <NumberInput
                                title="Min Duration (s)"
                                toolTipText="Minimum time a label must persist to be allowed to change. Helps reduce flickering."
                                param="min_duration"
                                value={params.min_duration}
                                step={0.1}
                                min={paramLimits.min_duration[0]}
                                max={paramLimits.min_duration[1]}
                                onChange={handleChange}
                                isSaving={saving}
                            />
                            <NumberInput
                                title="FPS"
                                toolTipText="Frames per second for video analysis."
                                param="fps"
                                value={params.fps}
                                step={1}
                                min={paramLimits.fps[0]}
                                max={paramLimits.fps[1]}
                                onChange={handleChange}
                                isSaving={saving}
                            />
                            <NumberInput
                                title="Video Bias"
                                toolTipText="Weight of YOLO video emotion in fusion. Higher values trust video predictions more."
                                param="video_bias"
                                value={params.video_bias}
                                step={0.1}
                                min={paramLimits.video_bias[0]}
                                max={paramLimits.video_bias[1]}
                                onChange={handleChange}
                                isSaving={saving}
                            />
                            <NumberInput
                                title="Audio Bias"
                                toolTipText="Weight of audio emotion in fusion. Higher values trust audio predictions more."
                                param="audio_bias"
                                value={params.audio_bias}
                                step={0.1}
                                min={paramLimits.audio_bias[0]}
                                max={paramLimits.audio_bias[1]}
                                onChange={handleChange}
                                isSaving={saving}
                            />
                            <NumberInput
                                title="LibreFace Bias"
                                toolTipText="Weight of LibreFace FER in fusion. Higher values trust LibreFace emotion predictions more."
                                param="libreface_bias"
                                value={params.libreface_bias}
                                step={0.1}
                                min={paramLimits.libreface_bias[0]}
                                max={paramLimits.libreface_bias[1]}
                                onChange={handleChange}
                                isSaving={saving}
                            />
                        </motion.div>

                        <motion.div
                            variants={itemVariants}
                            className="flex flex-col md:flex-row md:items-center justify-between gap-4 mt-6"
                        >
                            <NumberInput
                                title="Floor Prob"
                                toolTipText="Minimum probability value to avoid numerical instability."
                                param="floor_prob"
                                value={params.floor_prob}
                                step={1e-12}
                                min={paramLimits.floor_prob[0]}
                                max={paramLimits.floor_prob[1]}
                                onChange={handleChange}
                                isSaving={saving}
                                isRange={false}
                            />
                            <div className="flex items-center space-x-3 mt-8 md:mt-0">
                                <input
                                    type="checkbox"
                                    checked={params.debug}
                                    onChange={(e) => handleChange("debug", e.target.checked)}
                                    className="h-4 w-4"
                                    disabled={saving}
                                />
                                <label className="text-sm font-medium text-gray-400">
                                    Debug Mode
                                </label>
                            </div>
                        </motion.div>

                        <motion.button
                            variants={itemVariants}
                            onClick={handleSave}
                            disabled={saving}
                            className={`inline-flex mt-4 items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 h-10 px-4 py-2 w-full bg-primary text-primary-foreground hover:bg-primary/90
                                ${saving ? 'bg-gray-500 cursor-not-allowed' : ' shadow-lg'}
                            `}
                            whileHover={{ scale: saving ? 1 : 1.01 }}
                            whileTap={{ scale: saving ? 1 : 0.98 }}
                        >
                            {saving ? "Saving..." : "Save"}
                        </motion.button>
                        <AnimatePresence>
                            {status && (
                                <motion.p
                                    key="status-message"
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -10 }}
                                    transition={{ duration: 0.3 }}
                                    className={`mt-4 text-sm font-medium ${status.includes("Error") ? "text-red-500" : "text-green-500"}`}
                                >
                                    {status}
                                </motion.p>
                            )}
                        </AnimatePresence>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default FusionParamsEditor;

const NumberInput = ({
    title,
    param,
    value,
    step,
    min,
    max,
    onChange,
    isSaving,
    toolTipText,
    isRange = true,
}) => {
    const variants = {
        hidden: { opacity: 0, y: 20 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" } },
    };

    return (
        <motion.div className="flex flex-col w-full" variants={variants}>
            <div className="flex items-center justify-between mb-1">
                <label className="text-sm font-medium text-gray-400">{title}</label>
                {toolTipText && <Tooltip text={toolTipText} />}
            </div>
            <div className="flex w-full flex-1 items-center gap-4">
                {isRange && (
                    <input
                        type="range"
                        step={step}
                        min={min}
                        max={max}
                        value={value}
                        onChange={(e) => onChange(param, parseFloat(e.target.value))}
                        className="flex-1 w-full cursor-pointer accent-blue-500"
                        disabled={isSaving}
                    />
                )}
                <input
                    type="number"
                    step={step}
                    min={min}
                    max={max}
                    value={value}
                    onChange={(e) => onChange(param, parseFloat(e.target.value))}
                    className={`w-20 px-2 py-1 border border-gray-600 rounded-md bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-blue-500 ${!isRange && "w-full"}`}
                    disabled={isSaving}
                />
            </div>
        </motion.div>
    );
};