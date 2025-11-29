import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Tooltip from './Tooltip';
import { RotateCwIcon } from 'lucide-react';

const getProcessingModeTooltip = (mode) => {
    switch (mode) {
        case 'sparse':
            return "Sparse Sampling: analyzes every Nth frame only. Fastest, but may miss quick micro-changes.";
        case 'temporal_context':
            return "Temporal Context: samples frames but analyzes a temporal window around each one. Best speed/accuracy balance.";
        case 'dense_window':
            return "Dense Window: scans almost every frame in overlapping windows. Most detailed, but slowest.";
        default:
            return "Choose how video frames are sampled and grouped over time for emotion analysis.";
    }
};

const ModelSettings = () => {
    const [availableModels, setAvailableModels] = useState({
        audio_emotion: [],
        face: [],
        face_emotion: [],
    });
    const [currentModels, setCurrentModels] = useState({
        audio_emotion: '',
        face: '',
        face_emotion: '',
    });
    const [parameters, setParameters] = useState({});
    const [prevFrameStep, setPrevFrameStep] = useState(null);

    const [loading, setLoading] = useState(true);
    const [isSaving, setIsSaving] = useState(false);
    const [status, setStatus] = useState('');

    useEffect(() => {
        const fetchModels = async () => {
            setLoading(true);
            setStatus('Fetching models...');
            try {
                const response = await fetch('http://localhost:9090/get_models');
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                const apiResponse = await response.json();
                setAvailableModels(apiResponse.available);
                setCurrentModels({
                    audio_emotion: apiResponse.current.audio_emotion,
                    face: apiResponse.current.face,
                    face_emotion: apiResponse.current.face_emotion,
                });
                setParameters(apiResponse.current.parameters || {});
                setStatus('Models and parameters loaded successfully.');
            } catch (error) {
                console.error('Failed to fetch models:', error);
                setStatus('Error: Failed to fetch models.');
            } finally {
                setLoading(false);
            }
        };

        fetchModels();
    }, []);

    const handleModelChange = (modelType, value) => {
        setCurrentModels(prev => ({ ...prev, [modelType]: value }));
    };

    const handleParamChange = (param, value) => {
        setParameters(prev => {
            let updated = { ...prev, [param]: value };

            // Legacy dense_window toggle -> keep this
            if (param === 'dense_window') {
                updated.processing_mode = value ? 'dense_window' : 'sparse';
            }

            // When video processing mode changes
            if (param === 'processing_mode') {
                const oldMode = prev.processing_mode ?? 'sparse';

                if (value === 'dense_window') {
                    // Just entered dense_window: remember old frame_step once
                    if (oldMode !== 'dense_window' && prevFrameStep === null) {
                        setPrevFrameStep(prev.frame_step ?? 10);
                    }
                    // Force frame_step = 1 for dense mode
                    updated.frame_step = 1;
                } else {
                    // Leaving dense_window -> restore previous frame_step if we have it
                    if (oldMode === 'dense_window') {
                        updated.frame_step = prevFrameStep ?? prev.frame_step ?? 10;
                        setPrevFrameStep(null);
                    }
                }
            }

            return updated;
        });
    };

    const handleReset = () => {
        setParameters({
            emotion_conf: 0.5,
            face_conf: 0.75,
            frame_step: 10,
            segment_duration: 0.5,
            use_openface: false,
            use_libreface: false,
            processing_mode: 'sparse',
            temporal_context_window: 0.5,
            dense_window: false,
            window_size: 1.0,
            audio_sliding_window: false,
            audio_window_size: 1.0,
            audio_window_overlap: 0.5
        });
    };

    const handleSave = async () => {
        setIsSaving(true);
        setStatus('Saving models and parameters...');
        try {
            const response = await fetch('http://localhost:9090/set_models', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ...currentModels,
                    parameters: parameters,
                }),
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            setStatus('Saved successfully!');
        } catch (error) {
            console.error('Failed to save models/params:', error);
            setStatus('Error: Failed to save models/params.');
        } finally {
            setIsSaving(false);
        }
    };
    
    // Framer Motion variants for animated sections
    const variants = {
        hidden: { opacity: 0, y: 20 },
        visible: { opacity: 1, y: 0 }
    };

    return (
        <div className="p-4 bg-accent relative rounded-xl text-foreground w-full flex flex-col gap-6">
            <h2 className="text-xl font-semibold mb-4">Model & Parameter Settings</h2>
                <button
                onClick={handleReset}
                disabled={isSaving}
                className={`absolute flex gap-2 right-0 items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 h-10 px-4 py-2 text-secondary-foreground hover:bg-secondary/90
                    ${isSaving ? 'cursor-not-allowed' : ''}
                `}
                whileHover={{ scale: isSaving ? 1 : 1.01 }}
                whileTap={{ scale: isSaving ? 1 : 0.98 }}
            >
                <RotateCwIcon/>
                <Tooltip text={"Please save after resetting"}></Tooltip>
                </button>
            <AnimatePresence mode="wait">
                {loading ? (
                    <motion.div
                        key="loading"
                        className="text-gray-400"
                        initial="hidden"
                        animate="visible"
                        exit="hidden"
                        variants={variants}
                    >
                        {status}
                    </motion.div>
                ) : (
                    <motion.div
                        key="content"
                        initial="hidden"
                        animate="visible"
                        exit="hidden"
                        variants={{
                            hidden: { opacity: 0 },
                            visible: {
                                opacity: 1,
                                transition: {
                                    staggerChildren: 0.1
                                }
                            }
                        }}
                    >

                        {/* Model dropdowns */}
                        <motion.div variants={variants} className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <Dropdown
                                title="Audio Emotion Model"
                                modelType="audio_emotion"
                                options={availableModels.audio_emotion}
                                value={currentModels.audio_emotion}
                                onChange={handleModelChange}
                                isSaving={isSaving}
                            />
                            <Dropdown
                                title="Face Model"
                                modelType="face"
                                options={availableModels.face}
                                value={currentModels.face}
                                onChange={handleModelChange}
                                isSaving={isSaving}
                            />
                            <Dropdown
                                title="Face Emotion Model"
                                modelType="face_emotion"
                                options={availableModels.face_emotion}
                                value={currentModels.face_emotion}
                                onChange={handleModelChange}
                                isSaving={isSaving}
                            />
                        </motion.div>

                        {/* Parameter sliders */}
                        <motion.div variants={variants} className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                            <NumberInput
                                title="Frame Step"
                                toolTipText="Interval between frames to analyze. Higher values reduce computation but may miss rapid changes."
                                param="frame_step"
                                value={parameters.frame_step ?? 1}
                                step={1}
                                min={1}
                                max={100}
                                onChange={handleParamChange}
                                isSaving={isSaving || parameters.processing_mode === 'dense_window'}
                            />
                            <NumberInput
                                title="Face Confidence"
                                toolTipText="Confidence threshold for face detection. Higher values increase precision but may miss faces."
                                param="face_conf"
                                value={parameters.face_conf ?? 0.75}
                                step={0.05}
                                min={0.1}
                                max={1.0}
                                onChange={handleParamChange}
                                isSaving={isSaving}
                            />
                            <NumberInput
                                title="Emotion Confidence"
                                toolTipText="Confidence threshold for emotion recognition. Higher values increase precision but may miss subtle emotions."
                                param="emotion_conf"
                                value={parameters.emotion_conf ?? 0.75}
                                step={0.05}
                                min={0.1}
                                max={1.0}
                                onChange={handleParamChange}
                                isSaving={isSaving}
                            />
                        </motion.div>
                        
                        {/* Segment Duration */}
                        <motion.div variants={variants} className="mt-6">
                            <NumberInput
                                title="Segment Duration (s)"
                                toolTipText="Duration of audio segments to analyze (used when Audio Sliding Windows is disabled). Shorter durations may capture quick changes but can be less accurate."
                                param="segment_duration"
                                value={parameters.segment_duration ?? 0.5}
                                step={0.1}
                                min={0.1}
                                max={10.0}
                                onChange={handleParamChange}
                                isSaving={isSaving || parameters.audio_sliding_window}
                            />
                        </motion.div>

                        {/* Video Processing Parameters */}
                        <motion.div variants={variants} className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
                            <div className="md:col-span-2">
                                <div className="flex items-center justify-between mb-1">
                                    <label className="text-sm font-medium text-gray-400">Video Processing Mode</label>
                                    <Tooltip text={getProcessingModeTooltip(parameters.processing_mode ?? 'sparse')} />
                                </div>
                                <select
                                    value={parameters.processing_mode ?? 'sparse'}
                                    onChange={(e) => handleParamChange('processing_mode', e.target.value)}
                                    className="w-full py-2 px-3 transition-all duration-300 hover:bg-popover border border-gray-600 rounded-md bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    disabled={isSaving}
                                >
                                    <option value="sparse">Sparse Sampling (Fastest)</option>
                                    <option value="temporal_context">Temporal Context (Slow)</option>
                                    <option value="dense_window">Dense Window (Slow)</option>
                                </select>
                            </div>

                            <div className="md:col-span-2">
                                <NumberInput
                                    title="Temporal Context Window (s)"
                                    toolTipText="Size of temporal context window analyzed around each sampled frame"
                                    param="temporal_context_window"
                                    value={parameters.temporal_context_window ?? 0.5}
                                    step={0.1}
                                    min={0.1}
                                    max={2.0}
                                    onChange={handleParamChange}
                                    isSaving={isSaving || parameters.processing_mode !== 'temporal_context'}
                                />
                            </div>

                            <div className="md:col-span-2">
                                <NumberInput
                                    title="Dense Window Size (s)"
                                    toolTipText="Size of sliding window for dense processing"
                                    param="window_size"
                                    value={parameters.window_size ?? 1.0}
                                    step={0.1}
                                    min={0.1}
                                    max={5.0}
                                    onChange={handleParamChange}
                                    isSaving={isSaving || parameters.processing_mode !== 'dense_window'}
                                />
                            </div>
                        </motion.div>
                        
                        {/* Audio Window Parameters */}
                        <motion.div variants={variants} className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
                            <NumberInput
                                title="Audio Window Size (s)"
                                toolTipText="Size of audio sliding window (similar to Segment Duration, but with overlap for better temporal alignment)"
                                param="audio_window_size"
                                value={parameters.audio_window_size ?? 1.0}
                                step={0.1}
                                min={0.1}
                                max={5.0}
                                onChange={handleParamChange}
                                isSaving={isSaving || !parameters.audio_sliding_window}
                            />
                            <NumberInput
                                title="Audio Window Overlap"
                                toolTipText="Overlap ratio for audio sliding windows (0.5 = 50% overlap)"
                                param="audio_window_overlap"
                                value={parameters.audio_window_overlap ?? 0.5}
                                step={0.05}
                                min={0.0}
                                max={0.99}
                                onChange={handleParamChange}
                                isSaving={isSaving || !parameters.audio_sliding_window}
                            />
                        </motion.div>

                        {/* Feature Toggles */}
                        <motion.div variants={variants} className="flex flex-wrap gap-6 mt-6">
                            {/* OpenFace AU/Gaze */}
                            <div className="flex items-center space-x-3">
                                <input
                                    type="checkbox"
                                    checked={parameters.use_openface ?? false}
                                    onChange={(e) => handleParamChange("use_openface", e.target.checked)}
                                    className="h-4 w-4"
                                    disabled={isSaving}
                                />
                                <label className="text-sm font-medium text-gray-400">
                                    Use OpenFace (AUs + Gaze)
                                </label>
                                <Tooltip text="Runs the OpenFace AU + gaze pipeline; enables triple fusion when combined with audio and video." />
                            </div>

                            {/* LibreFace AU/Expression */}
                            <div className="flex items-center space-x-3">
                                <input
                                    type="checkbox"
                                    checked={parameters.use_libreface ?? false}
                                    onChange={(e) => handleParamChange("use_libreface", e.target.checked)}
                                    className="h-4 w-4"
                                    disabled={isSaving}
                                />
                                <label className="text-sm font-medium text-gray-400">
                                    Use LibreFace (AUs & Expression)
                                </label>
                                <Tooltip text="Runs the LibreFace AU + emotion pipeline; also available for fusion." />
                            </div>

                            <div className="flex items-center space-x-3">
                                <input
                                    type="checkbox"
                                    checked={parameters.audio_sliding_window ?? false}
                                    onChange={(e) => handleParamChange("audio_sliding_window", e.target.checked)}
                                    className="h-4 w-4"
                                    disabled={isSaving}
                                />
                                <label className="text-sm font-medium text-gray-400">
                                    Audio Sliding Windows
                                </label>
                                <Tooltip text="Uses overlapping windows for better temporal alignment between audio and video" />
                            </div>
                        </motion.div>

                        {/* Save Button */}
                        <motion.button
                            variants={variants}
                            type='button'
                            onClick={handleSave}
                            disabled={isSaving}
                            className={`inline-flex mt-4 items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 h-10 px-4 py-2 w-full bg-primary text-primary-foreground hover:bg-primary/90
                                ${isSaving ? 'cursor-not-allowed' : 'shadow-lg'}
                            `}
                            whileHover={{ scale: isSaving ? 1 : 1.01 }}
                            whileTap={{ scale: isSaving ? 1 : 0.98 }}
                        >
                            {isSaving ? 'Saving...' : 'Save'}
                        </motion.button>

                    

                        {/* Status Message */}
                        <AnimatePresence>
                            {status && (
                                <motion.p
                                    key="status-message"
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -10 }}
                                    transition={{ duration: 0.3 }}
                                    className={`mt-4 text-sm font-medium ${status.includes('Error') ? 'text-red-500' : 'text-green-500'}`}
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

export default ModelSettings;

const Dropdown = ({ title, modelType, options, value, onChange, isSaving }) => {
    const variants = {
        hidden: { opacity: 0, y: 20 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } }
    };

    return (
        <motion.div 
            className="flex flex-col"
            variants={variants}
        >
            <label className="text-sm font-medium text-gray-400 mb-1">{title}</label>
            <select
                value={value}
                onChange={(e) => onChange(modelType, e.target.value)}
                className="w-full py-2 px-3 transition-all duration-300 hover:bg-popover border border-gray-600 rounded-md bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={isSaving}
            >
                {options.map(opt => (
                    <option key={opt} value={opt}>{opt}</option>
                ))}
            </select>
        </motion.div>
    );
};

// NumberInput component
const NumberInput = ({ title, param, value, step, min, max, onChange, isSaving, toolTipText }) => {
    const variants = {
        hidden: { opacity: 0, y: 20 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } }
    };

    return (
        <motion.div 
            className="flex flex-col w-full"
            variants={variants}
        >
            <div className="flex items-center justify-between mb-1">
                <label className="text-sm font-medium text-gray-400">{title}</label>
                {toolTipText && <Tooltip text={toolTipText} />}
            </div>
            <div className="flex flex-1 w-full items-center gap-4">
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
                <input
                    type="number"
                    step={step}
                    min={min}
                    max={max}
                    value={value}
                    onChange={(e) => onChange(param, parseFloat(e.target.value))}
                    className="w-20 px-2 py-1 border border-gray-600 rounded-md bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-blue-500"
                    disabled={isSaving}
                />
            </div>
        </motion.div>
    );
};