import { ApiTokenInput } from './ui/ApiToken';
import FusionParamsEditor from './ui/FusionParamsEditor';
import { HealthStatusIndicator } from './ui/HealthStatusIndicator';
import ModelSettings from './ui/ModelSettings';
import { motion } from 'framer-motion';

export default function Dashboard() {
  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15 // Staggers the animation of each child element
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } }
  };

  return (
    <section className="mx-auto h-full w-full max-w-screen-2xl px-6 py-8 min-h-[calc(100vh-64px)] flex flex-col gap-8">
      {/* Header with simple motion */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center"
      >
        <h1 className="uppercase text-3xl font-bold text-primary">Dashboard</h1>
        <p className="text-muted-foreground mt-2">Manage your API settings, server health, and model configurations.</p>
      </motion.header>

      {/* Main Grid Layout with staggered animation */}
      <motion.div
        className="grid grid-cols-1 lg:grid-cols-3 gap-6"
        variants={containerVariants}
        initial="hidden"
        animate="show"
      >
        {/* Left Column: Model and Fusion Settings */}
        <div className="lg:col-span-2  flex flex-col gap-6">
          <motion.div variants={itemVariants} className="p-6 border rounded-xl shadow-sm bg-card">
            <h2 className="text-lg font-semibold text-foreground mb-4">Model Settings</h2>
            <ModelSettings />
          </motion.div>
          <motion.div variants={itemVariants} className="p-6 border rounded-xl shadow-sm bg-card">
            <h2 className="text-lg font-semibold text-foreground mb-4">Fusion Parameters</h2>
            <FusionParamsEditor />
          </motion.div>
        </div>

        {/* Right Column: Auth and Health */}
        <div className="lg:col-span-1 flex flex-col gap-6">
          <motion.div variants={itemVariants} className="p-6 border rounded-xl shadow-sm bg-card">
            <h2 className="text-lg font-semibold text-foreground mb-2">Authentication</h2>
            <p className="text-sm text-muted-foreground mb-4">Secure your dashboard with an API token.</p>
            <ApiTokenInput />
          </motion.div>
          <motion.div variants={itemVariants} className="p-6 border rounded-xl shadow-sm bg-card">
            <h2 className="text-lg font-semibold text-foreground mb-2">Server Health</h2>
            <p className="text-sm text-muted-foreground mb-4">Check the status of the server.</p>
            <HealthStatusIndicator interval={12000} />
          </motion.div>
        </div>
      </motion.div>
    </section>
  );
}