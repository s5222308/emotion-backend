import { useState } from "react";
import { Info } from "lucide-react";

const Tooltip = ({ text }) => {
  const [visible, setVisible] = useState(false);

  return (
    <div
      className="relative inline-flex items-center"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      <Info className="w-4 h-4 text-gray-400 cursor-pointer" />

      {visible && (
        <div className="absolute left-6 top-1/2 -translate-y-1/2 z-10 bg-tooltip text-gray-300 text-md px-3 py-2 font-semibold rounded shadow-lg break-words max-w-xs w-75">
          {text}
        </div>
      )}
    </div>
  );
};

export default Tooltip;
