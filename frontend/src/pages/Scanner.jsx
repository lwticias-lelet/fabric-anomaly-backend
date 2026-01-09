import { useState } from "react";
import CameraFeed from "../components/CameraFeed";
import HeatmapOverlay from "../components/HeatmapOverlay";
import { sendFrame } from "../api/sendFrame";

export default function Scanner() {
  const [heatmap, setHeatmap] = useState(null);

  async function handleFrame(blob) {
    const result = await sendFrame(blob);
    setHeatmap(result.heatmap);
  }

  return (
    <div className="scanner-container">
      <CameraFeed onFrame={handleFrame} />
      <HeatmapOverlay heatmap={heatmap} />
    </div>
  );
}

