export default function HeatmapOverlay({ heatmap }) {
  if (!heatmap) return null;
  return (
    <img
      src={`data:image/jpeg;base64,${heatmap}`}
      className="heatmap-overlay"
    />
  );
}

