import React, { useState, useEffect, useRef } from "react";

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [overlay, setOverlay] = useState(null);
  const [loading, setLoading] = useState(false);

  const backendURL = "http://127.0.0.1:8000/scan";

  useEffect(() => {
    startCamera();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false
      });
      videoRef.current.srcObject = stream;
    } catch (error) {
      alert("Erro ao acessar a câmera: " + error.message);
    }
  };

  const captureFrame = async () => {
    setLoading(true);

    const video = videoRef.current;
    const canvas = canvasRef.current;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg")
    );

    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    try {
      const response = await fetch(backendURL, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error("Erro na comunicação com backend");
      }

      const result = await response.json();
      setOverlay(result.heatmap_base64);
    } catch (error) {
      alert("Erro ao processar: " + error.message);
    }

    setLoading(false);
  };

  return (
    <div className="container">
      <h1>Fabric Anomaly Scanner</h1>

      <video ref={videoRef} autoPlay playsInline className="camera" />

      <button onClick={captureFrame} disabled={loading}>
        {loading ? "Processando..." : "SCANEAR TECIDO"}
      </button>

      <canvas ref={canvasRef} style={{ display: "none" }}></canvas>

      {overlay && (
        <div className="heatmap-container">
          <h2>Defeito detectado 🔍</h2>
          <img
            src={`data:image/png;base64,${overlay}`}
            alt="heatmap"
            className="heatmap"
          />
        </div>
      )}
    </div>
  );
}
