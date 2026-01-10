import React, { useEffect, useRef, useState } from "react";
import "./styles.css";
 

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const [resultImage, setResultImage] = useState(null);
  const [loading, setLoading] = useState(false);

  // =========================
  // Wake-up do backend (NÃO AFETA ESTÉTICA)
  // =========================
  useEffect(() => {
    fetch(BACKEND_URL).catch(() => {});
    startCamera();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });
      videoRef.current.srcObject = stream;
    } catch (err) {
      alert("Erro ao acessar a câmera");
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
    formData.append("file", blob, "image.jpg");

    try {
      const response = await fetch(`${BACKEND_URL}/scan`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResultImage(`data:image/png;base64,${data.heatmap_base64}`);
    } catch (error) {
      alert("Erro ao comunicar com o backend");
    }

    setLoading(false);
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Surface Anomaly Scanner</h1>
      </header>

      <main className="main-content">
        <div className="camera-container">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="camera-view"
          />
        </div>

        <button
          className="capture-btn"
          onClick={captureFrame}
          disabled={loading}
        >
          {loading ? "Processando..." : "Detectar Defeito"}
        </button>

        <canvas ref={canvasRef} className="hidden-canvas" />

        {resultImage && (
          <div className="result-container">
            <img
              src={resultImage}
              alt="Resultado da detecção"
              className="result-image"
            />
          </div>
        )}
      </main>
    </div>
  );
}
