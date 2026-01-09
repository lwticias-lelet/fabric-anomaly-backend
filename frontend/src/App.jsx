import React, { useState, useEffect, useRef } from "react";

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [overlay, setOverlay] = useState(null);
  const [loading, setLoading] = useState(false);
  const [hasDefect, setHasDefect] = useState(null);
  const [score, setScore] = useState(null);

  // BACKEND EM PRODUÇÃO
  const backendURL = "https://fabric-anomaly-backend.onrender.com/scan";
  // Para rodar local: const backendURL = "http://127.0.0.1:8000/scan";

  useEffect(() => {
    startCamera();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      alert("Erro ao acessar a câmera: " + error.message);
    }
  };

  const captureFrame = async () => {
    if (!videoRef.current || !canvasRef.current) {
      alert("Câmera não iniciada.");
      return;
    }

    const video = videoRef.current;

    if (video.videoWidth === 0 || video.videoHeight === 0) {
      alert("Aguardando a câmera iniciar. Tente novamente em alguns segundos.");
      return;
    }

    setLoading(true);

    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg")
    );

    if (!blob) {
      alert("Falha ao capturar imagem.");
      setLoading(false);
      return;
    }

    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    try {
      const response = await fetch(backendURL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Erro na comunicação com backend");
      }

      const result = await response.json();

      if (!result.heatmap_base64) {
        throw new Error("Resposta sem heatmap_base64");
      }

      setOverlay(result.heatmap_base64);
      setHasDefect(result.has_defect);
      setScore(result.anomaly_score?.toFixed(4));
    } catch (error) {
      alert("Erro ao processar: " + error.message);
    }

    setLoading(false);
  };

  const renderTitle = () => {
    if (hasDefect === null) return null;
    if (hasDefect) {
      return (
        <h2 style={{ color: "#ff4d4f" }}>
          Defeito detectado 🔍
          {score && <span style={{ fontSize: "0.8rem" }}> (score: {score})</span>}
        </h2>
      );
    }
    return (
      <h2 style={{ color: "#52c41a" }}>
        Nenhum defeito significativo ✅
        {score && <span style={{ fontSize: "0.8rem" }}> (score: {score})</span>}
      </h2>
    );
  };

  return (
    <div className="container">
      <h1>Fabric Anomaly Scanner</h1>

      <video ref={videoRef} autoPlay playsInline className="camera" />

      <button onClick={captureFrame} disabled={loading}>
        {loading ? "Processando..." : "SCANEAR TECIDO"}
      </button>

      <canvas ref={canvasRef} style={{ display: "none" }} />

      {overlay && (
        <div
          className="heatmap-container"
          style={{
            border: hasDefect ? "3px solid red" : "3px solid #52c41a",
            borderRadius: "12px",
            padding: "8px",
            marginTop: "16px",
          }}
        >
          {renderTitle()}
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
