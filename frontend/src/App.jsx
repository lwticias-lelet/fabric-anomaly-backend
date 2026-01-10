import React, { useState, useEffect, useRef } from "react";

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [resultImage, setResultImage] = useState(null);
  const [loading, setLoading] = useState(false);

  const backendURL = "https://fabric-anomaly-backend.onrender.com/scan";

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
      alert("Erro ao acessar câmera: " + error.message);
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

      const data = await response.json();

      if (data.image_base64) {
        setResultImage(`data:image/jpeg;base64,${data.image_base64}`);
      }
    } catch (err) {
      alert("Erro ao enviar imagem: " + err.message);
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

      <canvas ref={canvasRef} style={{ display: "none" }} />

      {resultImage && (
        <div className="heatmap-container">
          <h2>Resultado</h2>
          <img src={resultImage} alt="Resultado" className="heatmap" />
        </div>
      )}
    </div>
  );
}
