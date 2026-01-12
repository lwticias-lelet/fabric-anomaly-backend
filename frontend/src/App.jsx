import React, { useState, useEffect, useRef } from "react";

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [resultImage, setResultImage] = useState(null);
  const [loading, setLoading] = useState(false);

  // NOVO: estados para mensagem
  const [statusMessage, setStatusMessage] = useState("");
  const [hasDefect, setHasDefect] = useState(null);

  const backendBase = "https://fabric-anomaly-backend.onrender.com";
  const backendURL = `${backendBase}/scan`;

  useEffect(() => {
    startCamera();
    // opcional: testar conexão com backend ao carregar
    testBackendConnection();
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
      alert("Erro ao acessar câmera: " + error.message);
    }
  };

  const testBackendConnection = async () => {
    try {
      const res = await fetch(`${backendBase}/health`);
      if (!res.ok) {
        console.warn("Backend respondeu com status", res.status);
      } else {
        const data = await res.json();
        console.log("Health check:", data);
      }
    } catch (e) {
      console.warn("Não foi possível conectar ao backend (health):", e);
    }
  };

  // Função de fetch com timeout (útil pra celular)
  const fetchWithTimeout = (url, options = {}, timeoutMs = 20000) => {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeoutMs);
    return fetch(url, {
      ...options,
      signal: controller.signal,
    }).finally(() => clearTimeout(id));
  };

  const captureFrame = async () => {
    setLoading(true);
    setStatusMessage("");
    setHasDefect(null);

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas) {
      alert("Câmera não inicializada corretamente.");
      setLoading(false);
      return;
    }

    // 🔴 Redimensionar frame para ~640px de largura
    const targetWidth = 640;
    const scale = targetWidth / video.videoWidth;

    canvas.width = targetWidth;
    canvas.height = video.videoHeight * scale;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.8)
    );

    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    try {
      const response = await fetchWithTimeout(backendURL, {
        method: "POST",
        body: formData,
      }, 25000); // 25s de timeout

      if (!response.ok) {
        throw new Error("Erro na resposta do servidor: " + response.status);
      }

      const data = await response.json();
      console.log("Resposta do backend:", data);

      if (data.image_base64) {
        setResultImage(`data:image/jpeg;base64,${data.image_base64}`);
      }

      if (typeof data.has_defect === "boolean") {
        setHasDefect(data.has_defect);
      }

      if (data.message) {
        setStatusMessage(data.message);
      } else if (typeof data.has_defect === "boolean") {
        setStatusMessage(
          data.has_defect
            ? "Defeito detectado no tecido."
            : "Nenhum defeito detectado."
        );
      }
    } catch (err) {
      console.error("Erro ao enviar imagem:", err);
      if (err.name === "AbortError") {
        alert("Tempo esgotado ao comunicar com o servidor.");
      } else {
        alert("Erro ao enviar imagem: " + err.message);
      }
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
          {statusMessage && (
            <p
              className={
                hasDefect === true
                  ? "status-text status-defect"
                  : hasDefect === false
                  ? "status-text status-ok"
                  : "status-text"
              }
            >
              {statusMessage}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
