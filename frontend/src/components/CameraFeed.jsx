import { useEffect, useRef } from "react";

export default function CameraFeed({ onFrame }) {
  const videoRef = useRef();

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => { videoRef.current.srcObject = stream; });

    const interval = setInterval(() => {
      if (!videoRef.current) return;
      const canvas = document.createElement("canvas");
      canvas.width = 192;
      canvas.height = 192;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(videoRef.current, 0, 0, 192, 192);
      canvas.toBlob(blob => onFrame(blob), "image/jpeg");
    }, 500);

    return () => clearInterval(interval);
  }, []);

  return <video ref={videoRef} autoPlay className="camera" />;
}

