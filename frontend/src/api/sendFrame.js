export async function sendFrame(blob) {
  const form = new FormData();
  form.append("file", blob, "frame.jpg");

  const response = await fetch("https://SEU_BACKEND.onrender.com/scan", {
    method: "POST",
    body: form
  });

  return await response.json();
}

