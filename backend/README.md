📌 Fabric Anomaly Scanner
🔍 Sistema de Detecção de Defeitos em Tecidos (YOLO + Filtros + Heurísticas)

Desenvolvido por:
Letícia Delfino
Kaline Maria Carvalho

🌐 Demonstração Online
🔹 Backend (FastAPI + YOLO)

👉 https://fabric-anomaly-backend-bd85.vercel.app/

O backend possui os seguintes endpoints funcionando:

GET / → status do servidor

GET /health → teste de saúde

POST /scan → recebe imagem e retorna detecção

🧠 Sobre o Projeto

O Fabric Anomaly Scanner é um sistema de detecção automática de defeitos em superfícies têxteis utilizando:

✔ YOLOv8 para detecção baseada em IA
✔ Filtros de pré-processamento (CLAHE + Bilateral Filter)
✔ Heurística de contornos como fallback quando o modelo está incerto
✔ Retorno de imagem com heatmap e bounding boxes
✔ Detecção robusta para rasgos, manchas e irregularidades

O sistema foi projetado para funcionar tanto em navegadores desktop quanto mobile, acessando a câmera do dispositivo.

📁 Estrutura do Repositório
backend/
 ├── main.py           # Servidor FastAPI principal (em uso)
 ├── models/
 │    └── best.pt      # Modelo YOLO treinado
 ├── requirements.txt  # Dependências do backend
frontend/
 ├── src/
 │    └── App.jsx      # Frontend React
 ├── public/
 └── package.json

⚙️ Como Executar Localmente (Backend)
1) Clonar o repositório
git clone https://github.com/lwticias-lelet/fabric-anomaly-backend
cd backend

2) Criar um ambiente virtual
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

3) Instalar dependências
pip install -r requirements.txt

4) Verificar se o modelo está no local correto
backend/models/best.pt

5) Rodar o servidor local
uvicorn main:app --reload


Servidor disponível em:

👉 http://127.0.0.1:8000

👉 http://127.0.0.1:8000/docs
 (Swagger UI automático)

⚙️ Como Executar Localmente (Frontend)
1) Entrar no diretório
cd frontend

2) Instalar dependências
npm install

3) Criar o arquivo .env
VITE_BACKEND_URL=http://127.0.0.1:8000/scan

4) Rodar o frontend
npm run dev


Acesse:

👉 http://localhost:5173/

📌 Fluxo de Funcionamento

1️⃣ Frontend captura a imagem da câmera
2️⃣ Converte para JPEG e envia para /scan
3️⃣ Backend aplica filtros + YOLO
4️⃣ Se YOLO falhar, heurística tenta detectar padrões grandes
5️⃣ Backend retorna:

Imagem anotada (image_base64)

Maior confiança (confidence)

Origem da detecção (source)

Booleano has_defect

Mensagem amigável

6️⃣ Frontend exibe:

Heatmap/caixa vermelha

Mensagem colorida

Status de defeito ou não