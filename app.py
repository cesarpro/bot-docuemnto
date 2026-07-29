"""
Backend Flask para el bot de documento público.
Sirve el frontend y expone un endpoint /chat que llama a Gemini de forma segura
(la API key vive solo acá, nunca en el navegador del usuario).

Requisitos:
    pip install flask flask-limiter google-genai pypdf gunicorn

Variables de entorno necesarias:
    GEMINI_API_KEY   -> tu clave de Gemini

Uso local:
    python app.py
    (abrí http://localhost:5000 en el navegador)
"""

import os
import time
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from google import genai
from google.genai import types
from pypdf import PdfReader

MODELO = "gemini-flash-latest"
NOMBRE_PDF = "demanda_12_15.pdf"  # debe estar en la misma carpeta que este archivo

app = Flask(__name__, static_folder="static")

# --- Límite de uso: máximo 10 preguntas por minuto por visitante ---
# Esto protege la cuota gratuita de Gemini de un uso excesivo o abusivo.
limiter = Limiter(get_remote_address, app=app, default_limits=[])

# --- Cliente de Gemini (usa GEMINI_API_KEY del entorno automáticamente) ---
client = genai.Client()


def extraer_texto_pdf(ruta_pdf: str) -> str:
    lector = PdfReader(ruta_pdf)
    return "\n".join((pagina.extract_text() or "") for pagina in lector.pages)


def construir_instruccion_sistema(texto_doc: str) -> str:
    return f"""Sos un asistente que responde EXCLUSIVAMENTE preguntas sobre el siguiente documento.

REGLAS ESTRICTAS:
1. Solo respondé usando información presente en el texto de abajo.
2. Si la pregunta no se puede responder con el contenido, respondé: "Esa información no está en el documento."
3. No uses conocimiento externo, aunque lo sepas.
4. Respondé en español, de forma clara y concisa.

--- CONTENIDO DEL DOCUMENTO ---
{texto_doc}
--- FIN DEL CONTENIDO ---
"""


# Se carga UNA sola vez al iniciar el servidor (no en cada pregunta, para ahorrar tiempo/costo)
print("Cargando documento...")
TEXTO_DOCUMENTO = extraer_texto_pdf(NOMBRE_PDF)
INSTRUCCION_SISTEMA = construir_instruccion_sistema(TEXTO_DOCUMENTO)
print(f"Documento cargado ({len(TEXTO_DOCUMENTO)} caracteres).")

# --- Sesiones de chat en memoria (una por visitante) ---
# Nota: esto se resetea si el servidor reinicia. Para algo más persistente
# habría que guardar el historial en una base de datos.
sesiones = {}


def obtener_chat(session_id: str):
    if session_id not in sesiones:
        sesiones[session_id] = client.chats.create(
            model=MODELO,
            config=types.GenerateContentConfig(system_instruction=INSTRUCCION_SISTEMA),
        )
    return sesiones[session_id]


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/chat", methods=["POST"])
@limiter.limit("10 per minute")
def chat_endpoint():
    datos = request.get_json(force=True)
    mensaje = (datos.get("mensaje") or "").strip()
    session_id = datos.get("session_id") or str(uuid.uuid4())

    if not mensaje:
        return jsonify({"error": "Mensaje vacío"}), 400

    chat = obtener_chat(session_id)

    intentos_maximos = 3
    for intento in range(1, intentos_maximos + 1):
        try:
            respuesta = chat.send_message(mensaje)
            return jsonify({"respuesta": respuesta.text, "session_id": session_id})
        except Exception as e:
            es_saturacion = "503" in str(e) or "UNAVAILABLE" in str(e)
            if es_saturacion and intento < intentos_maximos:
                time.sleep(intento * 2)
            else:
                return jsonify({"error": str(e)}), 503

    return jsonify({"error": "No se pudo procesar la solicitud"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
