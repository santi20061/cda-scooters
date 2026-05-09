# CDA Scooters — Arquitectura de Microservicios

## Servicios

| Servicio             | Puerto | Descripción                              |
|----------------------|--------|------------------------------------------|
| api-gateway          | 3000   | Punto de entrada único, proxy a servicios|
| contact-service      | 3001   | Formulario de contacto (email + WA)      |
| appointment-service  | 3002   | CRUD de citas (MongoDB)                  |
| chatbot-service      | 3003   | Respuestas automáticas por intents       |
| whatsapp-service     | 3004   | Cliente WhatsApp singleton (QR)          |
| MongoDB              | 27017  | Base de datos de citas                   |

## Flujo de comunicación

```
Cliente HTTP
    │
    ▼
API Gateway :3000
    ├── /api/contact      ──→  contact-service :3001
    ├── /api/appointments ──→  appointment-service :3002
    └── /api/chatbot      ──→  chatbot-service :3003

contact-service     ──→  whatsapp-service :3004  (HTTP POST /send)
appointment-service ──→  whatsapp-service :3004  (HTTP POST /send)
appointment-service ──→  MongoDB :27017
```

## Inicio rápido (desarrollo local)

### 1. Instalar dependencias

```bash
cd backend/gateway            && npm install && cd ../..
cd backend/services/whatsapp-service  && npm install && cd ../../..
cd backend/services/contact-service   && npm install && cd ../../..
cd backend/services/appointment-service && npm install && cd ../../..
cd backend/services/chatbot-service   && npm install && cd ../../..
```

### 2. Configurar variables de entorno

Copia cada `.env.example` como `.env` y completa los valores:

```bash
cp backend/services/contact-service/.env.example     backend/services/contact-service/.env
cp backend/services/appointment-service/.env.example  backend/services/appointment-service/.env
cp backend/services/chatbot-service/.env.example      backend/services/chatbot-service/.env
cp backend/services/whatsapp-service/.env.example     backend/services/whatsapp-service/.env
cp backend/gateway/.env.example                        backend/gateway/.env
```

### 3. Arrancar servicios (terminales separadas)

```bash
# Terminal 1 — WhatsApp (escanea el QR que aparece)
cd backend/services/whatsapp-service && npm start

# Terminal 2
cd backend/services/contact-service && npm start

# Terminal 3
cd backend/services/appointment-service && npm start

# Terminal 4
cd backend/services/chatbot-service && npm start

# Terminal 5 — Gateway
cd backend/gateway && npm start
```

## Inicio con Docker Compose

```bash
cd backend
docker compose up --build
```

> **WhatsApp QR**: la primera vez debes ver los logs del contenedor `whatsapp-service` y escanear el QR:
> ```bash
> docker compose logs -f whatsapp-service
> ```

## Variables de entorno

### contact-service `.env`
| Variable              | Descripción                         |
|-----------------------|-------------------------------------|
| EMAIL_HOST            | SMTP host (smtp.gmail.com)          |
| EMAIL_PORT            | Puerto SMTP (587)                   |
| EMAIL_USER            | Correo remitente                    |
| EMAIL_PASS            | App Password de Gmail               |
| EMAIL_DEST            | Correo destinatario                 |
| WHATSAPP_SERVICE_URL  | URL del whatsapp-service            |

### appointment-service `.env`
| Variable              | Descripción                         |
|-----------------------|-------------------------------------|
| MONGODB_URI           | URI de MongoDB                      |
| WHATSAPP_SERVICE_URL  | URL del whatsapp-service            |

## Endpoints

### API Gateway (puerto 3000)

#### POST `/api/contact`
```json
{
  "nombre": "Juan García",
  "correo": "juan@example.com",
  "telefono": "3001234567",
  "mensaje": "Quisiera información sobre la revisión."
}
```

#### POST `/api/appointments`
```json
{
  "nombre": "Carlos López",
  "correo": "carlos@example.com",
  "telefono": "3109876543",
  "fecha": "10 de mayo",
  "hora": "10:00 AM",
  "servicio": "motocicleta"
}
```

#### GET `/api/appointments`
Retorna todas las citas ordenadas por fecha de creación.

#### DELETE `/api/appointments/:id`
Cancela la cita con el UUID indicado.

#### POST `/api/chatbot/chat`
```json
{ "message": "¿Cuál es el horario?" }
```
Respuesta:
```json
{ "response": "Atendemos de lunes a viernes de 7AM a 6PM..." }
```

## WhatsApp — Configuración QR

1. Arranca `whatsapp-service`
2. Aparece el código QR en consola
3. Abre WhatsApp en tu teléfono → **Dispositivos vinculados** → **Vincular dispositivo**
4. Escanea el QR
5. La sesión se guarda en `.wwebjs_auth/` — no necesitas escanear de nuevo al reiniciar

## Configuración en Render

Despliega cada servicio como un **Web Service** independiente:

1. Conecta el repositorio en Render
2. Configura el directorio raíz: `backend/services/<nombre-servicio>`
3. Comando de arranque: `node src/index.js`
4. Agrega las variables de entorno en el panel de Render

> **Nota**: `whatsapp-service` usa Puppeteer/Chromium. En Render, usa el plan **Standard** o superior que permita procesos headless. Para producción considera correr `whatsapp-service` en un VPS propio para mayor estabilidad del QR.
