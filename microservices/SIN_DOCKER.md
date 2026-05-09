# 🎯 GUÍA DOCKER

Para ejecutar los microservicios con **Docker Compose** en tu máquina.

---

## ⚡ Inicio Rápido

Desde la carpeta `microservices/`:

```bash
docker compose up --build
```

Para detenerlo:

```bash
docker compose down
```

---

## 🔧 Configuración `.env`

Crear archivo `.env` en la carpeta `microservices/` con las credenciales compartidas:

```env
SMTP_USER=tu-email@gmail.com
SMTP_PASS=tu-contraseña-app
ADMIN_EMAIL=admin@cdascooters.com

TWILIO_ACCOUNT_SID=xxx
TWILIO_AUTH_TOKEN=xxx
TWILIO_PHONE=+1234567890

OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-3.5-turbo
```

El gateway obtiene estas URLs dentro de Docker:

```env
CONTACT_SERVICE_URL=http://contact-service:3001
APPOINTMENT_SERVICE_URL=http://appointment-service:3002
CHATBOT_SERVICE_URL=http://chatbot-service:3003
```

---

## ✅ Verificar que funciona

```bash
curl http://localhost:3000/health
curl http://localhost:3000/api/status
curl -X POST http://localhost:3000/api/contact/send \
  -H "Content-Type: application/json" \
  -d '{"nombre":"Juan","email":"juan@example.com","telefono":"+573001234567","asunto":"Test","mensaje":"Prueba"}'
```

---

## 🔒 Credenciales Necesarias

- OpenAI API Key
- Gmail SMTP
- Twilio

---

## 🐳 Comando Único

```bash
docker compose up --build
```

---

## 📊 Puertos Locales

| Servicio | Puerto | URL |
|----------|--------|-----|
| API Gateway | 3000 | http://localhost:3000 |
| Contact | 3001 | http://localhost:3001 |
| Appointments | 3002 | http://localhost:3002 |
| Chatbot | 3003 | http://localhost:3003 |

---

¡Listo! 🚀
