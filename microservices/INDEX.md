# 📑 Índice - CDA Scooters Microservicios

## 📂 Estructura (Docker)

```
microservices/
├── api-gateway/
│   ├── Dockerfile
│   ├── gateway.js
│   ├── package.json
│   └── .env.example
├── contact-service/
│   ├── Dockerfile
│   ├── service.js
│   ├── package.json
│   └── .env.example
├── appointment-service/
│   ├── Dockerfile
│   ├── service.js
│   ├── package.json
│   └── .env.example
├── chatbot-service/
│   ├── Dockerfile
│   ├── service.js
│   ├── package.json
│   └── .env.example
├── docker-compose.yml
├── .env.example
├── README.md
├── SIN_DOCKER.md
├── RENDER_SETUP.md
├── microservices-client.js
└── example.html
```

---

## 🚀 Inicio Rápido

```bash
docker compose up --build
```

### Verificar
```bash
curl http://localhost:3000/health
```

---

## 📚 Documentación

| Archivo | Contenido |
|---------|----------|
| **[README.md](README.md)** | Guía principal con Docker |
| **[SIN_DOCKER.md](SIN_DOCKER.md)** | Guía Docker y compose |
| **[RENDER_SETUP.md](RENDER_SETUP.md)** | Deploy en Render |

---

## 🔌 Microservicios

- API Gateway: puerto 3000
- Contact Service: puerto 3001
- Appointment Service: puerto 3002
- Chatbot Service: puerto 3003

---

## 🔐 Credenciales

- OpenAI API Key
- Gmail SMTP
- Twilio

---

**Versión:** 2.0.0 (Docker)  
**CDA Scooters © 2025**
