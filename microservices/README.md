# 🛴 CDA Scooters - Microservicios

Sistema de microservicios para CDA Scooters Asociados SAS, implementando 3 servicios independientes conectados mediante un API Gateway y ejecutados con Docker.

## 📋 Servicios Disponibles

### 1. **API Gateway** (Puerto 3000)
Enrutador central que coordina todos los microservicios.

**Rutas:**
```
GET  /health                      # Health check
GET  /api/status                  # Estado del sistema
```

### 2. **Contact Service** (Puerto 3001)
Gestiona formularios de contacto con envío a Email y WhatsApp.

**Rutas:**
```
POST /contact/send                # Enviar contacto (Email + WhatsApp)
GET  /contact/messages            # Listar mensajes de contacto
```

### 3. **Appointment Service** (Puerto 3002)
Gestión de citas con notificaciones por WhatsApp.

**Rutas:**
```
POST   /appointments/book         # Agendar cita
GET    /appointments              # Listar todas las citas
GET    /appointments/:id          # Obtener cita específica
PUT    /appointments/:id          # Actualizar cita
DELETE /appointments/:id          # Cancelar cita
POST   /appointments/:id/remind   # Enviar recordatorio
```

### 4. **Chatbot Service** (Puerto 3003)
Asistente IA impulsado por OpenAI especializado en CDA Scooters.

**Rutas:**
```
POST /chat/message                # Enviar mensaje al chatbot
GET  /chat/history                # Obtener historial de conversación
GET  /chat/conversations          # Listar todas las conversaciones (Admin)
DELETE /chat/history/:sessionId   # Limpiar conversación
```

---

## 🚀 Instalación y Ejecución con Docker

### Opción recomendada: Docker Compose

Desde la carpeta `microservices/`:

```bash
docker compose up --build
```

Servicios expuestos:
- API Gateway: http://localhost:3000
- Contact Service: http://localhost:3001
- Appointment Service: http://localhost:3002
- Chatbot Service: http://localhost:3003

### Detener servicios

```bash
docker compose down
```

---

## 📝 Configuración de Credenciales

### Crear archivo `.env` en la carpeta `microservices/`

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

El API Gateway usa estas URLs dentro de la red de Docker:

```env
CONTACT_SERVICE_URL=http://contact-service:3001
APPOINTMENT_SERVICE_URL=http://appointment-service:3002
CHATBOT_SERVICE_URL=http://chatbot-service:3003
```

---

## 🧪 Ejemplos de Uso

### Contact Service

```bash
# Enviar contacto
curl -X POST http://localhost:3000/api/contact/send \
  -H "Content-Type: application/json" \
  -d '{
    "nombre": "Juan Pérez",
    "email": "juan@example.com",
    "telefono": "+573015551234",
    "asunto": "Consulta sobre servicios",
    "mensaje": "¿Cuáles son sus horarios?",
    "whatsapp": true
  }'

# Obtener mensajes
curl http://localhost:3000/api/contact/messages
```

### Appointment Service

```bash
# Agendar cita
curl -X POST http://localhost:3000/api/appointments/book \
  -H "Content-Type: application/json" \
  -d '{
    "nombre": "María García",
    "telefono": "+573001234567",
    "email": "maria@example.com",
    "fecha": "2025-05-15",
    "hora": "14:30",
    "servicio": "Mantenimiento"
  }'

# Listar citas
curl http://localhost:3000/api/appointments

# Enviar recordatorio
curl -X POST http://localhost:3000/api/appointments/uuid-123/remind
```

### Chatbot Service

```bash
# Enviar mensaje
curl -X POST http://localhost:3000/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "sessionId": "user-session-123",
    "mensaje": "¿Qué servicios ofrecen?"
  }'

# Obtener historial
curl "http://localhost:3000/api/chat/history?sessionId=user-session-123"
```

---

## 🌐 Interfaz de Prueba

Abre el archivo [example.html](example.html) en tu navegador para probar todos los servicios.

---

## 🔒 Seguridad (Checklist)

- [ ] Usar HTTPS en producción
- [ ] Agregar autenticación JWT
- [ ] Implementar rate limiting
- [ ] Validar y sanitizar inputs
- [ ] Usar .env para variables sensibles
- [ ] Configurar CORS restrictivo
- [ ] Agregar logs de auditoría
- [ ] Monitorear y alertar sobre errores

---

## 📚 Documentación Completa

- 📖 **[SIN_DOCKER.md](SIN_DOCKER.md)** - Guía Docker y compose
- 🚀 **[RENDER_SETUP.md](RENDER_SETUP.md)** - Deploy en Render
- 📑 **[INDEX.md](INDEX.md)** - Índice del proyecto

---

## 📞 Contacto y Soporte

Para soporte técnico:
- **Email:** tech@cdascooters.com
- **WhatsApp:** +573001234567

---

## 📄 Licencia

CDA Scooters Asociados SAS © 2025
