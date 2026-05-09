# ðŸš€ RENDER DEPLOYMENT - Sin Docker

GuÃ­a para desplegar los microservicios en Render sin Docker.

---

## ðŸ“‹ Requisitos

- Git
- Node.js 18+
- Render account (https://render.com)

---

## ðŸ”§ Setup Inicial

### 1. Crear .env en cada servicio

**microservices/api-gateway/.env**
```
GATEWAY_PORT=3000
CONTACT_SERVICE_URL=https://cda-contact-service.onrender.com
APPOINTMENT_SERVICE_URL=https://cda-appointment-service.onrender.com
CHATBOT_SERVICE_URL=https://cda-chatbot-service.onrender.com
```

**microservices/contact-service/.env**
```
CONTACT_SERVICE_PORT=3001
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_SECURE=false
SMTP_USER=tu-email@gmail.com
SMTP_PASS=tu-contraseÃ±a-app
SMTP_FROM="CDA Scooters" <noreply@cdascooters.com>
ADMIN_EMAIL=admin@cdascooters.com
TWILIO_ACCOUNT_SID=xxx
TWILIO_AUTH_TOKEN=xxx
TWILIO_PHONE=+1234567890
```

**microservices/appointment-service/.env**
```
APPOINTMENT_SERVICE_PORT=3002
TWILIO_ACCOUNT_SID=xxx
TWILIO_AUTH_TOKEN=xxx
TWILIO_PHONE=+1234567890
```

**microservices/chatbot-service/.env**
```
CHATBOT_SERVICE_PORT=3003
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

---

## ðŸ“¦ InstalaciÃ³n Local (Desarrollo)

```bash
# Api Gateway
cd microservices/api-gateway
npm install
npm start

# Contact Service (otra terminal)
cd microservices/contact-service
npm install
npm start

# Appointment Service (otra terminal)
cd microservices/appointment-service
npm install
npm start

# Chatbot Service (otra terminal)
cd microservices/chatbot-service
npm install
npm start
```

---

## ðŸŒ Deploy a Render

### OpciÃ³n 1: Crear cada servicio manualmente en Render

#### Para cada servicio:

1. **Ir a https://render.com/dashboard**
2. Click en **"New Web Service"**
3. Conectar repositorio GitHub
4. Configurar:
   - **Name:** `cda-api-gateway` (o el nombre del servicio)
   - **Environment:** Node
   - **Build command:** `npm install`
   - **Start command:** `npm start`
   - **Root directory:** `microservices/api-gateway` (segÃºn servicio)

5. **Environment Variables:** Agregar variables de `.env`

6. Click **Deploy**

---

### OpciÃ³n 2: Usar un Ãºnico web service con PM2

Crear un archivo `ecosystem.config.js` en la raÃ­z:

```javascript
module.exports = {
  apps: [
    {
      name: 'api-gateway',
      script: './microservices/api-gateway/gateway.js',
      env: {
        PORT: 3000,
        NODE_ENV: 'production'
      }
    },
    {
      name: 'contact-service',
      script: './microservices/contact-service/service.js',
      env: {
        PORT: 3001,
        NODE_ENV: 'production'
      }
    },
    {
      name: 'appointment-service',
      script: './microservices/appointment-service/service.js',
      env: {
        PORT: 3002,
        NODE_ENV: 'production'
      }
    },
    {
      name: 'chatbot-service',
      script: './microservices/chatbot-service/service.js',
      env: {
        PORT: 3003,
        NODE_ENV: 'production'
      }
    }
  ]
};
```

En Render:
- **Build command:** `npm install && npm install -g pm2`
- **Start command:** `pm2 start ecosystem.config.js && pm2 logs`

---

## âš™ï¸ ConfiguraciÃ³n de puertos en Render

En **desarrollo local**: puertos 3000-3003  
En **Render**: Render usa puerto 3000 (automÃ¡tico)

**Ajuste necesario:**

Cada servicio debe escuchar en el puerto asignado por Render:

```javascript
// En cada service.js
const PORT = process.env.PORT || 3000;
```

---

## ðŸ“ .env Render vs Local

### Local (.env)
```
GATEWAY_PORT=3000
CONTACT_SERVICE_URL=http://localhost:3001
APPOINTMENT_SERVICE_URL=http://localhost:3002
CHATBOT_SERVICE_URL=http://localhost:3003
```

### Render (.env)
```
GATEWAY_PORT=$PORT (automÃ¡tico)
CONTACT_SERVICE_URL=https://cda-contact-service.onrender.com
APPOINTMENT_SERVICE_URL=https://cda-appointment-service.onrender.com
CHATBOT_SERVICE_URL=https://cda-chatbot-service.onrender.com
```

---

## ðŸ” Monitoreo en Render

```bash
# Ver logs en tiempo real
curl https://cda-api-gateway.onrender.com/health

# Verificar estado
curl https://cda-api-gateway.onrender.com/api/status
```

---

## ðŸ› Troubleshooting

### Servicio no inicia

```bash
# Ver logs en Render dashboard
# Verificar que package.json tiene "start" script
```

### Variables de entorno no se cargan

```bash
# Verificar que estÃ¡n en Render Environment
# Click en servicio â†’ Environment â†’ revisar variables
```

### Servicios no se comunican

```bash
# Verificar URLs en CONTACT_SERVICE_URL, etc
# Deben ser URLs HTTPS completas
# No localhost, sino https://cda-service.onrender.com
```

---

## ðŸ“Š URLs en ProducciÃ³n

- **API Gateway:** https://cda-api-gateway.onrender.com
- **Contact:** https://cda-contact-service.onrender.com
- **Appointments:** https://cda-appointment-service.onrender.com
- **Chatbot:** https://cda-chatbot-service.onrender.com

---

## âœ… Checklist Deploy

- [ ] Crear 4 web services en Render
- [ ] Conectar repositorio GitHub
- [ ] Agregar .env variables en cada servicio
- [ ] Verificar que npm start funciona
- [ ] Health check: `curl https://api-gateway.onrender.com/health`
- [ ] Probar endpoints desde frontend
- [ ] Verificar logs en Render dashboard

---

**Â¡Listo!** ðŸŽ‰

