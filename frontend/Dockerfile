FROM nginx:1.25-alpine

# Crear config.js en tiempo de arranque para apuntar al gateway de Render
RUN apk add --no-cache gettext

# Copiar archivos estáticos del frontend
COPY . /usr/share/nginx/html
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Copiar configuración nginx específica para Docker
COPY nginx.conf /etc/nginx/conf.d/default.conf

ENTRYPOINT ["/docker-entrypoint.sh"]

EXPOSE 80
