FROM nginx:1.25-alpine

# Copiar archivos estáticos del frontend
COPY . /usr/share/nginx/html

# Copiar configuración nginx específica para Docker
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
