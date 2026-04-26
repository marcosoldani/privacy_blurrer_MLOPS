# Stage 1: build del bundle React+Vite con Node
FROM node:20-alpine AS build
WORKDIR /app

# Cache friendly: prima i manifest, poi il codice
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Stage 2: nginx serve i file statici buildati
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
