'use strict';

require('dotenv').config();
const express    = require('express');
const cors       = require('cors');
const { PORT }   = require('./config/config');
const chatRoutes = require('./routes/chatRoutes');

const app = express();
app.use(cors());
app.use(express.json());

app.get('/health', (_req, res) =>
  res.json({ status: 'ok', service: 'chatbot-service' })
);
app.use('/', chatRoutes);

app.use((err, _req, res, _next) => {
  console.error('[chatbot-service]', err.message);
  res.status(500).json({ error: err.message });
});

app.listen(PORT, () =>
  console.log(`🤖 Chatbot Service corriendo en puerto ${PORT}`)
);
