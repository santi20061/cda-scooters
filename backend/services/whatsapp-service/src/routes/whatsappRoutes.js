'use strict';

const { Router } = require('express');
const { send, status } = require('../controllers/whatsappController');

const router = Router();
router.get('/status', status);
router.post('/send',   send);

module.exports = router;
