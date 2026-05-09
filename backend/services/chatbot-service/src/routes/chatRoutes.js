'use strict';

const { Router } = require('express');
const { chat }   = require('../controllers/chatController');

const router = Router();
router.post('/chat', chat);

module.exports = router;
