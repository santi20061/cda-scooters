'use strict';

const { Router }        = require('express');
const { createContact } = require('../controllers/contactController');
const { validateContact } = require('../middleware/validator');

const router = Router();
router.post('/contact', validateContact, createContact);

module.exports = router;
