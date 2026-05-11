'use strict';

require('dotenv').config();
const mongoose = require('mongoose');

const URI = process.env.MONGODB_URI;

console.log('\n🔌 Probando conexión a MongoDB Atlas...');
console.log('   Base de datos: CDA_db');
console.log('   Colección:     nose\n');

mongoose.connect(URI)
  .then(async () => {
    console.log('✅ CONEXIÓN EXITOSA\n');

    // Insertar documento de prueba en CDA_db.nose
    const TestSchema = new mongoose.Schema({ test: String, fecha: Date }, { collection: 'nose' });
    const Test = mongoose.model('Test', TestSchema);

    const doc = await Test.create({ test: 'Conexión verificada CDA Scooters', fecha: new Date() });
    console.log('✅ Documento de prueba guardado en CDA_db.nose:');
    console.log('  ', JSON.stringify(doc.toObject(), null, 2));

    await mongoose.disconnect();
    console.log('\n✅ Todo funciona correctamente. Puedes eliminar test-db.js\n');
    process.exit(0);
  })
  .catch((err) => {
    console.error('❌ ERROR DE CONEXIÓN:\n', err.message);
    console.log('\n📋 Posibles causas:');
    console.log('   1. IP no está en Network Access → Atlas → Security → Network Access → Add 0.0.0.0/0');
    console.log('   2. Contraseña incorrecta en MONGODB_URI del .env');
    console.log('   3. Usuario sin permisos → Atlas → Security → Database Access\n');
    process.exit(1);
  });
