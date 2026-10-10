require('dotenv').config();
const express = require('express');
const etsyRoutes = require('./routes/etsy');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

// Routes
app.use('/api/etsy', etsyRoutes);

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
