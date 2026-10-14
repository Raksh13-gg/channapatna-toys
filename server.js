require('dotenv').config();
const express = require('express');
const etsyRoutes = require('./routes/etsy');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

app.get('/', (req, res) => {
  res.json({
    success: true,
    message: 'Channapatna Toys API is running.',
    endpoints: {
      ping: '/api/etsy/ping'
    }
  });
});

// Routes
app.use('/api/etsy', etsyRoutes);

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
