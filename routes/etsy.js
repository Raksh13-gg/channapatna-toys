const express = require('express');
const router = express.Router();
const etsyClient = require('../services/etsy/etsyClient');

router.get('/ping', async (req, res) => {
  try {
    if (!process.env.ETSY_API_KEY || !process.env.ETSY_SHARED_SECRET) {
      return res.status(500).json({
        success: false,
        message: 'Missing environment variables. Please configure ETSY_API_KEY and ETSY_SHARED_SECRET locally.'
      });
    }

    const result = await etsyClient.pingEtsyApi();
    
    return res.json({
      success: true,
      message: 'Etsy API connection successful',
      application_id: result.application_id || '...',
      data: result
    });
  } catch (error) {
    // Return safe error message without exposing credentials
    return res.status(500).json({
      success: false,
      message: error.message || 'An error occurred while connecting to the Etsy API.'
    });
  }
});

module.exports = router;
