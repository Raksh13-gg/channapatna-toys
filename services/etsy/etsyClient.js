const axios = require('axios');

const ETSY_API_KEY = process.env.ETSY_API_KEY;
const ETSY_SHARED_SECRET = process.env.ETSY_SHARED_SECRET;

const etsyClient = axios.create({
  baseURL: 'https://api.etsy.com/v3',
  headers: {
    'x-api-key': ETSY_API_KEY
  }
});

/**
 * Pings the Etsy OpenAPI
 * GET https://api.etsy.com/v3/application/openapi-ping
 */
async function pingEtsyApi() {
  try {
    const response = await etsyClient.get('/application/openapi-ping');
    return response.data;
  } catch (error) {
    if (error.response) {
      if (error.response.status === 429) {
        throw new Error('Etsy API Error: Too Many Requests (Rate limit exceeded)');
      } else if (error.response.status === 401) {
        throw new Error('Etsy API Error: Unauthorized');
      } else if (error.response.status === 403) {
        throw new Error('Etsy API Error: Forbidden');
      }
      throw new Error(`Etsy API Error: ${error.response.status}`);
    } else if (error.request) {
      throw new Error('Etsy API Error: Network timeout or no response');
    } else {
      throw new Error('Etsy API Error: Invalid request setup');
    }
  }
}

module.exports = {
  pingEtsyApi
};
