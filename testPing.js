require('dotenv').config();
const { pingEtsyApi } = require('./services/etsy/etsyClient');

async function test() {
  try {
    console.log('Testing Etsy API Ping...');
    const result = await pingEtsyApi();
    console.log('SUCCESS!');
    console.log(JSON.stringify(result, null, 2));
  } catch(e) {
    console.error('FAILED!');
    console.error(e.message);
  }
}

test();
