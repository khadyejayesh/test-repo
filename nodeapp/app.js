console.log("File started");
const express = require('express');
const os = require('os');

const app = express();
const PORT = process.env.PORT || 3000;

// Health check route
app.get('/', (req, res) => {
    res.json({
        message: "Node.js App Deployed Successfully 🚀",
        hostname: os.hostname(),
        uptime: process.uptime(),
        environment: process.env.NODE_ENV || "development",
        timestamp: new Date()
    });
});

// Sample API route
app.get('/api/status', (req, res) => {
    res.json({
        status: "Running",
        version: "1.0.0"
    });
});

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});