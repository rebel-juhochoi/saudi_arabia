var http = require('http');
var fs = require('fs');
var path = require('path');

var PORT = 8000;

var server = http.createServer(function(req, res) {
    // Handle API proxy requests
    if (req.url.startsWith('/api/')) {
        // Check if it's a batch inference request (goes to Triton)
        if (req.url.startsWith('/api/batch-inference/')) {
            var http = require('http');
            var options = {
                hostname: 'localhost',
                port: 8002,
                path: req.url,
                method: req.method,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization'
                }
            };
        } else {
            // Proxy other API requests to local FastAPI server
            var http = require('http');
            var options = {
                hostname: 'localhost',
                port: 8001,
                path: req.url,
                method: req.method,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization'
                }
            };
        }
        
        var proxyReq = http.request(options, function(proxyRes) {
            // Handle different stream types
            if (req.url.startsWith('/api/video-stream/')) {
                res.writeHead(proxyRes.statusCode, {
                    'Content-Type': proxyRes.headers['content-type'] || 'multipart/x-mixed-replace; boundary=frame',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive'
                });
            } else if (req.url.startsWith('/api/inference-stream/') || req.url.startsWith('/api/batch-inference/')) {
                res.writeHead(proxyRes.statusCode, {
                    'Content-Type': 'text/event-stream',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': 'Cache-Control',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive'
                });
            } else {
                res.writeHead(proxyRes.statusCode, {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization'
                });
            }
            proxyRes.pipe(res);
        });
        
        proxyReq.on('error', function(err) {
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Proxy error: ' + err.message }));
        });
        
        if (req.method === 'POST') {
            req.pipe(proxyReq);
        } else {
            proxyReq.end();
        }
        return;
    }
    
    // Handle Triton API proxy requests
    if (req.url.startsWith('/v2/')) {
        // Proxy Triton requests to local Triton server
        var http = require('http');
        var options = {
            hostname: 'localhost',
            port: 8002,
            path: req.url,
            method: req.method,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            }
        };
        
        var proxyReq = http.request(options, function(proxyRes) {
            res.writeHead(proxyRes.statusCode, {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            });
            proxyRes.pipe(res);
        });
        
        proxyReq.on('error', function(err) {
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Proxy error: ' + err.message }));
        });
        
        proxyReq.end();
        return;
    }
    
    // Handle static file requests
    var filePath = path.join(__dirname, req.url === '/' ? 'index.html' : req.url);
    var extname = String(path.extname(filePath)).toLowerCase();
    
    var mimeTypes = {
        '.html': 'text/html',
        '.js': 'text/javascript',
        '.css': 'text/css',
        '.json': 'application/json',
        '.png': 'image/png',
        '.jpg': 'image/jpg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml',
        '.wav': 'audio/wav',
        '.mp4': 'video/mp4',
        '.woff': 'application/font-woff',
        '.ttf': 'application/font-ttf',
        '.eot': 'application/vnd.ms-fontobject',
        '.otf': 'application/font-otf',
        '.wasm': 'application/wasm'
    };
    
    var contentType = mimeTypes[extname] || 'application/octet-stream';
    
    fs.readFile(filePath, function(error, content) {
        if (error) {
            if (error.code == 'ENOENT') {
                res.writeHead(404, { 'Content-Type': 'text/html' });
                res.end('<h1>404 - Page Not Found</h1>', 'utf-8');
            } else {
                res.writeHead(500);
                res.end('Server Error: ' + error.code + '\n');
            }
        } else {
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(content, 'utf-8');
        }
    });
});

server.listen(PORT, function() {
    console.log('Frontend server running at http://localhost:' + PORT);
    console.log('Backend servers:');
    console.log('  FastAPI: http://localhost:8001');
    console.log('  Triton:  http://localhost:8002');
});