#!/bin/bash
# Launch script for LADOS web interface

set -e

echo "=== LADOS Web Application Launcher ==="

# Check if API is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "Starting FastAPI server..."
    cd "$(dirname "$0")/.."
    source /home/makishima/.cache/pypoetry/virtualenvs/lados-C-p4PIwJ-py3.13/bin/activate
    export MODEL_PATH=runs/20251113_222434/checkpoint.pt
    export API_KEY=${API_KEY:-deb479f1-f7a7-4973-bf62-8090496f13d0}
    python -m uvicorn src.server.app:app --host 0.0.0.0 --port 8000 &
    API_PID=$!
    echo "API server started (PID: $API_PID)"
    sleep 3
else
    echo "✓ API server already running"
fi

# Check Flutter
if command -v flutter &> /dev/null; then
    echo "Launching Flutter web app..."
    cd flutter/sample_app
    flutter pub get
    flutter run -d chrome --web-port 8080
else
    echo ""
    echo "Flutter not found. Options:"
    echo "1. Install Flutter: https://flutter.dev/docs/get-started/install"
    echo "2. Use the API directly:"
    echo "   - API docs: http://localhost:8000/docs"
    echo "   - Health: http://localhost:8000/health"
    echo "   - Test prediction: curl -X POST http://localhost:8000/predict -F 'file=@image.jpg'"
    echo ""
    echo "Or create a simple HTML interface..."
    echo "Creating simple HTML interface at web/index.html"
    
    mkdir -p web
    cat > web/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>LADOS Classifier</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
        .result { margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; }
        .class-item { display: flex; justify-content: space-between; margin: 10px 0; }
        .progress-bar { height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background: #4CAF50; transition: width 0.3s; }
        button { padding: 10px 20px; background: #2196F3; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #1976D2; }
    </style>
</head>
<body>
    <h1>LADOS Image Classifier</h1>
    <div class="upload-area">
        <input type="file" id="imageInput" accept="image/*" style="display: none;">
        <button onclick="document.getElementById('imageInput').click()">Select Image</button>
        <button onclick="predict()">Predict</button>
        <div id="preview"></div>
    </div>
    <div id="result" class="result" style="display: none;"></div>
    
    <script>
        let selectedFile = null;
        document.getElementById('imageInput').addEventListener('change', function(e) {
            selectedFile = e.target.files[0];
            if (selectedFile) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('preview').innerHTML = 
                        '<img src="' + e.target.result + '" style="max-width: 300px; margin-top: 20px;">';
                };
                reader.readAsDataURL(selectedFile);
            }
        });
        
        async function predict() {
            if (!selectedFile) {
                alert('Please select an image first');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = 'Predicting...';
            
            try {
                const response = await fetch('http://localhost:8000/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                let html = '<h3>Prediction Results:</h3>';
                data.predictions.slice(0, 5).forEach(pred => {
                    const width = (pred.score * 100).toFixed(1);
                    html += `
                        <div class="class-item">
                            <span>${pred.class}</span>
                            <span>${width}%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${width}%"></div>
                        </div>
                    `;
                });
                html += `<p><small>Inference time: ${data.inference_time_ms.toFixed(2)}ms</small></p>`;
                resultDiv.innerHTML = html;
            } catch (error) {
                resultDiv.innerHTML = '<p style="color: red;">Error: ' + error.message + '</p>';
            }
        }
    </script>
</body>
</html>
EOF
    echo "✓ Simple HTML interface created at web/index.html"
    echo "Open it in your browser: file://$(pwd)/web/index.html"
fi

