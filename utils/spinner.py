spinner_html = """
    <div style="display: flex; justify-content: center; align-items: center; height: 300px;">
        <div style="text-align: center;">
            <div class="loader"></div>
            <p style="font-size: 20px; margin-top: 10px;">Running prompt...</p>
        </div>
    </div>
    <style>
    .loader {
    border: 12px solid #f3f3f3;
    border-top: 12px solid #3498db;
    border-radius: 50%;
    width: 80px;
    height: 80px;
    animation: spin 1s linear infinite;
    margin: auto;
    }
    @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
    }
    </style>
    """