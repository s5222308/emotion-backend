import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from Controllers.Endpoints import register_routes
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://localhost:3000", "http://annotation-tool:3000", "http://annotation-tool:5173"], supports_credentials=True)

register_routes(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)
