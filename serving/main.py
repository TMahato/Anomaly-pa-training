import pickle
from flask import Flask

app = Flask(__name__)

model = None

try:
    model = pickle.load(open('/mnt/models/4322_abod.pkl', 'rb'))
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")

@app.route("/")
def status():
    if model:
        return "✅ Model is ready"
    return "❌ Model not loaded"

if __name__ == "__main__":
    print("🚀 Starting server...")
    app.run(host="0.0.0.0", port=9001, debug=True)
