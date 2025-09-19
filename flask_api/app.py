import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from .burnout import BurnoutPredictor
from .rag import TaruRAG, get_simulated_data


def create_app() -> Flask:
    load_dotenv()

    app = Flask(__name__)

    cors_origins = os.environ.get("CORS_ORIGINS", "*")
    CORS(app, resources={r"/*": {"origins": cors_origins.split(",")}})

    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not google_api_key:
        app.logger.warning("GOOGLE_API_KEY is not set. /chat will not work until set.")

    # Init models and data once
    user_history, campus_resources, academic_articles = get_simulated_data()
    rag = TaruRAG.from_raw_texts(user_history, campus_resources, academic_articles, google_api_key)
    burnout = BurnoutPredictor()
    burnout.train(user_history)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/chat")
    def chat():
        data = request.get_json(silent=True) or {}
        question = data.get("message") or data.get("question") or ""
        if not question:
            return jsonify({"error": "Missing 'message' in JSON body"}), 400
        try:
            result = rag.query(question)
            return jsonify({
                "answer": result.answer,
                "actions": result.actions or [],
                "sources": result.sources or [],
                "confidence": result.confidence,
            })
        except Exception as exc:
            app.logger.exception("Chat endpoint error")
            return jsonify({"error": str(exc)}), 500

    @app.get("/risk")
    def risk():
        try:
            prob, reason = burnout.predict_risk_for_today(burnout.user_history_df)
            return jsonify({
                "risk_probability": prob,
                "reason": reason,
            })
        except Exception as exc:
            app.logger.exception("Risk endpoint error")
            return jsonify({"error": str(exc)}), 500

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)


