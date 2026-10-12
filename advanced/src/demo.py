"""Loopback-only learning demo: python src/demo.py --port 8765."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

from inference import load_latest, predict_message


def make_handler(bundle):
    class Handler(BaseHTTPRequestHandler):
        def send_body(self, status, body, content_type="application/json; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status, value):
            self.send_body(status, json.dumps(value, ensure_ascii=False).encode("utf-8"))

        def do_GET(self):
            if self.path == "/":
                self.send_body(200, Path(__file__).with_name("demo.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self.send_json(200, {"run_id": bundle["metadata"]["run_id"], "model": bundle["metadata"]["selected_model"], "labels": len(bundle["category_names"])})
            else:
                self.send_json(404, {"error": "Bulunamadı."})

        def do_POST(self):
            if self.path != "/api/predict":
                self.send_json(404, {"error": "Bulunamadı."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("İstek boyutu geçersiz.")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("JSON nesnesi bekleniyor.")
                result = predict_message(bundle, body.get("message"))
            except (ValueError, UnicodeDecodeError) as error:
                self.send_json(400, {"error": str(error)})
                return
            except Exception:
                self.send_json(500, {"error": "Tahmin sırasında hata oluştu."})
                return
            self.send_json(200, result)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(load_latest()))
    print(f"Demo: http://127.0.0.1:{args.port} — durdurmak için Ctrl+C", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
