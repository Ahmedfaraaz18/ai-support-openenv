from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware

from scripts.server import app as flask_app


app = FastAPI(title="OpenEnv Support Agent")
app.mount("/", WSGIMiddleware(flask_app))
