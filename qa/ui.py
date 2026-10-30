"""Loopback-only UI gateway. Academic API and its bearer contract stay separate."""
import json
import os
from pathlib import Path
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from qa.api import PredictRequest

ROOT = Path(__file__).resolve().parents[1]
ORIGINS = {'http://127.0.0.1:8765', 'http://localhost:8765'}


def create_app(transport=None):
    token = os.environ.get('QA_API_TOKEN', '')

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(base_url='http://127.0.0.1:8000', timeout=30,
                                     trust_env=False, transport=transport) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware('http')
    async def local_guard(request: Request, call_next):
        # Host allowlist prevents DNS rebinding; no CORS and strict Origin for writes.
        if request.headers.get('host') not in {'127.0.0.1:8765', 'localhost:8765'}:
            return JSONResponse({'error':'Acceso local no permitido.'}, 403)
        if request.method not in {'GET','HEAD'} and (
            request.headers.get('origin') not in ORIGINS or
            request.headers.get('x-pimientero-ui') != '1'
        ):
            return JSONResponse({'error':'Origen de la solicitud no permitido.'}, 403)
        response = await call_next(request)
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Cache-Control'] = 'no-store'
        return response

    async def forward(path, body=None):
        if not token:
            return JSONResponse({'error':'Inicia la aplicación con INICIAR_PIMIENTERO.bat.'}, 503)
        try:
            client = app.state.client
            headers = {'Authorization':'Bearer '+token}
            r = await client.get(path,headers=headers) if body is None else await client.post(path,headers=headers,json=body)
            if r.status_code != 200:
                return JSONResponse({'error':'La API está cargando o no está disponible. Revisa la ventana de inicio.'}, 503)
            return r.json()
        except (httpx.HTTPError, ValueError):
            return JSONResponse({'error':'No se pudo conectar con la API local. Revisa la ventana de inicio.'}, 503)

    @app.get('/')
    def index():
        return FileResponse(ROOT/'frontend/index.html')

    @app.get('/ui/status')
    async def status():
        return await forward('/health')

    @app.get('/ui/info')
    async def info():
        return await forward('/info')

    @app.get('/ui/evaluation')
    def evaluation():
        data = json.loads((ROOT/'logs/quality20261029/final_comparison.json').read_text())['profiles']['trained']
        return {'source':'logs/quality20261029/final_comparison.json · trained',
                'label':'quality40k · checkpoint-2000 · candidata evaluada',
                'metrics':data['groups']['all'],
                'answerable':data['groups']['answerable'],
                'unanswerable':data['groups']['unanswerable'],
                'weighted_75_25':data['weighted_75_25']}

    @app.post('/ui/predict')
    async def predict(body: PredictRequest):
        return await forward('/predict', body.model_dump())

    app.mount('/assets', StaticFiles(directory=ROOT/'frontend'), name='assets')
    return app


app = create_app()
