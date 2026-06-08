from fastapi import FastAPI 
from app import routes

app = FastAPI(title="BrushStrokeNeT API")

app.include_router(routes.nst_router)
