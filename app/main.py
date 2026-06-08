from fastapi import FastAPI 
from app.routes.style_transfer import router as nst_router

app = FastAPI(tile="BrushStrokeNeT API")

app.include_router(nst_router)
