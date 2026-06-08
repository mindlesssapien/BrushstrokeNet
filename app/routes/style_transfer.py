from app import services
from fastapi import UploadFile, File, APIRouter

router = APIRouter()

@router.post("/style-transfer")
async def style_transfer(content: UploadFile = File(...), style: UploadFile = File(...)):
    output_path = await services.run_nst(content, style)

    return {"output_image": output_path}
