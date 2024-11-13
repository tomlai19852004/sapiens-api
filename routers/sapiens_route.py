from fastapi import APIRouter, Request
from models.funcs import load_model, inference_model

router = APIRouter()


@router.post('/sapiens')
async def sapiens_func(request: Request):
    payload = {'result': "this is your result."}
    return payload