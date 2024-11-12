from fastapi import APIRouter, Request
from mmseg.apis import inference_model, init_model, show_result_pyplot

router = APIRouter()

@router.post('/sapiens')
async def sapiens_func(request: Request):
    payload = {'result': "this is your result."}
    return payload