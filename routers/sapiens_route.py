from typing import Annotated
from fastapi import APIRouter, File, HTTPException, UploadFile
import starlette
from tqdm import tqdm
import os
import torch

from models.funcs import load_model, inference_model, process_image_into_dataset, \
                        fake_pad_images_to_batchsize

checkpoint = os.getenv('CHECKPOINT')
use_torchscript = os.getenv('MODE', '').lower() == 'torchscript'
model = load_model(checkpoint, use_torchscript)

if not checkpoint:
    exit

## no precision conversion needed for torchscript. run at fp32
if not use_torchscript:
    dtype = torch.half if os.getenv('FP16', None) else torch.bfloat16
    model.to(dtype)
    model = torch.compile(model, mode="max-autotune", fullgraph=True)
else:
    dtype = torch.float32  # TorchScript models use float32
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Device is {}'.format(DEVICE))
    model = model.to(DEVICE)

router = APIRouter()

# img_data: Annotated[bytes, File(description="A file read as bytes")]
@router.post('/sapiens-seg-img')
async def sapiens_func(file: UploadFile):
    global model
    
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail='Missing required parameter.')
    contents = await file.read()
    inf_dataset, inf_dataloader = process_image_into_dataset(contents)

    total_results = []
    image_paths = []

    for batch_idx, (batch_image_name, batch_orig_imgs, batch_imgs) in tqdm(
        enumerate(inf_dataloader), total=len(inf_dataloader)
    ):
        valid_images_len = len( batch_imgs )
        batch_imgs = fake_pad_images_to_batchsize( batch_imgs )
        
        result = inference_model( model, batch_imgs, dtype=dtype )

        print( batch_imgs.shape )
        print( result.shape )

    
    payload = {'result': "this is your result."}
    return payload