from typing import Annotated
from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket
from tqdm import tqdm
from models.classes_and_palettes import GOLIATH_CLASSES, GOLIATH_PALETTE


import os
import torch

from models.funcs import load_model, inference_model, process_image_into_dataset, \
                        generate_image_mask, decode_base64_to_img, encode_img_to_base64

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




# Image segmentation for a single image
@router.post('/sapiens-seg-img')
async def sapiens_func(file: UploadFile):
    global model
    
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail='Missing required parameter.')
    contents = await file.read()
    inf_dataset, inf_dataloader = process_image_into_dataset(contents)

    payload = {'img_mask': None}

    for batch_idx, (batch_image_name, batch_orig_imgs, batch_imgs) in tqdm(
        enumerate(inf_dataloader), total=len(inf_dataloader)
    ):
        valid_images_len = len( batch_imgs )
        
        result = inference_model( model, batch_imgs, dtype=dtype )
        img_mask = generate_image_mask(batch_orig_imgs[0], result[0], GOLIATH_CLASSES, GOLIATH_PALETTE)
        payload['img_mask'] = img_mask

    
    return payload


@router.websocket("/sapiens-seg-stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive base64 encoded frame from client
            data = await websocket.receive_text()
            print( data )
            
            frame = decode_base64_to_img(data)
            
            # Process frame using existing pipeline
            inf_dataset, inf_dataloader = process_image_into_dataset(frame)
            
            for _, (_, batch_orig_imgs, batch_imgs) in enumerate(inf_dataloader):
                result = inference_model(model, batch_imgs, dtype=dtype)
                img_mask = generate_image_mask(batch_orig_imgs[0], result[0], 
                                            GOLIATH_CLASSES, GOLIATH_PALETTE)
                
                # img_str = encode_img_to_base64(img_mask)
                
                # Send processed frame back to client
                await websocket.send_text(f"data:image/png;base64,{img_mask}")
                
    except Exception as e:
        print(f"Error in websocket connection: {str(e)}")
    finally:
        await websocket.close()