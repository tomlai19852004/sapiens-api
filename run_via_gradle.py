from gradio_client import Client, handle_file
from PIL import Image
import numpy as np
import numba as nb
import hashlib

@nb.njit
def replace_where(arr, needle, replace):
    arr = arr.ravel()
    needles = set(needle)
    for idx in range(arr.size):
        if arr[idx] in needles:
            arr[idx] = replace

client = Client("facebook/sapiens-seg")
result = client.predict(
    image=handle_file('https://firebasestorage.googleapis.com/v0/b/oh-my-ink.appspot.com/o/users%2F0MZ3PhcormOg22OhuOK0kbkcyNO2%2Ftemp-uploads%2Fbody-image%2F074242e6eb053e1ca23e701a4a666c58febfb04b77e37d310a14b5aab90816ab.png?alt=media&token=519ed203-a06f-413a-a41d-1ff03113a07a'),
    model_name="1b",
    api_name="/process_image"
    )

# left low leg
# #34ffa6
# rgba(52,255,166,255)

# right hand
# #fe3233
# rgba(254,50,51,255)

# torso
# #3ed633
# rgba(62,214,51,255)

# right low arm
# #6cfe32
# rgba(109,255,50,255)

# face neck
# #99d4ff
# rgba(153,212,255,255)

# right low leg
# #cad732
# rgba(202,215,50,255)

# lower lip
# #efd693
# rgba(239,214,147,255)

# hair
# #fe3289
# rgba(254,50,137,255)

# left up arm
# #ff8a32
# rgba(255,138,50,255)

# upper lip
# #8fea8d
# rgba(143,234,141,255)

# left foot
# #ca32d5
# rgba(202,50,213,255)

# left up leg
# #34fefe
# rgba(52,254,254,255)

# left hand
# #fe32e1
# rgba(254,50,225,255)

# right up arm
# #4c6dd6
# rgba(76,109,214,255)

# left low arm
# #32b4d6
# rgba(50,180,214,255)

# right foot
# #d69c31
# rgba(214,156,49,255)

# right up leg
# #31a7ff
# rgba(49,167,255,255)


# if result[0]:
#     im = Image.open(result[0])
#     print( im )    
#     print( type( im ) )

#     im_rgb = im.convert('RGB')
#     pixels = im_rgb.load()
#     width, height = im_rgb.size

#     coords = []
#     for x in range(width):
#         for y in range(height):
#             if pixels[x, y] == (162, 176, 188):
#                 coords.append((x, y))

#     print( coords )
    # colors = im.convert('RGB').getcolors(maxcolors=256)

    # print( colors )

white_map = np.array([2,4,5,6,7,10,11,13,14,15,16,19,20,21])
black_map = np.array([1,3,8,9,12,17,18,22,23,24,25,26,27])

pixels = []
if result[1]:
    data = np.load(result[1])
    
    replace_where(data, white_map, 255)
    replace_where(data, black_map, 0)
    
    img_array = data.astype(np.uint8)
    img = Image.fromarray(img_array)
    print( type(img))
    print( hashlib.sha256(img.tobytes()).hexdigest() )

    img.save('results/img.png')

    # print( data )
    # print( data[0])
    # print( len(data))
    # print( len(data[0]))

    # for row in data:
    #     for column in row:
    #         if column == 2:
                