import uvicorn
import sys
from fastapi import FastAPI

from routers import sapiens_route
app = FastAPI()

def collect_argv(arg_list):
    arg_dict = dict()
    key = ''
    for ii, arg in enumerate(arg_list):
        if ii % 2:
            key = arg
        elif key != '':
            # print( 'key: {} and arg: {}'.format( key, arg )  )
            arg_dict[key] = arg
    return arg_dict

app.include_router(sapiens_route.router)

if __name__ == "__main__":
    arg_dict = collect_argv( sys.argv )
    print( arg_dict)

    port = arg_dict.get('--port', 8000)
    host = arg_dict.get('--host', '0.0.0.0')
    uvicorn.run(app, host=host, port=int(port))