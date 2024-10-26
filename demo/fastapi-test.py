import time
import functools
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


app = FastAPI()
limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def measure_time_test(func):
    """
    在不启动服务时，自测用例时使用。（输出控制台：方法名和耗时） 不记录数据库
    """

    @functools.wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        start_time = time.time()
        result = await func(request, *args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        milliseconds = round(elapsed_time, 4)
        print(f'{func.__name__}: {milliseconds} ms')
        return result

    return wrapper


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
@app.get("/items/{item_id}/{q}")
async def read_item(item_id: int = 1, q: str = None):
    return {"item_id": item_id, "q": q}


async def async_function1():
    import asyncio
    print("Start 1")
    await asyncio.sleep(3)
    print("End 1")


async def async_function2():
    import asyncio
    print("Start 2")
    await asyncio.sleep(2)
    print("End 2")


@app.post("/items/")
@measure_time_test
@limiter.limit("2/minute")
async def post_item(request: Request):
    item = await request.json()

    time.sleep(3)

    import asyncio
    task1 = asyncio.create_task(async_function1())
    task2 = asyncio.create_task(async_function2())

    # await task1
    # await task2
    await asyncio.gather(task1, task2)

    item["test"] = "test"
    return JSONResponse(
        status_code=200,
        headers={"X-Foo": "bar"},
        content=item
    )


if __name__ == "__main__":
    from uvicorn import run

    run(app, host="127.0.0.1", port=8000)
