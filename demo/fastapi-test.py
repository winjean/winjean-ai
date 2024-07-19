from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
@app.get("/items/{item_id}/{q}")
@app.get("/aaaa")
async def read_item(item_id: int = 1, q: str = None):
    return {"item_id": item_id, "q": q}


@app.post("/items/")
async def create_item(item: dict):
    return item

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)