# FastAPI Tutorial & Architecture Guide

## What is FastAPI?
FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints.

### Key Features
- **Fast**: Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic). One of the fastest Python frameworks available.
- **Fast to code**: Increase the speed to develop features by about 200% to 300%.
- **Fewer bugs**: Reduce about 40% of human (developer) induced errors.
- **Intuitive**: Great editor support. Completion everywhere. Less time debugging.
- **Easy**: Designed to be easy to use and learn. Less time reading docs.
- **Short**: Minimize code duplication. Multiple features from each parameter declaration.
- **Robust**: Get production-ready code. With automatic interactive documentation.
- **Standards-based**: Based on (and fully compatible with) the open standards for APIs: OpenAPI (previously known as Swagger) and JSON Schema.

## Creating a Simple Endpoint
Here is how to create a basic FastAPI endpoint:

```python
from fastapi import FastAPI

app = FastAPI(title="Sample API", version="1.0.0")

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

## Dependency Injection System
FastAPI includes an extremely powerful and intuitive Dependency Injection system.
Dependencies can be declared using `Depends()` from `fastapi`:

```python
from fastapi import Depends, FastAPI

app = FastAPI()

async def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons
```

### Benefits of Dependencies
1. Shared logic (the same code logic again and again).
2. Database connection sharing.
3. Security, authentication, and role authorization enforcement.
4. Minimize code duplication across route handlers.
