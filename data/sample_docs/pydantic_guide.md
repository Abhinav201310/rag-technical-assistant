# Pydantic V2 Data Validation and Settings Management

## What is Pydantic?
Pydantic is the most widely used data validation library for Python. Fast and extensible, Pydantic plays nicely with your linters, IDEs, and Python static analysis.

## Core Features in Pydantic V2
- **Core in Rust**: Pydantic V2 has its validation engine rewritten in Rust (`pydantic-core`), making it 5x to 50x faster than V1.
- **BaseModel**: The standard way to define structured schema.
- **Field Validator**: Define custom data validation logic with `@field_validator`.
- **Model Validator**: Validate multiple fields together with `@model_validator(mode='after')` or `mode='before'`.
- **Type Serialization**: Fast JSON serialization using `.model_dump()` and `.model_dump_json()`.

## Defining Models and Validation

```python
from pydantic import BaseModel, Field, EmailStr, field_validator
from typing import List, Optional

class UserSchema(BaseModel):
    id: int
    name: str = Field(..., min_length=2, description="User full name")
    email: EmailStr
    tags: List[str] = Field(default_factory=list)
    age: Optional[int] = Field(default=None, ge=18, le=100)

    @field_validator('name')
    @classmethod
    def name_must_contain_space(cls, v: str) -> str:
        if ' ' not in v:
            raise ValueError('Must contain first and last name')
        return v.title()

# Usage
user = UserSchema(id=1, name="alice smith", email="alice@example.com", age=25)
print(user.model_dump())
```

## Settings Management with pydantic-settings
Pydantic also provides `BaseSettings` for reading environment variables automatically:

```python
from pydantic_settings import BaseSettings

class AppSettings(BaseSettings):
    groq_api_key: str
    database_url: str = "sqlite:///./test.db"
    debug: bool = False

    class Config:
        env_file = ".env"
```
