from typing import Optional
import psycopg2
from fastapi import FastAPI

app = FastAPI() # 建立一個 Fast API application

@app.get("/start") 
def read_root():
    
    return {"Hello": "World"}


@app.get("/users/{user_id}") 
def read_user(user_id: int, q: Optional[str] = None):
    return {"user_id": user_id, "q": q}