from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from agent import Agent

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

class GoalRequest(BaseModel):
    goal: str

@app.post("/api/chat")
async def chat_endpoint(req: GoalRequest):
    agent = Agent()
    return StreamingResponse(
        agent.run_generator(req.goal),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
