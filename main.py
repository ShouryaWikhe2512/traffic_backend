from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import atcs

app = FastAPI(title="ATCS Intelligence Backend")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routes
app.include_router(atcs.router, prefix="/api/atcs")

@app.get("/")
async def root():
    return {"status": "online", "system": "ATCS Intelligence Unit"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
