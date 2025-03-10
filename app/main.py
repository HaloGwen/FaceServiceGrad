from common import comlogger
from dotenv import load_dotenv

from fastapi import FastAPI
from core.face_service import FaceService
from db.milvus import MilvusClient
from config import Settings

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse 
from starlette.middleware.cors import CORSMiddleware

logger = comlogger.get_shared_logger()
env = load_dotenv()
settings = Settings()

app = FastAPI()
router = APIRouter()

# Initialize clients
milvus_client = MilvusClient(settings.MILVUS_HOST, settings.MILVUS_PORT)

# Initialize face service
face_service = FaceService(
    model_path=settings.MODEL_PATH,
    threshold=settings.SIMILARITY_THRESHOLD,
    milvus_client=milvus_client,
)

@router.post("/enroll")
async def enroll_face(
    file: UploadFile = File(...),
):
    """
    Enroll a new face with name
    """
    contents = await file.read()
    result = face_service.enroll_face(contents)
    return JSONResponse(content=result)

@router.post("/check-in")
async def check_in(
    file: UploadFile = File(...),
):
    """
    Check in a face
    """
    contents = await file.read()
    result = face_service.check_in(contents)
    return JSONResponse(content=result)

@router.put("/update")
async def update_face(
    face_id: str = Form(...),  # Nhận face_id từ form
    file: UploadFile = File(...),  # Nhận file ảnh từ form
):
    """
    Update an existing face by face_id with new image data
    """
    contents = await file.read()
    result = face_service.update_face(face_id, contents)
    return JSONResponse(content=result)

@router.delete("/delete-all")
async def delete_all_faces():
    """
    Delete all faces from the database
    """
    result = face_service.delete_all_faces()
    return JSONResponse(content=result)

@router.delete("/delete")
async def delete_face(
    face_id: str = Form(...),  # Nhận face_id từ form
):
    """
    Update an existing face by face_id with new image data
    """
    result = face_service.delete_face_by_id(face_id)
    return JSONResponse(content=result)

app.include_router(
    router,
    prefix="/api/v1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
