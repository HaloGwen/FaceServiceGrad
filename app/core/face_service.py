from datetime import datetime
import uuid
import torch
from PIL import Image
import io
from fastapi import HTTPException
from db.milvus import MilvusClient
from models.face_model import FaceIdentification
from utils.image_processing import transform_image
from pymilvus import Collection
from common import comlogger
from facenet_pytorch import MTCNN

logger = comlogger.get_shared_logger()

class FaceService:
    def __init__(self, model_path, threshold: float, milvus_client: MilvusClient):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._load_model(model_path)
        self.milvus_client = milvus_client
        self.threshold = threshold
        
    def _load_model(self, model_path):
        model = FaceIdentification(num_classes=1020)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def extract_embedding(self, image_data):
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # add simple face detection and crop face only
        mtcnn = MTCNN(keep_all=True)
        boxes, _ = mtcnn.detect(image)
        if boxes is not None:
            box = boxes[0]
            padding = 20
            left, top, right, bottom = box
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(image.width, right + padding)
            bottom = min(image.height, bottom + padding)
            face = image.crop((left, top, right, bottom))
            image = transform_image(face).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model.extract_features(image)
                return embedding.cpu().numpy()[0]
        else:
            return {
                "statusCode": 400,
                "message": "No face detected",
                "data": None
            }

    def enroll_face(self, image_data: bytes):
        
        embedding_response = self.extract_embedding(image_data)
        
        if isinstance(embedding_response, dict):
            if embedding_response.get("statusCode") == 400:
                return embedding_response
        
        embedding = embedding_response

        logger.info(f"embedding length: {len(embedding)}")

        # Check if face already exists
        collection = Collection(self.milvus_client.collection_name)
        results = collection.search(
            data=[embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=1
        )
        logger.info(f"similarity: {results}")
        
        if results[0] and results[0][0].score > self.threshold:
            return {
                "statusCode": 400,
                "message": "Face already exists in the database",
                "data": {"distance": results[0][0].score}
            }
        
        # Generate unique face_id and timestamp
        face_id = str(uuid.uuid4())
        created_at = datetime.now()
        
        # Insert into Milvus
        collection.insert([
            [embedding],
            [face_id],
            [created_at.isoformat()]
        ])
        collection.flush()
        
        return {
            "statusCode": 200,
            "message": "Face enrolled successfully",
            "data": {"face_id": face_id}
        }

    def check_in(self, image_data: bytes):
        embedding_response = self.extract_embedding(image_data)
        
        if isinstance(embedding_response, dict):
            if embedding_response.get("statusCode") == 400:
                return embedding_response

        embedding = embedding_response
        logger.info(f"embedding length: {len(embedding)}")

        collection = Collection("face_embeddings")
        results = collection.search(
            data=[embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=1,
            output_fields=["face_id"]
        )
        logger.info(f"similarity: {results}")
        
        if not results[0] or results[0][0].score < self.threshold:
            return {
                "statusCode": 400,
                "message": "No matching face found",
                "data": {"Distance nearly": results[0][0].score}
            }
        
        match = results[0][0]
        face_id = match.entity.get('face_id')
        
        return {
            "statusCode": 200,
            "message": "Face matched successfully",
            "data": {"face_id": face_id, "similarity": float(match.score)}
        }
        
    def update_face(self, face_id: str, image_data: bytes):
        # Extract embedding from new image data
        embedding_response = self.extract_embedding(image_data)
        
        if isinstance(embedding_response, dict):
            if embedding_response.get("statusCode") == 400:
                return embedding_response

        embedding = embedding_response
        
        logger.info(f"New embedding length: {len(embedding)}")
        
        # Tạo tên collection để thao tác
        collection = Collection(self.milvus_client.collection_name)
        
        # Tìm kiếm face_id cũ
        results = collection.search(
            data=[embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=1,
            output_fields=["face_id"]
        )
        
        # Nếu không tìm thấy hoặc không đủ điểm tương đồng
        if not results:
            return {
                "statusCode": 400,
                "message": "face_id not found",
                "data": None
            }

        # Nếu đã tìm thấy face_id, xóa bản ghi cũ
        collection.delete(expr=f"face_id == '{face_id}'")
        
        # Tạo lại ID mới và timestamp cho face
        new_face_id = str(uuid.uuid4())
        created_at = datetime.now()
        
        # Thêm dữ liệu mới vào collection
        collection.insert([
            [embedding],
            [new_face_id],
            [created_at.isoformat()]
        ])
        
        collection.flush()  # Lưu các thay đổi
        
        return {
            "statusCode": 200,
            "message": "Face updated successfully",
            "data": {"face_id": new_face_id}
        }
        
    def delete_face_by_id(self, face_id: str):
        collection = Collection(self.milvus_client.collection_name)
        
        # Expression to match the specific face_id
        expr = f"face_id == '{face_id}'"
        delete_result = collection.delete(expr)
        
        print(f"Delete result: {delete_result}")
        
        if delete_result.delete_count > 0:
            return {
                "statusCode": 200,
                "message": f"Face with face_id {face_id} has been deleted from the database",
                "data": None
            }
        else:
            return {
                "statusCode": 400,
                "message": f"Face with face_id {face_id} not found",
                "data": None
            }

    def delete_all_faces(self):
        collection = Collection(self.milvus_client.collection_name)
        
        expr = "face_id != ''" 
        collection.delete(expr)
        
        return {
            "statusCode": 200,
            "message": "All faces have been deleted from the database",
            "data": None
        }
