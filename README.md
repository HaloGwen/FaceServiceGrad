# Face Recognition API

## Start service

- Tải file weight từ [link](https://drive.google.com/file/d/1dhQgduyONLj1_8jQ5Di9BEt5mj9-QUdv/view?usp=sharing) rồi lưu vào đường dẫn `app/weights/face_extraction.pth`
- Sửa đường dẫn model trong file `.env` thành `MODEL_PATH="weights/face_extraction.pth"`
- Thay đổi threshold tùy thích trong file `.env` (default: 0.6)
- Chạy docker compose

```bash
cd app
docker compose up -d
```


## Enroll a new face

```bash
curl -X POST "http://localhost:8001/api/v1/enroll" \
-H "Content-Type: multipart/form-data" \
-F "file=@path/to/image.jpg"
```

# Check in a face

```bash
curl -X POST "http://localhost:8001/api/v1/check-in" \
-H "Content-Type: multipart/form-data" \
-F "file=@path/to/image.jpg"
```

## Run Attu for DB Management

```bash
docker run -p 8000:3000 -e MILVUS_URL=0.0.0.0:19530 zilliz/attu
```