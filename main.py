import cv2
import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from insightface.app import FaceAnalysis
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# 1. Khởi tạo Model (Chạy lần đầu sẽ tự tải model buffalo_l về)
# 'buffalo_l' là gói model lớn (Large) bao gồm cả detect (tìm mặt) và recognize (nhận diện).
# providers=['CUDAExecutionProvider']: Cấu hình chạy trên GPU NVIDIA.
ai_model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])

# ctx_id=0: Sử dụng GPU số 0. Nếu chạy CPU thì tham số này bị bỏ qua hoặc đặt < 0.
# det_size=(640, 640): Kích thước ảnh đầu vào được resize trước khi detect.
# 640x640 là chuẩn cân bằng giữa tốc độ và độ chính xác.
ai_model.prepare(ctx_id=0, det_size=(640, 640))

@app.post("/extract-user-face")
async def extract_user_face(file: UploadFile = File(...)):
    """
    API dùng cho User upload ảnh selfie để tìm kiếm.
    Chỉ lấy khuôn mặt to nhất/rõ nhất.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        faces = ai_model.get(img)
        
        if not faces:
            raise HTTPException(status_code=400, detail="No face detected")
            
        # Lấy mặt có diện tích lớn nhất (giả sử là người dùng)
        # Công thức tính diện tích hình chữ nhật: (Rộng) * (Cao)
        # Rộng = x2 - x1 (bbox[2] - bbox[0])
        # Cao  = y2 - y1 (bbox[3] - bbox[1])
        target_face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
        print(target_face.det_score)
        return {
            "code": 200,
            "status": "success",
            "embedding": target_face.embedding.tolist(),
            "det_score": float(target_face.det_score)
        }
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

class ImageUrlRequest(BaseModel):
    url: str

@app.post("/extract-faces-by-url")
async def extract_faces_by_url(request: ImageUrlRequest):
    """
    API nhận URL ảnh (từ Cloudinary), tự động tải và trích xuất khuôn mặt.
    """
    try:
        # 1. Tải ảnh từ Cloudinary URL
        response = requests.get(request.url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Cannot download image from URL")

        # 2. Chuyển đổi content tải về thành numpy array và giải mã bằng OpenCV
        nparr = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # 3. Chạy AI detect
        faces = ai_model.get(img)
        
        results = []
        for face in faces:
            results.append({
                "embedding": face.embedding.tolist(),  # Vector 512 chiều đại diện cho khuôn mặt
                "bbox": face.bbox.astype(int).tolist(), # [x1, y1, x2, y2]: để vẽ khung hình chữ nhật quanh mặt 
                "det_score": float(face.det_score) # # Độ tin cậy (ví dụ: 0.98 nghĩa là AI chắc chắn 98% đó là mặt người)
            })
            
        return {"code": 200, "status": "success", "faces": results}

    except Exception as e:
        return {"code": 500, "status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)