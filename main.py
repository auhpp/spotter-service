import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from insightface.app import FaceAnalysis
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

@app.post("/extract-faces")
async def extract_faces(file: UploadFile = File(...)):
    """
    API dùng cho Admin upload ảnh sự kiện.
    Trả về danh sách tất cả các mặt trong ảnh.
    """
    try:
        # Đọc ảnh từ request
        contents = await file.read()               # 1. Đọc file upload dưới dạng bytes
        nparr = np.frombuffer(contents, np.uint8)  # 2. Chuyển bytes thành mảng số nguyên (numpy array)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # 3. Dùng OpenCV giải mã mảng đó thành ma trận ảnh (BGR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Chạy AI detect
        faces = ai_model.get(img)
        
        results = []
        for face in faces:
            results.append({
                "embedding": face.embedding.tolist(), # Vector 512 chiều đại diện cho khuôn mặt
                "bbox": face.bbox.astype(int).tolist(), # [x1, y1, x2, y2]: để vẽ khung hình chữ nhật quanh mặt 
                "det_score": float(face.det_score) # Độ tin cậy (ví dụ: 0.98 nghĩa là AI chắc chắn 98% đó là mặt người)
            })
            
        return {"code": 200, "status": "success", "faces": results}

    except Exception as e:
        return {"code": 500, "status": "error", "message": str(e)}

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)