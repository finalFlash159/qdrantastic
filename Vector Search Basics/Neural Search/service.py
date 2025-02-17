# Deploy the search with FastAPI
#1. Install FastAPI.
# pip install fastapi uvicorn

from fastapi import FastAPI  # Import FastAPI để tạo ứng dụng API
from neural_searcher import NeuralSearcher  # Import lớp NeuralSearcher để xử lý tìm kiếm

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Tạo một thể hiện của NeuralSearcher với collection "startups"
neural_searcher = NeuralSearcher(collection_name="startups")

# Định nghĩa một API endpoint để tìm kiếm thông tin startup
@app.get("/api/search")  # Định nghĩa một route GET tại đường dẫn /api/search
def search_startup(q: str):
    """
    Endpoint tìm kiếm startup dựa trên truy vấn văn bản.
    - Nhận tham số 'q' từ URL (chuỗi văn bản cần tìm kiếm).
    - Gửi truy vấn đến NeuralSearcher để lấy kết quả.
    - Trả về kết quả dưới dạng JSON.
    """
    return {"result": neural_searcher.search(text=q)}  # Gọi phương thức search và trả về kết quả

# Chạy ứng dụng bằng Uvicorn nếu script được thực thi trực tiếp
if __name__ == "__main__":
    import uvicorn  # Import Uvicorn, server để chạy FastAPI

    # Chạy server trên tất cả các địa chỉ mạng (0.0.0.0) với cổng 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)