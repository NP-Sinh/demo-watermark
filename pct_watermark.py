import numpy as np
import cv2
from sklearn.cluster import KMeans # type: ignore
import os

class PCTWatermark:
    def __init__(self):
        # Kích thước khối ảnh để xử lý
        self.block_size = 8
        # Độ mạnh của watermark (hệ số alpha)
        self.alpha = 0.1
        # Số lượng cụm để phân loại nội dung
        self.num_clusters = 3
        
    def create_pct(self, image_block):
        """Tạo Cây Nội dung Tri giác (PCT) cho một khối ảnh"""
        # Tính toán các đặc trưng của khối
        mean = np.mean(image_block)  # Giá trị trung bình
        std = np.std(image_block)    # Độ lệch chuẩn
        # Tính entropy của khối ảnh
        entropy = -np.sum(np.histogram(image_block, bins=256, density=True)[0] * 
                         np.log2(np.histogram(image_block, bins=256, density=True)[0] + 1e-10))
        return np.array([mean, std, entropy])
    
    def embed_watermark(self, cover_image_path, watermark_path):
        """Nhúng watermark sử dụng kỹ thuật PCT"""
        # Đọc ảnh gốc và ảnh watermark
        cover_image = cv2.imread(cover_image_path, cv2.IMREAD_GRAYSCALE)
        watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        # Điều chỉnh kích thước watermark cho phù hợp với ảnh gốc
        watermark = cv2.resize(watermark, (cover_image.shape[1], cover_image.shape[0]))
        
        # Tạo thư mục PCT để lưu kết quả nếu chưa tồn tại
        if not os.path.exists('PCT'):
            os.makedirs('PCT')
            
        h, w = cover_image.shape
        blocks = []
        features = []
        
        # Chia ảnh thành các khối và trích xuất đặc trưng
        for i in range(0, h-self.block_size+1, self.block_size):
            for j in range(0, w-self.block_size+1, self.block_size):
                block = cover_image[i:i+self.block_size, j:j+self.block_size]
                blocks.append((i, j, block))
                features.append(self.create_pct(block))
                
        # Phân cụm các khối dựa trên đặc trưng bằng thuật toán K-Means
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Tạo ảnh đã nhúng watermark
        watermarked = cover_image.copy()
        
        # Nhúng watermark dựa trên mức độ quan trọng của cụm
        for idx, (i, j, block) in enumerate(blocks):
            cluster = clusters[idx]
            # Điều chỉnh độ mạnh của watermark theo cụm
            strength = self.alpha * (cluster + 1) / self.num_clusters
            
            watermark_block = watermark[i:i+self.block_size, j:j+self.block_size]
            # Thêm watermark vào khối ảnh với độ mạnh tương ứng
            modified_block = block + strength * watermark_block
            
            # Đảm bảo giá trị pixel nằm trong khoảng hợp lệ [0, 255]
            modified_block = np.clip(modified_block, 0, 255)
            watermarked[i:i+self.block_size, j:j+self.block_size] = modified_block
            
        # Lưu ảnh đã nhúng watermark
        cv2.imwrite('PCT/watermarked_pct.png', watermarked)
        return watermarked
        
    def extract_watermark(self, watermarked_path, original_path):
        """Trích xuất watermark từ ảnh đã nhúng"""
        # Đọc ảnh đã nhúng watermark và ảnh gốc
        watermarked = cv2.imread(watermarked_path, cv2.IMREAD_GRAYSCALE)
        original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        
        # Trích xuất watermark bằng cách lấy hiệu của ảnh và chia cho hệ số alpha
        extracted = (watermarked.astype(float) - original.astype(float)) / self.alpha
        extracted = np.clip(extracted, 0, 255).astype(np.uint8)
        
        # Lưu watermark đã trích xuất
        cv2.imwrite('PCT/extracted_watermark_pct.png', extracted)
        return extracted

# Ví dụ sử dụng:
if __name__ == "__main__":
    pct = PCTWatermark()
    # Nhúng watermark
    watermarked = pct.embed_watermark("cover.png", "watermark.jpg")
    # Trích xuất watermark
    extracted = pct.extract_watermark("PCT/watermarked_pct.png", "cover.png")