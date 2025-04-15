# dct_watermark.py
"""
Kỹ thuật thủy vân ảnh số sử dụng Biến đổi Cosin Rời rạc (DCT)

Module này triển khai kỹ thuật thủy vân mạnh mẽ sử dụng miền DCT.
Thuật toán nhúng thông tin vào các hệ số tần số trung bình của các khối DCT,
tạo sự cân bằng tốt giữa tính ẩn và độ bền vững.

Lý thuyết:
- DCT biến đổi dữ liệu từ miền không gian sang các thành phần tần số
- Các khối DCT 8x8 được sử dụng rộng rãi trong nén ảnh (như JPEG)
- Hệ số tần số trung bình cung cấp độ bền vững mà không ảnh hưởng nhiều đến chất lượng hình ảnh
- Vị trí ngẫu nhiên sử dụng khóa tăng cường tính bảo mật
"""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import os


def dct_watermark_embed(
    cover_img: np.ndarray,
    watermark: np.ndarray,
    alpha: float = 0.1,
    key: int = 42,
    quantization: bool = True,
    block_selection: str = 'variance'
) -> np.ndarray:
    """
    Nhúng thủy vân vào ảnh gốc sử dụng biến đổi DCT.
    
    Tham số:
        cover_img: Ảnh gốc (màu hoặc xám)
        watermark: Ảnh thủy vân (sẽ được nhị phân hóa)
        alpha: Cường độ nhúng (giá trị cao hơn làm thủy vân bền vững hơn nhưng có thể giảm chất lượng)
        key: Khóa bí mật cho vị trí giả ngẫu nhiên
        quantization: Có sử dụng kỹ thuật nhúng dựa trên lượng tử hóa (bền vững hơn)
        block_selection: Phương pháp chọn khối DCT ('all', 'variance', 'texture')
    
    Trả về:
        Ảnh đã nhúng thủy vân có cùng kích thước với ảnh gốc
    """
    # Kiểm tra đầu vào
    if cover_img.dtype != np.uint8 or watermark.dtype != np.uint8:
        raise ValueError("Ảnh đầu vào phải có kiểu dữ liệu uint8")
    
    # Thiết lập lại seed ngẫu nhiên để đảm bảo tính tái lập
    np.random.seed(key)
    
    # Xử lý ảnh màu bằng cách làm việc trên kênh độ sáng
    if len(cover_img.shape) == 3:
        ycrcb = cv2.cvtColor(cover_img, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0].copy()
    else:
        y_channel = cover_img.copy()

    # Chuẩn bị thủy vân (điều chỉnh kích thước và nhị phân hóa)
    # Giữ thủy vân nhỏ để chất lượng tốt hơn
    wm_height = max(y_channel.shape[0] // 32, 16)  # Đảm bảo kích thước tối thiểu nhưng nhỏ hơn
    wm_width = max(y_channel.shape[1] // 32, 16)
    
    watermark_resized = cv2.resize(watermark, (wm_width, wm_height))
    # Áp dụng ngưỡng mạnh hơn để có giá trị nhị phân rõ ràng
    _, watermark_bin = cv2.threshold(watermark_resized, 128, 1, cv2.THRESH_BINARY)
    watermark_bin = watermark_bin.astype(np.float32)
    
    # In kích thước thủy vân để gỡ lỗi
    print(f"Kích thước ảnh gốc: {cover_img.shape}")
    print(f"Kích thước thủy vân sau khi điều chỉnh: {watermark_bin.shape}")
    
    # Lấy các khối phù hợp để nhúng dựa trên phân tích kết cấu
    suitable_blocks = None
    if block_selection == 'variance':
        suitable_blocks = _get_suitable_blocks(y_channel, threshold=15.0)
    elif block_selection == 'texture':
        suitable_blocks = _get_textured_blocks(y_channel)
    
    # Nhúng thủy vân
    watermarked_y = _process_dct_blocks(
        y_channel, 
        watermark_bin, 
        alpha, 
        'embed', 
        quantization=quantization,
        suitable_blocks=suitable_blocks
    )

    # Tái tạo ảnh màu nếu cần
    if len(cover_img.shape) == 3:
        ycrcb[:, :, 0] = watermarked_y
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        result = watermarked_y

    # Tính toán và hiển thị các chỉ số chất lượng
    print(f"[PSNR] {psnr(cover_img, result):.2f} dB")
    
    # Sửa tính toán SSIM cho ảnh nhỏ
    try:
        min_dim = min(cover_img.shape[:2])
        # Đảm bảo win_size là số lẻ và ít nhất là 3 (kích thước hợp lệ nhỏ nhất)
        if min_dim < 7:
            win_size = 3
        else:
            win_size = min(min_dim, 7)
            if win_size % 2 == 0:
                win_size -= 1  # Đảm bảo win_size là số lẻ

        # Xử lý kênh màu đúng cách
        if len(cover_img.shape) == 3:
            ssim_value = ssim(cover_img, result, channel_axis=2, win_size=win_size)
        else:
            ssim_value = ssim(cover_img, result, win_size=win_size)
        print(f"[SSIM] {ssim_value:.4f}")
    except Exception as e:
        print(f"Lỗi tính toán SSIM: {e}")
        print("Tiếp tục mà không có SSIM...")
    
    return np.clip(result, 0, 255).astype(np.uint8)


def dct_watermark_extract(
    watermarked_img: np.ndarray,
    watermark_shape: tuple,
    alpha: float = 0.1,
    key: int = 42,
    quantization: bool = True,
    block_selection: str = 'variance'
) -> np.ndarray:
    """
    Trích xuất thủy vân từ ảnh đã được nhúng.
    
    Tham số:
        watermarked_img: Ảnh chứa thủy vân
        watermark_shape: Kích thước dự kiến của thủy vân gốc
        alpha: Cường độ nhúng đã sử dụng khi nhúng
        key: Khóa bí mật đã sử dụng khi nhúng
        quantization: Có sử dụng kỹ thuật lượng tử hóa khi nhúng
        block_selection: Phương pháp chọn khối đã sử dụng khi nhúng
    
    Trả về:
        Ảnh thủy vân đã trích xuất
    """
    # Thiết lập lại seed ngẫu nhiên để đảm bảo tính tái lập
    np.random.seed(key)
    
    if len(watermarked_img.shape) == 3:
        y_channel = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    else:
        y_channel = watermarked_img.copy()
    
    # Tính toán kích thước cho thủy vân được trích xuất
    # Điều này phải khớp với kích thước đã sử dụng khi nhúng
    wm_height = max(y_channel.shape[0] // 32, 16)
    wm_width = max(y_channel.shape[1] // 32, 16)
    
    print(f"Dự kiến thủy vân có kích thước: {(wm_height, wm_width)}")
    
    # Lấy các khối phù hợp - phải khớp với khi nhúng
    suitable_blocks = None
    if block_selection == 'variance':
        suitable_blocks = _get_suitable_blocks(y_channel, threshold=15.0)
    elif block_selection == 'texture':
        suitable_blocks = _get_textured_blocks(y_channel)

    extracted = _process_dct_blocks(
        y_channel, 
        (wm_height, wm_width), 
        alpha, 
        'extract',
        quantization=quantization,
        suitable_blocks=suitable_blocks
    )
    
    # Áp dụng xử lý hậu kỳ để cải thiện kết quả trích xuất
    extracted_uint8 = (extracted * 255).astype(np.uint8)
    
    # Áp dụng ngưỡng mạnh hơn để loại bỏ nhiễu
    _, extracted_binary = cv2.threshold(extracted_uint8, 127, 255, cv2.THRESH_BINARY)
    
    # Áp dụng các phép toán hình thái học để làm sạch thủy vân đã trích xuất
    kernel = np.ones((3, 3), np.uint8)
    extracted_binary = cv2.morphologyEx(extracted_binary, cv2.MORPH_OPEN, kernel)
    
    return extracted_binary


def _get_suitable_blocks(
    img: np.ndarray,
    threshold: float = 15.0
) -> np.ndarray:
    """
    Tìm các khối có đủ độ biến thiên cho việc nhúng thủy vân mạnh mẽ.
    Các khối có độ biến thiên cao thường chứa nhiều kết cấu và
    ít nhạy cảm về mặt thị giác đối với các thay đổi.
    
    Tham số:
        img: Ảnh đầu vào
        threshold: Ngưỡng độ biến thiên
        
    Trả về:
        Mặt nạ boolean của các khối phù hợp (True nghĩa là phù hợp)
    """
    h, w = img.shape[:2]
    block_size = 8
    rows, cols = h // block_size, w // block_size
    
    suitable = np.zeros((rows, cols), dtype=bool)
    
    for i in range(rows):
        for j in range(cols):
            block = img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            var = np.var(block)
            suitable[i, j] = var > threshold
            
    return suitable


def _get_textured_blocks(
    img: np.ndarray,
    threshold: float = 25.0
) -> np.ndarray:
    """
    Tìm các khối có kết cấu sử dụng phát hiện cạnh.
    Các vùng có kết cấu phù hợp hơn cho việc nhúng thủy vân vì
    các thay đổi ít nhận thấy hơn.
    
    Tham số:
        img: Ảnh đầu vào
        threshold: Ngưỡng cường độ cạnh
        
    Trả về:
        Mặt nạ boolean của các khối có kết cấu (True nghĩa là có kết cấu)
    """
    h, w = img.shape[:2]
    block_size = 8
    rows, cols = h // block_size, w // block_size
    
    # Phát hiện cạnh
    edges = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
    edges = np.abs(edges)
    
    textured = np.zeros((rows, cols), dtype=bool)
    
    for i in range(rows):
        for j in range(cols):
            block = edges[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            edge_intensity = np.mean(block)
            textured[i, j] = edge_intensity > threshold
            
    return textured


def _process_dct_blocks(
    channel: np.ndarray,
    data: Union[np.ndarray, Tuple[int, int]],
    alpha: float,
    mode: str,
    quantization: bool = False,
    suitable_blocks: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Xử lý các khối DCT cho cả hai chế độ nhúng và trích xuất.
    
    Thuật toán chia ảnh thành các khối 8x8, áp dụng DCT cho mỗi khối,
    và điều chỉnh các hệ số tần số trung bình dựa trên dữ liệu thủy vân.
    
    Tham số:
        channel: Kênh ảnh cần xử lý
        data: Dữ liệu thủy vân hoặc kích thước cho trích xuất
        alpha: Cường độ nhúng
        mode: 'embed' hoặc 'extract'
        quantization: Có sử dụng nhúng dựa trên lượng tử hóa
        suitable_blocks: Mặt nạ boolean chỉ ra khối nào sử dụng
    
    Trả về:
        Kênh ảnh đã xử lý hoặc thủy vân đã trích xuất
    """
    block_size = 8  # Kích thước khối DCT tiêu chuẩn (giống như trong JPEG)
    result = np.zeros_like(channel, dtype=np.float32)
    
    # Nếu đang trích xuất, chuẩn bị mảng kết quả dựa trên kích thước thủy vân
    if mode == 'extract':
        if isinstance(data, tuple):
            result = np.zeros((data[0], data[1]), dtype=np.float32)
        else:
            result = np.zeros(data.shape, dtype=np.float32)
    
    # Sử dụng một hệ số tần số trung bình cụ thể - đáng tin cậy hơn việc sử dụng nhiều vị trí
    # Vị trí này được chọn vì tính ổn định và cân bằng giữa độ bền vững và tính ẩn
    target_pos = (1, 3)  # Hệ số tần số trung bình
    
    for i in range(0, channel.shape[0] - block_size + 1, block_size):
        for j in range(0, channel.shape[1] - block_size + 1, block_size):
            # Bỏ qua các khối không phù hợp nếu mặt nạ được cung cấp
            if suitable_blocks is not None:
                block_i, block_j = i // block_size, j // block_size
                if block_i < suitable_blocks.shape[0] and block_j < suitable_blocks.shape[1]:
                    if not suitable_blocks[block_i, block_j]:
                        # Nếu đang nhúng, sao chép khối gốc
                        if mode == 'embed':
                            result[i:i+block_size, j:j+block_size] = channel[i:i+block_size, j:j+block_size]
                        continue
            
            # Ánh xạ đến vị trí thủy vân
            watermark_i, watermark_j = i // block_size, j // block_size
            
            # Xử lý khối bằng DCT
            block = channel[i:i+block_size, j:j+block_size].astype(np.float32)
            dct_block = cv2.dct(block)
            
            if mode == 'embed':
                # Chỉ nhúng nếu chúng ta đang trong kích thước thủy vân
                if (isinstance(data, np.ndarray) and 
                    watermark_i < data.shape[0] and watermark_j < data.shape[1]):
                    
                    # Lấy bit thủy vân
                    watermark_bit = data[watermark_i, watermark_j]
                    
                    # Lấy hệ số DC làm tham chiếu
                    dc_coef = abs(dct_block[0, 0])
                    if dc_coef < 1:  # Tránh chia cho không hoặc giá trị rất nhỏ
                        dc_coef = 1.0
                    
                    if quantization:
                        # Kỹ thuật nhúng dựa trên lượng tử hóa cải tiến
                        q_step = alpha * 50  # Kích thước bước lớn hơn để phát hiện tốt hơn
                        
                        if watermark_bit > 0.5:
                            # Cho bit 1: Đảm bảo hệ số là dương với độ lớn alpha*dc_coef
                            dct_block[target_pos] = q_step * (2 * int(abs(dct_block[target_pos]) / (2*q_step)) + 1)
                        else:
                            # Cho bit 0: Đảm bảo hệ số là âm với độ lớn alpha*dc_coef
                            dct_block[target_pos] = -q_step * (2 * int(abs(dct_block[target_pos]) / (2*q_step)) + 1)
                    else:
                        # Kỹ thuật nhúng đơn giản dựa trên dấu với điều chỉnh mạnh
                        if watermark_bit > 0.5:
                            # Cho bit 1: Buộc hệ số là dương
                            dct_block[target_pos] = alpha * dc_coef
                        else:
                            # Cho bit 0: Buộc hệ số là âm
                            dct_block[target_pos] = -alpha * dc_coef
                
                # Chuyển lại về miền không gian
                processed_block = cv2.idct(dct_block)
                result[i:i+block_size, j:j+block_size] = processed_block
                
            elif mode == 'extract':
                # Chỉ trích xuất nếu chúng ta đang trong kích thước thủy vân dự kiến
                if ((isinstance(data, tuple) and watermark_i < data[0] and watermark_j < data[1]) or
                    (isinstance(data, np.ndarray) and watermark_i < data.shape[0] and watermark_j < data.shape[1])):
                    
                    if quantization:
                        # Phát hiện bit đơn giản dựa trên dấu hệ số
                        extracted_bit = 1 if dct_block[target_pos] > 0 else 0
                    else:
                        # Phát hiện bit đơn giản dựa trên dấu hệ số
                        extracted_bit = 1 if dct_block[target_pos] > 0 else 0
                    
                    # Lưu kết quả
                    if isinstance(data, tuple):
                        result[watermark_i, watermark_j] = extracted_bit
                    else:
                        result[watermark_i, watermark_j] = extracted_bit
    
    return result


def evaluate_robustness(
    cover_img: np.ndarray,
    watermark: np.ndarray,
    alpha: float = 0.1,
    key: int = 42,
    attacks: list = None,
    quantization: bool = True,
    block_selection: str = 'variance'
) -> dict:
    """
    Đánh giá độ bền vững của thủy vân đối với các tấn công phổ biến.
    
    Tham số:
        cover_img: Ảnh gốc
        watermark: Ảnh thủy vân
        alpha: Cường độ nhúng
        key: Khóa bí mật
        attacks: Danh sách tấn công cần đánh giá
        quantization: Có sử dụng nhúng dựa trên lượng tử hóa
        block_selection: Phương pháp chọn khối
        
    Trả về:
        Từ điển kết quả tấn công với điểm tương quan
    """
    if attacks is None:
        attacks = [
            'jpeg_compression', 
            'gaussian_noise', 
            'rotation', 
            'scaling', 
            'cropping'
        ]
    
    # Nhúng thủy vân
    watermarked = dct_watermark_embed(
        cover_img.copy(), 
        watermark.copy(), 
        alpha, 
        key,
        quantization=quantization,
        block_selection=block_selection
    )
    
    results = {}
    watermark_bin = (watermark > 128).astype(np.uint8)
    
    for attack in attacks:
        attacked_img = None
        
        if attack == 'jpeg_compression':
            # Nén JPEG với chất lượng 50
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            _, buffer = cv2.imencode('.jpg', watermarked, encode_param)
            attacked_img = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
            
        elif attack == 'gaussian_noise':
            # Thêm nhiễu Gaussian
            attacked_img = watermarked.copy()
            noise = np.random.normal(0, 15, attacked_img.shape).astype(np.float32)
            attacked_img = np.clip(attacked_img + noise, 0, 255).astype(np.uint8)
            
        elif attack == 'rotation':
            # Xoay 2 độ và quay lại
            h, w = watermarked.shape[:2]
            center = (w//2, h//2)
            M = cv2.getRotationMatrix2D(center, 2, 1.0)
            rotated = cv2.warpAffine(watermarked, M, (w, h))
            
            # Xoay lại
            M = cv2.getRotationMatrix2D(center, -2, 1.0)
            attacked_img = cv2.warpAffine(rotated, M, (w, h))
            
        elif attack == 'scaling':
            # Thu nhỏ và phóng to
            h, w = watermarked.shape[:2]
            scaled_down = cv2.resize(watermarked, (w//2, h//2))
            attacked_img = cv2.resize(scaled_down, (w, h))
            
        elif attack == 'cropping':
            # Cắt 10% từ mỗi cạnh và điều chỉnh kích thước lại
            h, w = watermarked.shape[:2]
            crop_h, crop_w = int(h*0.1), int(w*0.1)
            cropped = watermarked[crop_h:h-crop_h, crop_w:w-crop_w]
            attacked_img = cv2.resize(cropped, (w, h))
        
        # Trích xuất thủy vân từ ảnh bị tấn công
        extracted = dct_watermark_extract(
            attacked_img, 
            watermark.shape, 
            alpha, 
            key,
            quantization=quantization,
            block_selection=block_selection
        )
        
        # Tính toán độ tương đồng
        extracted_resized = cv2.resize(extracted, (watermark.shape[1], watermark.shape[0]))
        extracted_bin = (extracted_resized > 128).astype(np.uint8)
        
        # Tương quan chuẩn hóa
        correlation = np.corrcoef(watermark_bin.flatten(), extracted_bin.flatten())[0,1]
        
        # Tỷ lệ lỗi bit (BER)
        ber = np.sum(watermark_bin != extracted_bin) / watermark_bin.size
        
        results[attack] = {
            'correlation': correlation,
            'ber': ber,
            'extracted': extracted
        }
        
    return results


def plot_results(cover_img, watermark, watermarked, extracted, results=None):
    """
    Vẽ đồ thị kết quả thủy vân và tùy chọn kết quả tấn công.
    """
    plt.figure(figsize=(16, 10))
    
    # Ảnh gốc
    plt.subplot(231)
    plt.title('Ảnh gốc')
    plt.imshow(cv2.cvtColor(cover_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(232)
    plt.title('Thủy vân')
    plt.imshow(watermark, cmap='gray')
    plt.axis('off')
    
    plt.subplot(233)
    plt.title('Ảnh đã nhúng thủy vân')
    plt.imshow(cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(234)
    plt.title('Thủy vân trích xuất')
    plt.imshow(extracted, cmap='gray')
    plt.axis('off')
    
    # Nếu có kết quả tấn công
    if results:
        plt.figure(figsize=(15, 10))
        plt.suptitle('Trích xuất thủy vân sau các tấn công', fontsize=16)
        
        for i, (attack, data) in enumerate(results.items()):
            plt.subplot(2, 3, i+1)
            plt.title(f'{attack.replace("_", " ").title()}\nCorr: {data["correlation"]:.2f}, BER: {data["ber"]:.2f}')
            plt.imshow(data['extracted'], cmap='gray')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Ví dụ sử dụng
    import sys
    
    # Tạo thư mục DCT nếu nó chưa tồn tại
    output_dir = "DCT"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục {output_dir}")
    
    cover = cv2.imread("cover.png")
    watermark = cv2.imread("watermark.jpg", cv2.IMREAD_GRAYSCALE)
    
    if cover is None:
        print("Lỗi: Không thể tải ảnh gốc. Hãy chắc chắn rằng file cover.png tồn tại.")
        sys.exit(1)
    if watermark is None:
        print("Lỗi: Không thể tải ảnh thủy vân. Hãy chắc chắn rằng file watermark.jpg tồn tại.")
        sys.exit(1)
        
    # Đảm bảo thủy vân được nhị phân hóa đúng cách
    _, watermark = cv2.threshold(watermark, 128, 255, cv2.THRESH_BINARY)
    
    # Nhúng thủy vân với các tính năng nâng cao - sử dụng alpha lớn hơn để nhúng mạnh hơn
    embedded_img = dct_watermark_embed(
        cover, 
        watermark, 
        alpha=0.4,  # Tăng cường độ nhúng đáng kể
        key=42,
        quantization=True,
        block_selection='all'  # Sử dụng tất cả các khối để bao phủ tốt hơn
    )
    
    # Lưu ảnh vào thư mục DCT
    watermarked_path = os.path.join(output_dir, "watermarked_dct.jpg")
    cv2.imwrite(watermarked_path, embedded_img)
    print(f"Đã lưu ảnh đã nhúng thủy vân vào {watermarked_path}")
    
    # Trích xuất thủy vân
    extracted_watermark = dct_watermark_extract(
        embedded_img, 
        watermark.shape, 
        alpha=0.4,  # Giống như khi nhúng
        key=42,
        quantization=True,
        block_selection='all'  # Sử dụng tất cả các khối để bao phủ tốt hơn
    )
    
    # Lưu ảnh thủy vân trích xuất vào thư mục DCT
    extracted_path = os.path.join(output_dir, "extracted_watermark_dct.jpg")
    cv2.imwrite(extracted_path, extracted_watermark)
    print(f"Đã lưu ảnh thủy vân trích xuất vào {extracted_path}")
    
    # Tính toán độ tương đồng giữa thủy vân gốc và trích xuất
    # Điều chỉnh kích thước thủy vân để khớp với kích thước đã sử dụng trong nhúng/trích xuất
    wm_height = max(cover.shape[0] // 32, 16)
    wm_width = max(cover.shape[1] // 32, 16)
    watermark_resized = cv2.resize(watermark, (wm_width, wm_height))
    
    # Chuyển đổi cả hai sang nhị phân để so sánh tốt hơn
    _, watermark_bin = cv2.threshold(watermark_resized, 128, 255, cv2.THRESH_BINARY)
    
    # Tính toán các chỉ số
    correlation = np.corrcoef(watermark_bin.flatten(), extracted_watermark.flatten())[0,1]
    print(f"Tương quan giữa thủy vân gốc và trích xuất: {correlation:.4f}")
    
    # Tính tỷ lệ lỗi bit
    total_pixels = watermark_bin.size
    different_pixels = np.sum(watermark_bin != extracted_watermark)
    ber = different_pixels / total_pixels
    print(f"Tỷ lệ lỗi bit: {ber:.4f} ({different_pixels}/{total_pixels} pixels)")
    
    # Tính độ chính xác phần trăm
    accuracy = (1 - ber) * 100
    print(f"Độ chính xác trích xuất: {accuracy:.2f}%")
    
    # Lưu các phiên bản nhị phân để so sánh trực quan vào thư mục DCT
    watermark_bin_path = os.path.join(output_dir, "watermark_binary.jpg")
    extracted_bin_path = os.path.join(output_dir, "extracted_binary.jpg")
    
    cv2.imwrite(watermark_bin_path, watermark_bin)
    cv2.imwrite(extracted_bin_path, extracted_watermark)
    print(f"Đã lưu ảnh thủy vân nhị phân gốc và trích xuất vào {output_dir}")
    
    # Tùy chọn đánh giá độ bền vững - lưu kết quả vào thư mục DCT
    # results = evaluate_robustness(cover, watermark, alpha=0.4, key=42)
    # plot_results(cover, watermark, embedded_img, extracted_watermark, results)
    # plt.savefig(os.path.join(output_dir, "robustness_results.png"))
    
    print(f"Tất cả các kết quả đã được lưu vào thư mục {output_dir}")
    