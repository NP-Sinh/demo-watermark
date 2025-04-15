import cv2
import numpy as np
import math
import hashlib
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from cryptography.fernet import Fernet
import base64
import json
import pywt
from typing import Tuple, List, Dict, Any

def generate_crypto_key(key: int) -> bytes:
    """Tạo khóa mã hóa từ số nguyên"""
    seed = str(key).encode()
    # Tạo 32 byte key từ seed bằng cách sử dụng SHA256
    hashed = hashlib.sha256(seed).digest()
    return base64.urlsafe_b64encode(hashed)

def _calculate_block_variance(block: np.ndarray) -> float:
    """Tính phương sai của một block"""
    return np.var(block)

def _find_significant_blocks(img: np.ndarray, block_size: int, threshold_ratio: float = 0.5) -> List[Tuple[int, int]]:
    """
    Tìm các block có độ phức tạp cao (phương sai lớn)
    
    Parameters:
    -----------
    img: Ảnh đầu vào (grayscale)
    block_size: Kích thước block
    threshold_ratio: Tỉ lệ số block được chọn so với tổng số block
    
    Returns:
    --------
    positions: Danh sách vị trí (hàng, cột) của các block đáng kể
    """
    h, w = img.shape[:2]
    blocks_h, blocks_w = h // block_size, w // block_size
    
    # Tính phương sai cho mỗi block
    variance_map = []
    for i in range(blocks_h):
        for j in range(blocks_w):
            block = img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            variance = _calculate_block_variance(block)
            variance_map.append((variance, (i, j)))
    
    # Sắp xếp theo phương sai giảm dần
    variance_map.sort(reverse=True)
    
    # Chọn các block có phương sai cao nhất
    num_blocks = int(len(variance_map) * threshold_ratio)
    significant_blocks = [pos for _, pos in variance_map[:num_blocks]]
    
    return significant_blocks

def _modify_wu_lee_block(block: np.ndarray, watermark_bit: int, alpha: float) -> np.ndarray:
    """
    Thực hiện thuật toán Wu-Lee trên một block
    
    Parameters:
    -----------
    block: Block ảnh gốc
    watermark_bit: Bit watermark (0 hoặc 1)
    alpha: Cường độ nhúng
    
    Returns:
    --------
    modified_block: Block sau khi nhúng watermark
    """
    # Chia block thành 4 vùng bằng nhau
    h, w = block.shape
    h_half, w_half = h // 2, w // 2
    
    A = block[:h_half, :w_half].astype(float)
    B = block[:h_half, w_half:].astype(float)
    C = block[h_half:, :w_half].astype(float)
    D = block[h_half:, w_half:].astype(float)
    
    # Tính giá trị trung bình
    mean_A = np.mean(A)
    mean_B = np.mean(B)
    mean_C = np.mean(C)
    mean_D = np.mean(D)
    
    # Thực hiện thuật toán Wu-Lee
    modified_block = block.copy().astype(float)
    
    if watermark_bit == 1:
        # Nếu bit watermark là 1, tăng chênh lệch giữa A,D và B,C
        modified_block[:h_half, :w_half] = A + alpha     # A+
        modified_block[h_half:, w_half:] = D + alpha     # D+
        modified_block[:h_half, w_half:] = B - alpha     # B-
        modified_block[h_half:, :w_half] = C - alpha     # C-
    else:
        # Nếu bit watermark là 0, tăng chênh lệch giữa B,D và A,C
        modified_block[:h_half, :w_half] = A - alpha     # A-
        modified_block[h_half:, w_half:] = D - alpha     # D-
        modified_block[:h_half, w_half:] = B + alpha     # B+
        modified_block[h_half:, :w_half] = C + alpha     # C+
    
    # Giới hạn các giá trị pixel trong khoảng [0, 255]
    modified_block = np.clip(modified_block, 0, 255)
    
    return modified_block.astype(np.uint8)

def calculate_adaptive_threshold(block: np.ndarray) -> float:
    """
    Tính ngưỡng thích nghi cho block
    
    Parameters:
    -----------
    block: Block ảnh cần tính ngưỡng
    
    Returns:
    --------
    threshold: Giá trị ngưỡng
    """
    # Sử dụng phương sai của block để điều chỉnh ngưỡng
    block_variance = np.var(block)
    # Ngưỡng càng cao nếu block càng phức tạp (phương sai lớn)
    return 0.1 * math.sqrt(block_variance)

def _extract_wu_lee_bit_improved(block: np.ndarray) -> int:
    """
    Trích xuất bit watermark từ một block với ngưỡng thích nghi
    
    Parameters:
    -----------
    block: Block ảnh đã nhúng watermark
    
    Returns:
    --------
    bit: Bit watermark trích xuất (0 hoặc 1)
    """
    # Chia block thành 4 vùng bằng nhau
    h, w = block.shape
    h_half, w_half = h // 2, w // 2
    
    A = block[:h_half, :w_half].astype(float)
    B = block[:h_half, w_half:].astype(float)
    C = block[h_half:, :w_half].astype(float)
    D = block[h_half:, w_half:].astype(float)
    
    # Tính giá trị trung bình
    mean_A = np.mean(A)
    mean_B = np.mean(B)
    mean_C = np.mean(C)
    mean_D = np.mean(D)
    
    # Tính ngưỡng thích nghi
    threshold = calculate_adaptive_threshold(block)
    
    # Tính chênh lệch
    diff = (mean_A + mean_D) - (mean_B + mean_C)
    
    # Nếu chênh lệch > ngưỡng, bit là 1; ngược lại, bit là 0
    return 1 if diff > threshold else 0

def _extract_wu_lee_bit(block: np.ndarray) -> int:
    """
    Trích xuất bit watermark từ một block
    
    Parameters:
    -----------
    block: Block ảnh đã nhúng watermark
    
    Returns:
    --------
    bit: Bit watermark trích xuất (0 hoặc 1)
    """
    # Chia block thành 4 vùng bằng nhau
    h, w = block.shape
    h_half, w_half = h // 2, w // 2
    
    A = block[:h_half, :w_half].astype(float)
    B = block[:h_half, w_half:].astype(float)
    C = block[h_half:, :w_half].astype(float)
    D = block[h_half:, w_half:].astype(float)
    
    # Tính giá trị trung bình
    mean_A = np.mean(A)
    mean_B = np.mean(B)
    mean_C = np.mean(C)
    mean_D = np.mean(D)
    
    # Tính chênh lệch
    diff1 = (mean_A + mean_D) - (mean_B + mean_C)
    
    # Nếu chênh lệch dương, bit là 1; ngược lại, bit là 0
    return 1 if diff1 > 0 else 0

def _prepare_watermark(watermark: np.ndarray, num_blocks: int) -> np.ndarray:
    """
    Chuẩn bị watermark để nhúng
    
    Parameters:
    -----------
    watermark: Ảnh watermark gốc
    num_blocks: Số block có thể nhúng watermark
    
    Returns:
    --------
    binary_watermark: Mảng các bit watermark
    """
    # Chuyển watermark sang grayscale nếu cần
    if len(watermark.shape) > 2:
        watermark_gray = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
    else:
        watermark_gray = watermark
    
    # Chuẩn hóa về kích thước phù hợp với số block
    target_size = math.ceil(math.sqrt(num_blocks))
    resized_wm = cv2.resize(watermark_gray, (target_size, target_size))
    
    # Nhị phân hóa watermark
    _, binary_wm = cv2.threshold(resized_wm, 127, 1, cv2.THRESH_BINARY)
    return binary_wm

def _save_metadata(watermark: np.ndarray, key: int, cover_img_name: str, 
                  block_size: int = 8, alpha: float = 5.0,
                  significant_blocks: List[Tuple[int, int]] = None):
    """
    Lưu metadata của watermark để trích xuất sau này
    
    Parameters:
    -----------
    watermark: Ảnh watermark
    key: Khóa dùng để nhúng
    cover_img_name: Tên ảnh gốc
    block_size: Kích thước block
    alpha: Cường độ nhúng
    significant_blocks: Danh sách các block đáng kể
    """
    # Tạo thư mục lưu metadata nếu chưa tồn tại
    os.makedirs("WU_LEE", exist_ok=True)
    
    # Tạo metadata
    metadata = {
        "shape": watermark.shape,
        "hash": hashlib.md5(watermark.tobytes()).hexdigest(),
        "block_size": block_size,
        "alpha": alpha,
        "significant_blocks": significant_blocks
    }
    
    # Mã hóa metadata
    crypto_key = generate_crypto_key(key)
    fernet = Fernet(crypto_key)
    encrypted_data = fernet.encrypt(json.dumps(metadata).encode())
    
    # Lưu metadata
    with open(f"WU_LEE/{cover_img_name}_metadata.enc", "wb") as f:
        f.write(encrypted_data)

def _read_metadata(key: int, cover_img_name: str) -> Dict[str, Any]:
    """
    Đọc metadata của watermark
    
    Parameters:
    -----------
    key: Khóa dùng khi nhúng
    cover_img_name: Tên ảnh gốc
    
    Returns:
    --------
    metadata: Thông tin metadata
    """
    try:
        # Đọc dữ liệu mã hóa
        with open(f"WU_LEE/{cover_img_name}_metadata.enc", "rb") as f:
            encrypted_data = f.read()
        
        # Giải mã
        crypto_key = generate_crypto_key(key)
        fernet = Fernet(crypto_key)
        decrypted_data = fernet.decrypt(encrypted_data)
        
        # Parse JSON
        return json.loads(decrypted_data.decode())
    except Exception as e:
        print(f"Lỗi khi đọc metadata: {e}")
        return None

def embed_wu_lee(cover_img: np.ndarray, watermark: np.ndarray, key: int, 
                 cover_img_name: str = "cover", block_size: int = 8, 
                 alpha: float = 5.0) -> np.ndarray:
    """
    Nhúng watermark vào ảnh sử dụng phương pháp Wu-Lee
    
    Parameters:
    -----------
    cover_img: Ảnh gốc
    watermark: Ảnh watermark
    key: Khóa để sinh vị trí ngẫu nhiên
    cover_img_name: Tên ảnh gốc (để lưu metadata)
    block_size: Kích thước mỗi block (mặc định: 8x8)
    alpha: Cường độ nhúng (mặc định: 5.0)
    
    Returns:
    --------
    watermarked_img: Ảnh đã nhúng watermark
    """
    # Chuyển ảnh gốc sang grayscale nếu cần
    if len(cover_img.shape) > 2:
        cover_gray = cv2.cvtColor(cover_img, cv2.COLOR_BGR2GRAY)
    else:
        cover_gray = cover_img
    
    # Tạo bản sao để không thay đổi ảnh gốc
    watermarked_img = cover_img.copy()
    
    # Tìm các block có độ phức tạp cao
    significant_blocks = _find_significant_blocks(cover_gray, block_size)
    
    # Chuẩn bị watermark
    binary_watermark = _prepare_watermark(watermark, len(significant_blocks))
    watermark_flat = binary_watermark.flatten()
    
    # Nhúng watermark vào các block đáng kể
    for idx, (i, j) in enumerate(significant_blocks):
        if idx >= len(watermark_flat):
            break
            
        if len(cover_img.shape) > 2:  # Ảnh màu
            # Xử lý từng kênh màu
            for c in range(3):
                block = cover_img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, c]
                modified_block = _modify_wu_lee_block(block, watermark_flat[idx], alpha)
                watermarked_img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, c] = modified_block
        else:  # Ảnh grayscale
            block = cover_img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            modified_block = _modify_wu_lee_block(block, watermark_flat[idx], alpha)
            watermarked_img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = modified_block
    
    # Lưu metadata
    _save_metadata(binary_watermark, key, cover_img_name, block_size, alpha, significant_blocks)
    
    # Tính các metric chất lượng
    psnr_value = psnr(cover_img, watermarked_img)
    print(f"PSNR: {psnr_value:.2f} dB")
    
    if len(cover_img.shape) == 3:
        ssim_value = ssim(cover_img, watermarked_img, channel_axis=2)
    else:
        ssim_value = ssim(cover_img, watermarked_img)
    print(f"SSIM: {ssim_value:.4f}")
    
    mse_value = np.mean((cover_img.astype(float) - watermarked_img.astype(float)) ** 2)
    print(f"MSE: {mse_value:.2f}")
    
    return watermarked_img

def extract_wu_lee(watermarked_img: np.ndarray, key: int, 
                   cover_img_name: str = "cover") -> np.ndarray:
    """
    Trích xuất watermark từ ảnh đã nhúng
    
    Parameters:
    -----------
    watermarked_img: Ảnh đã nhúng watermark
    key: Khóa sử dụng khi nhúng
    cover_img_name: Tên ảnh gốc (để tìm metadata)
    
    Returns:
    --------
    watermark: Ảnh watermark đã trích xuất
    """
    # Đọc metadata
    metadata = _read_metadata(key, cover_img_name)
    if not metadata:
        raise ValueError("Không thể đọc metadata hoặc key không chính xác")
    
    block_size = metadata.get("block_size", 8)
    watermark_shape = metadata.get("shape")
    significant_blocks = metadata.get("significant_blocks")
    
    if not watermark_shape or not significant_blocks:
        raise ValueError("Không tìm thấy thông tin kích thước watermark hoặc vị trí các block")
    
    # Chuyển ảnh sang grayscale nếu cần
    if len(watermarked_img.shape) > 2:
        watermarked_gray = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2GRAY)
    else:
        watermarked_gray = watermarked_img
    
    # Khởi tạo mảng bit watermark
    extracted_bits = []
    
    # Trích xuất bit từ các block đáng kể
    for i, j in significant_blocks:
        block = watermarked_gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
        bit = _extract_wu_lee_bit(block)  # Sử dụng hàm gốc thay vì hàm improved
        extracted_bits.append(bit)
    
    # Chuyển mảng bit thành ảnh
    watermark_size = math.ceil(math.sqrt(len(extracted_bits)))
    extracted_watermark = np.zeros((watermark_size, watermark_size), dtype=np.uint8)
    
    for idx, bit in enumerate(extracted_bits):
        if idx >= watermark_size * watermark_size:
            break
        i, j = idx // watermark_size, idx % watermark_size
        extracted_watermark[i, j] = bit * 255
    
    # Resize về kích thước gốc
    h, w = watermark_shape[:2]
    extracted_watermark = cv2.resize(extracted_watermark, (w, h))
    
    return extracted_watermark

def calculate_ber(original, extracted):
    """Tính toán Bit Error Rate giữa watermark gốc và watermark trích xuất"""
    if original.shape != extracted.shape:
        # Resize về cùng kích thước
        extracted = cv2.resize(extracted, (original.shape[1], original.shape[0]))
    
    # Nhị phân hóa
    _, original_bin = cv2.threshold(original, 127, 1, cv2.THRESH_BINARY)
    _, extracted_bin = cv2.threshold(extracted, 127, 1, cv2.THRESH_BINARY)
    
    # Tính BER
    xor_result = np.bitwise_xor(original_bin, extracted_bin)
    error_bits = np.sum(xor_result)
    total_bits = original_bin.size
    
    return error_bits / total_bits

def test_robustness(cover_img, watermark, key, cover_img_name="cover", block_size=8, alpha=5.0):
    """Kiểm tra độ bền vững của thuật toán với các tấn công"""
    # Nhúng watermark
    watermarked = embed_wu_lee(cover_img, watermark, key, cover_img_name, block_size, alpha)
    
    # Trích xuất watermark (không có tấn công)
    extracted = extract_wu_lee(watermarked, key, cover_img_name)
    ber_original = calculate_ber(watermark, extracted)
    print(f"BER (không tấn công): {ber_original:.6f}")
    
    # Thêm nhiễu Gaussian
    noisy_img = watermarked.copy()
    noise = np.random.normal(0, 15, watermarked.shape).astype(np.uint8)
    noisy_img = cv2.add(noisy_img, noise)
    extracted_noisy = extract_wu_lee(noisy_img, key, cover_img_name)
    ber_noise = calculate_ber(watermark, extracted_noisy)
    print(f"BER (nhiễu Gaussian): {ber_noise:.6f}")
    
    # Nén JPEG
    jpeg_path = f"WU_LEE/temp_jpeg.jpg"
    cv2.imwrite(jpeg_path, watermarked, [cv2.IMWRITE_JPEG_QUALITY, 70])
    jpeg_img = cv2.imread(jpeg_path)
    extracted_jpeg = extract_wu_lee(jpeg_img, key, cover_img_name)
    ber_jpeg = calculate_ber(watermark, extracted_jpeg)
    print(f"BER (nén JPEG - 70%): {ber_jpeg:.6f}")
    os.remove(jpeg_path)
    
    # Lọc trung vị
    median_img = cv2.medianBlur(watermarked, 3)
    extracted_median = extract_wu_lee(median_img, key, cover_img_name)
    ber_median = calculate_ber(watermark, extracted_median)
    print(f"BER (lọc trung vị): {ber_median:.6f}")
    
    # Cắt ảnh
    cropped_img = watermarked.copy()
    h, w = cropped_img.shape[:2]
    cropped_img = cropped_img[10:h-10, 10:w-10]
    cropped_img = cv2.resize(cropped_img, (w, h))
    extracted_crop = extract_wu_lee(cropped_img, key, cover_img_name)
    ber_crop = calculate_ber(watermark, extracted_crop)
    print(f"BER (cắt ảnh): {ber_crop:.6f}")
    
    # Xoay ảnh
    rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), 2, 1)
    rotated_img = cv2.warpAffine(watermarked, rotation_matrix, (w, h))
    extracted_rot = extract_wu_lee(rotated_img, key, cover_img_name)
    ber_rot = calculate_ber(watermark, extracted_rot)
    print(f"BER (xoay 2 độ): {ber_rot:.6f}")
    
    # Trả về các kết quả để so sánh
    return {
        "Ảnh gốc": ber_original,
        "Nhiễu Gaussian": ber_noise,
        "Nén JPEG": ber_jpeg,
        "Lọc trung vị": ber_median,
        "Cắt ảnh": ber_crop,
        "Xoay ảnh": ber_rot
    }

# Ví dụ sử dụng:
if __name__ == "__main__":
    # Đọc ảnh gốc và ảnh watermark
    cover_img = cv2.imread("cover.png")
    watermark = cv2.imread("watermark.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Nhúng watermark
    key = 12345
    block_size = 16  # Block lớn hơn cho biến đổi hình học tốt hơn
    alpha = 8.0  # Tăng độ bền, giảm chất lượng hình ảnh
    threshold_ratio = 0.3  # Chọn ít block hơn nhưng chất lượng cao hơn
    
    watermarked = embed_wu_lee(cover_img, watermark, key, "cover", block_size, alpha)
    
    # Lưu ảnh đã nhúng watermark
    cv2.imwrite("WU_LEE/watermarked_wu_lee.png", watermarked)
    
    # Trích xuất watermark
    extracted = extract_wu_lee(watermarked, key)
    
    # Lưu ảnh watermark đã trích xuất
    cv2.imwrite("WU_LEE/extracted_wu_lee.jpg", extracted)
    
    # Kiểm tra độ bền vững
    test_results = test_robustness(cover_img, watermark, key, "cover", block_size, alpha)
    
    # In kết quả test
    print("\nKết quả đánh giá:")
    for attack, ber in test_results.items():
        print(f"{attack}: {ber:.6f}")
