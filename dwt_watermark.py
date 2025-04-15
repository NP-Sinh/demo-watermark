import cv2
import numpy as np
import pywt
import os
import hashlib
import math
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union

def dwt_watermark_embed(
    cover_img: np.ndarray,
    watermark: np.ndarray,
    alpha: float = 0.1,
    wavelet: str = 'haar',
    level: int = 2,
    key: int = 42,
    subband: str = 'LL'
) -> np.ndarray:
    """
    Nhúng thủy vân vào ảnh gốc sử dụng biến đổi DWT (Discrete Wavelet Transform).
    
    Tham số:
        cover_img: Ảnh gốc (màu hoặc xám)
        watermark: Ảnh thủy vân (sẽ được nhị phân hóa)
        alpha: Cường độ nhúng (giá trị cao hơn làm thủy vân bền vững hơn nhưng có thể giảm chất lượng)
        wavelet: Loại wavelet sử dụng (haar, db1, db2, sym2, etc.)
        level: Số cấp độ phân tách DWT
        key: Khóa bí mật cho vị trí giả ngẫu nhiên
        subband: Dải con để nhúng ('LL', 'LH', 'HL', 'HH')
    
    Trả về:
        Ảnh đã nhúng thủy vân có cùng kích thước với ảnh gốc
    """
    # Kiểm tra đầu vào
    if cover_img.dtype != np.uint8 or watermark.dtype != np.uint8:
        raise ValueError("Ảnh đầu vào phải có kiểu dữ liệu uint8")
    
    # Thiết lập seed ngẫu nhiên
    np.random.seed(key)
    
    # Xử lý ảnh màu bằng cách làm việc trên kênh độ sáng
    if len(cover_img.shape) == 3:
        ycrcb = cv2.cvtColor(cover_img, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0].copy().astype(np.float32)
    else:
        y_channel = cover_img.copy().astype(np.float32)
    
    # Chuẩn bị thủy vân
    # Điều chỉnh kích thước thủy vân để phù hợp với DWT của ảnh gốc
    # Sau khi biến đổi DWT cấp 2, ảnh gốc sẽ giảm kích thước xuống 1/4
    h, w = y_channel.shape
    wm_height = h // (2**level)
    wm_width = w // (2**level)
    
    watermark_resized = cv2.resize(watermark, (wm_width, wm_height))
    _, watermark_bin = cv2.threshold(watermark_resized, 128, 1, cv2.THRESH_BINARY)
    watermark_bin = watermark_bin.astype(np.float32)
    
    # Lưu thông tin metadata
    os.makedirs("DWT", exist_ok=True)
    _save_metadata(watermark_bin, key, wavelet, level, subband, alpha)
    
    # Áp dụng DWT
    coeffs = pywt.wavedec2(y_channel, wavelet, level=level)
    
    # Chọn dải tần thích hợp để nhúng
    if level == 1:
        # Cấp 1: coeffs = [cA, (cH, cV, cD)]
        if subband == 'LL':
            target_band = coeffs[0]
        elif subband == 'LH':
            target_band = coeffs[1][0]
        elif subband == 'HL':
            target_band = coeffs[1][1]
        elif subband == 'HH':
            target_band = coeffs[1][2]
    else:
        # Cấp >= 2: coeffs = [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
        if subband == 'LL':
            target_band = coeffs[0]  # Lấy dải LL ở cấp cao nhất
        elif subband == 'LH':
            target_band = coeffs[1][0]  # Lấy dải LH ở cấp cao nhất
        elif subband == 'HL':
            target_band = coeffs[1][1]  # Lấy dải HL ở cấp cao nhất
        elif subband == 'HH':
            target_band = coeffs[1][2]  # Lấy dải HH ở cấp cao nhất
    
    # Nhúng thủy vân vào dải tần đã chọn
    h, w = target_band.shape
    wm_h, wm_w = watermark_bin.shape
    
    # Đảm bảo thủy vân nhỏ hơn dải tần mục tiêu
    if wm_h >= h or wm_w >= w:
        # Nếu thủy vân lớn hơn hoặc bằng dải tần, điều chỉnh kích thước lại
        new_wm_h = max(1, h // 2)
        new_wm_w = max(1, w // 2)
        watermark_bin = cv2.resize(watermark_bin, (new_wm_w, new_wm_h), interpolation=cv2.INTER_AREA)
        wm_h, wm_w = watermark_bin.shape
        print(f"Đã điều chỉnh kích thước thủy vân thành {new_wm_w}x{new_wm_h} để phù hợp với dải tần")
    
    # Đảm bảo có đủ không gian để đặt thủy vân
    if h <= wm_h or w <= wm_w:
        # Nếu dải tần không đủ lớn, chọn vị trí cố định (0,0)
        x_pos = 0
        y_pos = 0
        print("Dải tần quá nhỏ, sử dụng vị trí cố định (0,0)")
    else:
        # Tạo vị trí ngẫu nhiên trong dải tần để nhúng thủy vân
        x_positions = np.random.permutation(h - wm_h)[:1]
        y_positions = np.random.permutation(w - wm_w)[:1]
        
        x_pos = x_positions[0]
        y_pos = y_positions[0]
    
    # Nhúng thủy vân
    for i in range(wm_h):
        for j in range(wm_w):
            if watermark_bin[i, j] == 1:
                target_band[x_pos + i, y_pos + j] += alpha * abs(target_band[x_pos + i, y_pos + j])
            else:
                target_band[x_pos + i, y_pos + j] -= alpha * abs(target_band[x_pos + i, y_pos + j])
    
    # Cập nhật lại hệ số trong dải tần
    if level == 1:
        if subband == 'LL':
            coeffs[0] = target_band
        elif subband == 'LH':
            coeffs[1] = (target_band, coeffs[1][1], coeffs[1][2])
        elif subband == 'HL':
            coeffs[1] = (coeffs[1][0], target_band, coeffs[1][2])
        elif subband == 'HH':
            coeffs[1] = (coeffs[1][0], coeffs[1][1], target_band)
    else:
        if subband == 'LL':
            coeffs[0] = target_band
        elif subband == 'LH':
            coeffs[1] = (target_band, coeffs[1][1], coeffs[1][2])
        elif subband == 'HL':
            coeffs[1] = (coeffs[1][0], target_band, coeffs[1][2])
        elif subband == 'HH':
            coeffs[1] = (coeffs[1][0], coeffs[1][1], target_band)
    
    # Áp dụng nghịch đảo DWT
    watermarked_y = pywt.waverec2(coeffs, wavelet)
    
    # Lưu ý: Kích thước sau khi khôi phục có thể khác với kích thước ban đầu
    # Cần điều chỉnh để đảm bảo kích thước khớp với ảnh gốc
    orig_h, orig_w = y_channel.shape
    if watermarked_y.shape != (orig_h, orig_w):
        print(f"Điều chỉnh kích thước từ {watermarked_y.shape} về {(orig_h, orig_w)}")
        # Cắt nếu lớn hơn hoặc padding nếu nhỏ hơn
        if watermarked_y.shape[0] >= orig_h and watermarked_y.shape[1] >= orig_w:
            # Nếu lớn hơn thì cắt
            watermarked_y = watermarked_y[:orig_h, :orig_w]
        else:
            # Nếu nhỏ hơn thì tạo mảng mới và sao chép dữ liệu
            new_y = np.zeros((orig_h, orig_w), dtype=watermarked_y.dtype)
            h = min(watermarked_y.shape[0], orig_h)
            w = min(watermarked_y.shape[1], orig_w)
            new_y[:h, :w] = watermarked_y[:h, :w]
            watermarked_y = new_y

    # Chuyển về định dạng uint8
    watermarked_y = np.clip(watermarked_y, 0, 255).astype(np.uint8)
    
    # Tái tạo ảnh màu nếu cần
    if len(cover_img.shape) == 3:
        ycrcb[:, :, 0] = watermarked_y
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        result = watermarked_y
    
    # Tính toán và hiển thị các chỉ số chất lượng
    psnr_value = psnr(cover_img, result)
    print(f"PSNR: {psnr_value:.2f} dB")
    
    # Đánh giá độ giống nhau về cấu trúc, độ sáng, và độ tương phản
    ssim_value = ssim(cover_img, result, channel_axis=2 if len(cover_img.shape) == 3 else None)
    print(f"SSIM: {ssim_value:.4f}")
    
    # MSE (Mean Squared Error)
    mse_value = np.mean((cover_img.astype(np.float32) - result.astype(np.float32)) ** 2)
    print(f"MSE: {mse_value:.2f}")
    
    return result

def dwt_watermark_extract(
    watermarked_img: np.ndarray,
    original_watermark_shape: Optional[Tuple[int, int]] = None,
    key: int = 42,
    wavelet: str = 'haar',
    level: int = 2,
    subband: str = 'LL',
    alpha: float = 0.1
) -> np.ndarray:
    """
    Trích xuất thủy vân từ ảnh đã được nhúng.
    
    Tham số:
        watermarked_img: Ảnh chứa thủy vân
        original_watermark_shape: Kích thước ban đầu của thủy vân (nếu biết)
        key: Khóa bí mật đã sử dụng khi nhúng
        wavelet: Loại wavelet đã sử dụng
        level: Số cấp độ phân tách DWT đã sử dụng
        subband: Dải con đã nhúng ('LL', 'LH', 'HL', 'HH')
        alpha: Cường độ nhúng đã sử dụng
    
    Trả về:
        Ảnh thủy vân đã trích xuất
    """
    # Đọc metadata nếu không cung cấp tham số
    if original_watermark_shape is None or key is None or wavelet is None or level is None or subband is None or alpha is None:
        key, wavelet, level, subband, alpha, original_watermark_shape = _load_metadata()
    
    # Thiết lập seed ngẫu nhiên
    np.random.seed(key)
    
    # Chuyển ảnh sang kênh độ sáng nếu cần
    if len(watermarked_img.shape) == 3:
        y_channel = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    else:
        y_channel = watermarked_img.copy().astype(np.float32)
    
    # Áp dụng DWT
    coeffs = pywt.wavedec2(y_channel, wavelet, level=level)
    
    # Chọn dải tần đã sử dụng để nhúng
    if level == 1:
        if subband == 'LL':
            target_band = coeffs[0]
        elif subband == 'LH':
            target_band = coeffs[1][0]
        elif subband == 'HL':
            target_band = coeffs[1][1]
        elif subband == 'HH':
            target_band = coeffs[1][2]
    else:
        if subband == 'LL':
            target_band = coeffs[0]
        elif subband == 'LH':
            target_band = coeffs[1][0]
        elif subband == 'HL':
            target_band = coeffs[1][1]
        elif subband == 'HH':
            target_band = coeffs[1][2]
    
    # Tính toán kích thước thủy vân
    wm_h, wm_w = original_watermark_shape
    h, w = target_band.shape
    
    # Đảm bảo thủy vân nhỏ hơn dải tần mục tiêu
    if wm_h >= h or wm_w >= w:
        # Nếu thủy vân lớn hơn dải tần, điều chỉnh kích thước
        wm_h = max(1, h // 2)
        wm_w = max(1, w // 2)
        print(f"Điều chỉnh kích thước thủy vân trích xuất thành {wm_w}x{wm_h} để phù hợp với dải tần")

    # Đảm bảo có đủ không gian để trích xuất thủy vân
    if h <= wm_h or w <= wm_w:
        # Nếu dải tần không đủ lớn, chọn vị trí cố định (0,0)
        x_pos = 0
        y_pos = 0
        print("Dải tần quá nhỏ, sử dụng vị trí cố định (0,0)")
    else:
        # Tạo vị trí ngẫu nhiên trong dải tần để trích xuất thủy vân (giống khi nhúng)
        x_positions = np.random.permutation(h - wm_h)[:1]
        y_positions = np.random.permutation(w - wm_w)[:1]
        
        x_pos = x_positions[0]
        y_pos = y_positions[0]
    
    # Trích xuất thủy vân
    extracted_watermark = np.zeros((wm_h, wm_w), dtype=np.float32)
    reference_block = target_band[x_pos:x_pos+wm_h, y_pos:y_pos+wm_w].copy()

    # Tính giá trị ngưỡng tự động sử dụng phương pháp Otsu
    # thay vì chỉ dùng giá trị trung bình đơn giản
    # Chuẩn bị dữ liệu để phân ngưỡng
    flat_block = reference_block.flatten()
    flat_block = (flat_block - np.min(flat_block)) / (np.max(flat_block) - np.min(flat_block)) * 255
    flat_block = flat_block.astype(np.uint8)

    # Sử dụng phương pháp Otsu để tìm ngưỡng tối ưu
    ret, threshold = cv2.threshold(flat_block, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_threshold = ret / 255.0 * (np.max(reference_block) - np.min(reference_block)) + np.min(reference_block)

    # Sử dụng ngưỡng Otsu để phân loại
    for i in range(wm_h):
        for j in range(wm_w):
            if target_band[x_pos + i, y_pos + j] > otsu_threshold:
                extracted_watermark[i, j] = 1
            else:
                extracted_watermark[i, j] = 0

    # Chuyển về định dạng uint8 để hiển thị
    extracted_watermark = (extracted_watermark * 255).astype(np.uint8)

    # Áp dụng lọc nhiễu (median filter) để làm mịn kết quả
    extracted_watermark = cv2.medianBlur(extracted_watermark, 3)

    # Áp dụng ngưỡng để làm rõ ảnh
    _, extracted_binary = cv2.threshold(extracted_watermark, 127, 255, cv2.THRESH_BINARY)

    # Áp dụng các phép toán hình thái học để làm sạch thủy vân
    kernel = np.ones((3, 3), np.uint8)
    # Phép xói mòn để loại bỏ điểm nhiễu
    extracted_binary = cv2.erode(extracted_binary, kernel, iterations=1)
    # Phép giãn nở để phục hồi hình dạng
    extracted_binary = cv2.dilate(extracted_binary, kernel, iterations=1)
    # Phép đóng (closing) để khép các vùng gần nhau
    extracted_binary = cv2.morphologyEx(extracted_binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return extracted_binary

def _save_metadata(watermark: np.ndarray, key: int, wavelet: str, level: int, subband: str, alpha: float):
    """Lưu thông tin metadata cho quá trình trích xuất"""
    metadata_file = os.path.join("DWT", "watermark_metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write(f"key: {key}\n")
        f.write(f"wavelet: {wavelet}\n")
        f.write(f"level: {level}\n")
        f.write(f"subband: {subband}\n")
        f.write(f"alpha: {alpha}\n")
        f.write(f"shape: {watermark.shape[0]} {watermark.shape[1]}\n")
    
    # Lưu hash để xác minh tính toàn vẹn
    watermark_hash = hashlib.md5(watermark.tobytes()).hexdigest()
    hash_file = os.path.join("DWT", "watermark_hash.txt")
    with open(hash_file, 'w') as f:
        f.write(watermark_hash)

def _load_metadata():
    """Đọc thông tin metadata cho quá trình trích xuất"""
    metadata_file = os.path.join("DWT", "watermark_metadata.txt")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Không tìm thấy file metadata {metadata_file}")
    
    key = None
    wavelet = None
    level = None
    subband = None
    alpha = None
    shape = None
    
    with open(metadata_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("key:"):
                key = int(line.split(":")[1].strip())
            elif line.startswith("wavelet:"):
                wavelet = line.split(":")[1].strip()
            elif line.startswith("level:"):
                level = int(line.split(":")[1].strip())
            elif line.startswith("subband:"):
                subband = line.split(":")[1].strip()
            elif line.startswith("alpha:"):
                alpha = float(line.split(":")[1].strip())
            elif line.startswith("shape:"):
                shape_parts = line.split(":")[1].strip().split()
                shape = (int(shape_parts[0]), int(shape_parts[1]))
    
    return key, wavelet, level, subband, alpha, shape

def calculate_ber(original, extracted):
    """Tính toán Bit Error Rate giữa thủy vân gốc và thủy vân trích xuất"""
    original_bits = np.unpackbits(original.flatten())
    extracted_bits = np.unpackbits(extracted.flatten())
    min_len = min(len(original_bits), len(extracted_bits))
    return np.sum(original_bits[:min_len] != extracted_bits[:min_len]) / min_len

def evaluate_robustness(
    cover_img: np.ndarray,
    watermark: np.ndarray,
    alpha: float = 0.1,
    wavelet: str = 'haar',
    level: int = 2,
    key: int = 42,
    subband: str = 'LL',
    attacks: list = None
) -> dict:
    """
    Đánh giá độ bền vững của thủy vân trước các tấn công khác nhau.
    
    Tham số:
        cover_img: Ảnh gốc
        watermark: Ảnh thủy vân
        alpha, wavelet, level, key, subband: Các tham số nhúng
        attacks: Danh sách tấn công để đánh giá
    
    Trả về:
        Từ điển chứa kết quả BER cho từng loại tấn công
    """
    if attacks is None:
        attacks = [
            "jpeg_compression",
            "gaussian_noise",
            "salt_pepper_noise",
            "median_filter",
            "gaussian_blur",
            "rotation",
            "scaling",
            "cropping"
        ]
    
    # Nhúng thủy vân
    watermarked = dwt_watermark_embed(
        cover_img, watermark, alpha, wavelet, level, key, subband
    )
    
    # Điều chỉnh kích thước thủy vân để phù hợp với DWT
    h, w = cover_img.shape[:2] if len(cover_img.shape) == 3 else cover_img.shape
    wm_height = h // (2**level)
    wm_width = w // (2**level)
    watermark_resized = cv2.resize(watermark, (wm_width, wm_height))
    _, watermark_bin = cv2.threshold(watermark_resized, 128, 255, cv2.THRESH_BINARY)
    
    results = {}
    
    for attack in attacks:
        attacked_img = None
        
        # Thực hiện tấn công
        if attack == "jpeg_compression":
            # Lưu và nén bằng JPEG với chất lượng 70%
            temp_file = os.path.join("DWT", "temp_jpeg.jpg")
            cv2.imwrite(temp_file, watermarked, [cv2.IMWRITE_JPEG_QUALITY, 70])
            attacked_img = cv2.imread(temp_file)
        
        elif attack == "gaussian_noise":
            # Thêm nhiễu Gaussian
            noise = np.random.normal(0, 15, watermarked.shape).astype(np.float32)
            attacked_img = np.clip(watermarked + noise, 0, 255).astype(np.uint8)
        
        elif attack == "salt_pepper_noise":
            # Thêm nhiễu muối tiêu
            attacked_img = watermarked.copy()
            prob = 0.05
            thres = 1 - prob 
            for i in range(attacked_img.shape[0]):
                for j in range(attacked_img.shape[1]):
                    rdn = np.random.random()
                    if rdn < prob:
                        attacked_img[i][j] = 0
                    elif rdn > thres:
                        attacked_img[i][j] = 255
        
        elif attack == "median_filter":
            # Lọc trung vị
            attacked_img = cv2.medianBlur(watermarked, 5)
        
        elif attack == "gaussian_blur":
            # Làm mờ Gaussian
            attacked_img = cv2.GaussianBlur(watermarked, (5, 5), 0)
        
        elif attack == "rotation":
            # Xoay ảnh 5 độ và quay lại
            center = (watermarked.shape[1] // 2, watermarked.shape[0] // 2)
            rot_mat = cv2.getRotationMatrix2D(center, 5, 1.0)
            rotated = cv2.warpAffine(watermarked, rot_mat, (watermarked.shape[1], watermarked.shape[0]))
            rot_mat = cv2.getRotationMatrix2D(center, -5, 1.0)
            attacked_img = cv2.warpAffine(rotated, rot_mat, (watermarked.shape[1], watermarked.shape[0]))
        
        elif attack == "scaling":
            # Thu nhỏ và phóng to lại
            small = cv2.resize(watermarked, (watermarked.shape[1] // 2, watermarked.shape[0] // 2))
            attacked_img = cv2.resize(small, (watermarked.shape[1], watermarked.shape[0]))
        
        elif attack == "cropping":
            # Cắt và điền lại
            attacked_img = watermarked.copy()
            h, w = attacked_img.shape[:2]
            attacked_img[h//4:h//4*3, w//4:w//4*3] = 0
        
        if attacked_img is not None:
            # Trích xuất thủy vân
            extracted = dwt_watermark_extract(
                attacked_img, 
                original_watermark_shape=(wm_height, wm_width),
                key=key, 
                wavelet=wavelet, 
                level=level, 
                subband=subband, 
                alpha=alpha
            )
            
            # Tính BER
            ber = calculate_ber(watermark_bin, extracted)
            results[attack] = ber
            
            # Lưu kết quả
            os.makedirs(os.path.join("DWT", "attacked"), exist_ok=True)
            cv2.imwrite(os.path.join("DWT", "attacked", f"{attack}_img.png"), attacked_img)
            cv2.imwrite(os.path.join("DWT", "attacked", f"{attack}_extracted.png"), extracted)
    
    # In kết quả
    print("\n===== KẾT QUẢ ĐÁNH GIÁ ĐỘ BỀN VỮNG =====")
    for attack, ber in results.items():
        print(f"{attack}: BER = {ber:.6f}")
    
    return results

def plot_results(cover_img, watermark, watermarked, extracted, results=None):
    """
    Hiển thị kết quả nhúng và trích xuất thủy vân với giao diện đẹp mắt.
    
    Tham số:
        cover_img: Ảnh gốc
        watermark: Ảnh thủy vân gốc
        watermarked: Ảnh đã nhúng thủy vân
        extracted: Ảnh thủy vân đã trích xuất
        results: Kết quả đánh giá độ bền vững (nếu có)
    """
    # Tạo một bản sao của các ảnh để tránh thay đổi ảnh gốc
    cover_display = cover_img.copy()
    watermarked_display = watermarked.copy()
    watermark_display = watermark.copy()
    extracted_display = extracted.copy()
    
    # Tạo figure với nền màu trắng và viền đen
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    fig = plt.figure(figsize=(16, 12), dpi=120, facecolor='white', edgecolor='black', linewidth=2)
    
    # Sử dụng GridSpec để kiểm soát bố cục tốt hơn
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], 
                  wspace=0.25, hspace=0.25)
    
    # Thêm tiêu đề chính
    fig.suptitle('KẾT QUẢ THỦY VÂN DWT', fontsize=24, fontweight='bold', y=0.98)
    
    # Tạo function để thiết lập style cho mỗi subplot
    def setup_subplot(ax, img, title, cmap=None):
        # Thêm viền đậm cho subplot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)
        
        # Thêm background cho subplot
        ax.set_facecolor('#f0f0f0')
        
        # Hiển thị ảnh
        if cmap:
            img_handle = ax.imshow(img, cmap=cmap)
        else:
            if len(img.shape) == 3:
                img_handle = ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                img_handle = ax.imshow(img, cmap='gray')
        
        # Thêm tiêu đề với background đẹp
        ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
        ax.title.set_backgroundcolor('#f0f0f0')
        
        # Tắt các trục
        ax.axis('off')
        
        return img_handle
    
    # Hiển thị ảnh gốc
    ax1 = fig.add_subplot(gs[0, 0])
    setup_subplot(ax1, cover_display, 'Ảnh gốc')
    
    # Hiển thị thủy vân gốc
    ax2 = fig.add_subplot(gs[0, 1])
    setup_subplot(ax2, watermark_display, 'Thủy vân gốc', cmap='gray')
    
    # Hiển thị ảnh đã nhúng thủy vân
    ax3 = fig.add_subplot(gs[1, 0])
    setup_subplot(ax3, watermarked_display, 'Ảnh đã nhúng thủy vân')
    
    # Hiển thị thủy vân đã trích xuất
    ax4 = fig.add_subplot(gs[1, 1])
    setup_subplot(ax4, extracted_display, 'Thủy vân đã trích xuất', cmap='gray')
    
    # Cân bằng các subplot để có cùng tỷ lệ
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_aspect('equal')
    
    # Thêm thông tin PSNR và SSIM
    psnr_value = psnr(cover_img, watermarked)
    ssim_value = ssim(cover_img, watermarked, channel_axis=2 if len(cover_img.shape) == 3 else None)
    
    # Thêm một bảng thông tin ở dưới
    info_text = f'PSNR: {psnr_value:.2f} dB   |   SSIM: {ssim_value:.4f}'
    
    # Tạo một panel đẹp để chứa thông tin không có khung
    info_panel = plt.axes([0.25, 0.04, 0.5, 0.05], frameon=False)
    info_panel.text(0.5, 0.5, info_text, 
                 ha='center', va='center', 
                 fontsize=16, fontweight='bold', color='#003366')
    info_panel.axis('off')
    
    # Thêm watermark cho tác giả
    plt.figtext(0.95, 0.01, 'DWT Watermarking', 
                fontsize=10, color='gray', ha='right', fontstyle='italic')
    
    # Lưu hình ảnh với chất lượng cao
    plt.savefig(os.path.join("DWT", "results.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Hiển thị biểu đồ đánh giá độ bền vững nếu có
    if results:
        # Tạo figure mới cho biểu đồ đánh giá
        plt.figure(figsize=(14, 8), dpi=120, facecolor='white', 
                  edgecolor='black', linewidth=2)
        
        # Lấy tên các tấn công và giá trị BER
        attacks = list(results.keys())
        bers = list(results.values())
        
        # Tạo màu gradient đẹp cho các cột
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(attacks)))
        
        # Vẽ biểu đồ cột với style hiện đại
        ax = plt.gca()
        ax.set_facecolor('#f5f5f5')
        
        bars = plt.bar(attacks, bers, color=colors, width=0.65, 
                     edgecolor='black', linewidth=1.5, alpha=0.85)
        
        # Thêm giá trị lên đầu mỗi cột
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),  # 3 điểm offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Thêm tiêu đề và nhãn với style hiện đại
        plt.title('ĐÁNH GIÁ ĐỘ BỀN VỮNG CỦA THỦY VÂN DWT', 
                 fontsize=20, fontweight='bold', pad=20)
        
        plt.xlabel('LOẠI TẤN CÔNG', fontsize=16, fontweight='bold', labelpad=15)
        plt.ylabel('BER (BIT ERROR RATE)', fontsize=16, fontweight='bold', labelpad=15)
        
        # Tùy chỉnh lưới
        plt.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
        
        # Tùy chỉnh ticks
        plt.xticks(rotation=40, ha='right', fontsize=13, fontweight='bold')
        plt.yticks(fontsize=13)
        
        # Thêm đường viền cho biểu đồ
        for spine in ax.spines.values():
            spine.set_color('navy')
            spine.set_linewidth(1.5)
        
        # Thêm annotation giải thích BER
        plt.annotate('BER càng thấp càng tốt', xy=(0.5, 0.02), xycoords='figure fraction',
                   fontsize=12, ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join("DWT", "robustness.png"), dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # Đọc ảnh gốc và watermark
    cover = cv2.imread("cover.png", cv2.IMREAD_COLOR)
    watermark = cv2.imread("watermark.jpg", cv2.IMREAD_GRAYSCALE)
    
    if cover is None or watermark is None:
        raise FileNotFoundError("Không tìm thấy file ảnh.")
    
    # Tham số
    alpha = 0.1  # Cường độ nhúng
    wavelet = 'haar'  # Loại wavelet
    level = 2  # Cấp độ phân tách DWT
    key = 42  # Khóa bí mật
    subband = 'LL'  # Dải tần để nhúng
    
    # Nhúng thủy vân
    watermarked = dwt_watermark_embed(
        cover, watermark, alpha, wavelet, level, key, subband
    )
    
    # Lưu ảnh đã nhúng
    os.makedirs("DWT", exist_ok=True)
    cv2.imwrite(os.path.join("DWT", "watermarked.png"), watermarked)
    
    # Trích xuất thủy vân
    extracted = dwt_watermark_extract(
        watermarked, key=key, wavelet=wavelet, level=level, subband=subband, alpha=alpha
    )
    
    # Lưu ảnh thủy vân trích xuất
    cv2.imwrite(os.path.join("DWT", "extracted_watermark.png"), extracted)
    
    # Tính BER
    h, w = cover.shape[:2] if len(cover.shape) == 3 else cover.shape
    wm_height = h // (2**level)
    wm_width = w // (2**level)
    watermark_resized = cv2.resize(watermark, (wm_width, wm_height))
    _, watermark_bin = cv2.threshold(watermark_resized, 128, 255, cv2.THRESH_BINARY)
    
    ber_value = calculate_ber(watermark_bin, extracted)
    print(f"BER: {ber_value:.6f}")
    
    # Hiển thị kết quả
    plot_results(cover, watermark, watermarked, extracted)
    
    print("Hoàn tất!")
