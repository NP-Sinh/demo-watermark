# Digital Watermarking App

Nghiên cứu kỹ thuật thủy vân số và xây dựng ứng dụng bảo vệ bản quyền ảnh số
- Tìm hiểu về các kỹ thuật che giấu tập tin
- Tìm hiểu phương pháp và mô hình thủy vân số
- Tìm hiểu về các thuật toán thủy vân theo miền không gian ảnh (SW; WU- LEE; LBS; PCT,...)
- Tìm hiểu về các thuật toán thủy vân theo mền tần số (DCT; DWT)
- Xây dựng chương trình thử nghiệm cài đặt một số thuật toán thủy vân nhằm ứng dụng vào việc xác thực thông tin và bảo vệ bản quyền cho dữ liệu ảnh số

## Cài đặt
```bash
pip install -r requirements.txt
```
## Tham khảo bảng PSNR:
```bash
PSNR (dB)   | Đánh giá độ giống nhau
50          | 'Rất cao' – Không phân biệt bằng mắt
40 - 50     | 'Cao' – Khó thấy sự khác biệt
30 - 40     | 'Trung bình' – Có thể thấy chút khác biệt
30          | 'Thấp' – Thấy rõ sự biến dạng
```
## ✨ Tóm tắt:
```bash
Chỉ số  | Vai trò chính                         | Thang đo  | Ý nghĩa
PSNR    | Độ sai lệch về độ sáng                | dB        | Càng cao càng tốt
SSIM    | Độ tương đồng về cấu trúc hình ảnh    | 0 → 1     | Càng gần 1 càng tốt
MSE     | Sai số trung bình bình phương         | ≥ 0       | Càng thấp càng tốt
BER     | Tỉ lệ bit sai (cho watermark)         | 0 → 1     | Càng thấp càng tốt
```
## 🏷️ LSB - Kết quả:
```bash

Kết quả đã khả quan hơn nhiều và có một số điểm đáng chú ý:

1. **Cơ bản**: Cả hai phương pháp đều đạt BER = 0.000000, tức là hoàn hảo khi không có tác động bên ngoài.

2. **Nhiễu Gaussian**:
   - LSB thông thường: 0.16692
   - LSB thích nghi: 0.16512
   - LSB thích nghi tốt hơn một chút, tuy không đáng kể.

3. **Nén JPEG**:
   - Cả hai phương pháp đều yếu (BER ~0.499)
   - Đây là điểm yếu lớn nhất của LSB.

4. **Lọc trung vị**:
   - LSB thông thường: 0.43521
   - LSB thích nghi: 0.41290
   - LSB thích nghi tốt hơn khoảng 2%.

5. **Chất lượng ảnh**:
   - PSNR: 68.86 dB (rất tốt)
   - SSIM: ~1.0000 (hoàn hảo)

Kết luận: Phương pháp LSB thích nghi đã được sửa thành công và hoạt động tốt, có cải thiện nhẹ đối với nhiễu Gaussian và lọc trung vị, nhưng vẫn yếu với nén JPEG. Đây là giới hạn của LSB - nếu cần độ bền cao hơn, nên xem xét DCT hoặc DWT.

```

## 🏷️ Thuật toán Wu-lee:
Thực thi:
```bash
1. **Đọc các tập tin đầu vào**:
   - Đọc ảnh gốc (`cover.png`) - ảnh cần được bảo vệ bản quyền
   - Đọc ảnh thủy vân (`watermark.jpg`) - ảnh chứa thông tin nhận dạng/bản quyền

2. **Thiết lập tham số tối ưu**:
   - `key = 12345` - khóa bảo mật để mã hóa/giải mã
   - `block_size = 16` - kích thước block lớn hơn giúp tăng khả năng chống biến đổi hình học
   - `alpha = 8.0` - cường độ nhúng cao hơn giúp tăng khả năng chống nhiễu
   - `threshold_ratio = 0.3` - chỉ chọn 30% block có phương sai lớn nhất

3. **Quá trình nhúng thủy vân**:
   - Nhúng thủy vân vào ảnh gốc sử dụng thuật toán Wu-Lee
   - Lưu kết quả trong thư mục `WU_LEE`

4. **Trích xuất và kiểm tra**:
   - Trích xuất thủy vân từ ảnh đã nhúng
   - Lưu thủy vân đã trích xuất để kiểm tra bằng mắt

5. **Đánh giá độ bền vững**:
   - Kiểm tra khả năng chống lại các loại tấn công:
     * Nhiễu Gaussian
     * Nén JPEG
     * Lọc trung vị
     * Cắt ảnh
     * Xoay ảnh
   - Tính toán và in BER (Bit Error Rate) cho mỗi loại tấn công
```
Kết quả
```bash
'Chất lượng hình ảnh:'
- PSNR: 33.13 dB - Vẫn trong dải "Trung bình" (30-40 dB), chấp nhận được
- SSIM: 0.9278 - Khá cao, cấu trúc hình ảnh được bảo toàn tốt
- MSE: 31.64 - Cao hơn trước do tăng alpha, nhưng chấp nhận được

'Độ bền với các tấn công:'
- Không tấn công: 10.7% - Khá tốt, đã giảm so với trước
- **Nhiễu Gaussian: 17.1% - Cải thiện đáng kể (trước đó là 30.8%)**
- Nén JPEG: 10.9% - Rất tốt, gần như không ảnh hưởng
- Lọc trung vị: 13.5% - Cải thiện tốt (trước đó là 18.1%)
- Cắt ảnh: 51.7% - Vẫn kém
- Xoay 2 độ: 51.2% - Vẫn kém

'Đánh giá chung:'
- Việc tăng block_size lên 16 và alpha lên 8.0 đã cải thiện rõ rệt khả năng chống nhiễu và lọc
- Đánh đổi hợp lý giữa chất lượng ảnh và độ bền của thủy vân
- Thuật toán này phù hợp với ứng dụng không yêu cầu chống tấn công biến đổi hình học
- Khả năng chống nén JPEG rất tốt - đây là lợi thế lớn vì ảnh thường được lưu dưới dạng JPEG
```