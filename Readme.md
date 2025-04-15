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