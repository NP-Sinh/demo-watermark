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

## 🗿 Chạy chương trình
```bash
python main_watermarking_app.py
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
## Chất lượng hình ảnh:
'PSNR: 33.13 dB' - Mức trung bình (30-40 dB), có thể thấy chút khác biệt nhưng chấp nhận được
'SSIM: 0.9278' - Rất tốt, cấu trúc hình ảnh sau khi nhúng vẫn được bảo toàn cao
'MSE: 31.64' - Sai số trung bình ở mức hợp lý

## Khả năng chống tấn công (BER):
'Không tấn công: 10.66%' - Vẫn có sai số cơ bản, cần cải thiện
'Nhiễu Gaussian: 16.50%' - Khả năng chống nhiễu tương đối tốt
'Nén JPEG (70%): 10.89%' - Xuất sắc, hầu như không bị ảnh hưởng bởi nén
'Lọc trung vị: 13.49%' - Khá tốt, thủy vân vẫn giữ được thông tin
'Cắt ảnh: 51.69%' - Kém, không giữ được thông tin (gần 50% tương đương đoán ngẫu nhiên)
'Xoay 2 độ: 51.15%' - Kém, không chống được biến đổi hình học nhỏ

## Kết luận:
'Ưu điểm': Thuật toán Wu-Lee có sức mạnh đáng kể trong việc chống lại nén JPEG và nhiễu, phù hợp với ứng dụng thực tế khi ảnh thường được lưu ở định dạng nén.

'*Hạn chế': Rất yếu với các biến đổi hình học như cắt và xoay. Đây là hạn chế lớn nếu ứng dụng trong môi trường có thể xảy ra các biến đổi này.

'Đánh đổi': Thay đổi tham số (alpha=8.0, block_size=16) đã cải thiện được khả năng chống nhiễu so với cấu hình mặc định nhưng vẫn duy trì được chất lượng hình ảnh ở mức chấp nhận được.

'Hướng cải tiến': Kết hợp với kỹ thuật dựa trên điểm đặc trưng bất biến với phép biến đổi hình học để cải thiện khả năng chống tấn công cắt và xoay.
```

# 💻 Hướng dẫn sử dụng các thuật toán thủy vân ảnh
```bash
## 1. Thuật toán Wu-Lee
- **Kích thước khối**: 8-16 là tốt nhất cho hầu hết ảnh. Khối lớn (>16) ít hiển thị nhưng kém bền vững, khối nhỏ (4-8) bền vững hơn nhưng có thể ảnh hưởng đến chất lượng hình ảnh.
- **Alpha (cường độ)**: 5-10 là cân bằng tốt. Giá trị cao hơn (>10) tăng độ bền vững nhưng giảm chất lượng. Giá trị thấp hơn (<5) giữ chất lượng nhưng dễ mất khi chỉnh sửa ảnh.
- **Sử dụng khi**: Cần độ bền vững cao đối với các tác động như nén, lọc, và cắt ảnh.

## 2. Thuật toán LSB (Least Significant Bit)
- **Số bit LSB**: 1-2 bit cho ảnh thông thường. 3-4 bit cho dung lượng thủy vân lớn nhưng sẽ dễ nhìn thấy.
- **Chế độ**: "Cơ bản" đơn giản và nhanh. "Thích nghi" thông minh hơn, nhúng vào vùng chi tiết phức tạp để giảm khả năng phát hiện.
- **Sử dụng khi**: Cần nhúng nhiều dữ liệu hoặc cho ứng dụng steganography. Không phù hợp khi cần độ bền vững cao vì rất dễ bị mất khi chỉnh sửa ảnh.

## 3. Thuật toán DCT (Discrete Cosine Transform)
- **Kích thước khối**: 8x8 là tiêu chuẩn và tương thích với JPEG. Khối lớn hơn (16-32) có thể tốt hơn cho ảnh độ phân giải cao.
- **Hệ số nhúng**: 20-40 cung cấp cân bằng tốt. Giá trị cao (>50) cho độ bền vững tốt hơn nhưng có thể gây méo ảnh.
- **Sử dụng khi**: Cần khả năng chống nén JPEG tốt và độ bền vững trung bình đến cao.

## 4. Thuật toán DWT (Discrete Wavelet Transform)
- **Mức phân tách**: Mức 2 phù hợp với hầu hết ứng dụng. Mức 1 cho chất lượng cao hơn, mức 3 cho độ bền vững cao hơn.
- **Wavelet**: "haar" đơn giản và nhanh. "db1", "db2" cho kết quả mượt hơn. "sym2" và "coif1" phức tạp hơn nhưng có thể cải thiện chất lượng.
- **Sử dụng khi**: Cần độ bền vững cao đối với nhiều loại tấn công khác nhau và chất lượng hình ảnh tốt hơn so với DCT.

## 5. Thuật toán Spread Spectrum
- **Hệ số khuếch đại**: 0.05-0.2 phù hợp cho hầu hết ứng dụng. Giá trị cao hơn (>0.3) làm thủy vân bền vững hơn nhưng có thể làm giảm chất lượng hình ảnh.
- **Độ dài chuỗi PN**: 1000-5000 phù hợp cho hầu hết ứng dụng. Chuỗi dài hơn tăng độ an toàn nhưng cũng tăng thời gian xử lý.
- **Sử dụng khi**: Cần bảo mật cao và độ bền vững vượt trội đối với các tấn công cố ý (cắt xén, lọc, xoay ảnh).

## Lời khuyên khi chọn thuật toán:
- **Cho bảo vệ bản quyền**: Wu-Lee, DWT hoặc Spread Spectrum
- **Cho dữ liệu ẩn (steganography)**: LSB
- **Cho ảnh sẽ được chia sẻ online (JPEG)**: DCT hoặc DWT
- **Cho độ bền vững tối đa**: Spread Spectrum
- **Cho cân bằng giữa chất lượng và độ bền vững**: DWT với mức phân tách 2

Các tham số mặc định đã được tối ưu cho hầu hết các trường hợp sử dụng, chỉ nên điều chỉnh khi có yêu cầu cụ thể về chất lượng hoặc độ bền vững.

```