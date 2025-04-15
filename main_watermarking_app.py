import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
from PIL import Image, ImageTk

# Import các thuật toán thủy vân
from wu_lee_watermark import embed_wu_lee, extract_wu_lee, test_robustness as test_wu_lee
from lsb_watermark import embed_lsb, extract_lsb, test_robustness as test_lsb
from dct_watermark import dct_watermark_embed, dct_watermark_extract, evaluate_robustness as test_dct
from dwt_watermark import dwt_watermark_embed as embed_dwt, dwt_watermark_extract as extract_dwt, evaluate_robustness as test_dwt
from sw_watermark import sw_watermark_embed as embed_sw, sw_watermark_extract as extract_sw, evaluate_robustness as test_sw

class MainWatermarkingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Thủy vân Ảnh")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Khởi tạo biến
        self.cover_image_path = tk.StringVar()
        self.watermark_path = tk.StringVar()
        self.key = tk.IntVar(value=random.randint(10000, 99999))
        self.block_size = tk.IntVar(value=8)
        self.alpha = tk.DoubleVar(value=5.0)
        
        # Algorithm selection
        self.algorithm = tk.StringVar(value="Wu-Lee")
        
        self.cover_img = None
        self.watermark_img = None
        self.watermarked_img = None
        self.extracted_watermark = None
        
        # Create the GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls
        control_frame = ttk.LabelFrame(main_frame, text="Điều khiển", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Thuật toán selection
        ttk.Label(control_frame, text="Chọn thuật toán:").grid(row=0, column=0, sticky=tk.W, pady=5)
        algorithms = ["Wu-Lee", "LSB", "DCT", "DWT", "Spread Spectrum"]
        algorithm_dropdown = ttk.Combobox(control_frame, textvariable=self.algorithm, values=algorithms, state="readonly", width=28)
        algorithm_dropdown.grid(row=0, column=1, columnspan=2, sticky=tk.W, pady=5)
        algorithm_dropdown.bind("<<ComboboxSelected>>", self.on_algorithm_change)
        
        # Image selection
        ttk.Label(control_frame, text="Ảnh gốc:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(control_frame, textvariable=self.cover_image_path, width=30).grid(row=1, column=1, pady=5)
        ttk.Button(control_frame, text="Chọn", command=self.browse_cover_image).grid(row=1, column=2, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Ảnh thủy vân:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(control_frame, textvariable=self.watermark_path, width=30).grid(row=2, column=1, pady=5)
        ttk.Button(control_frame, text="Chọn", command=self.browse_watermark).grid(row=2, column=2, padx=5, pady=5)
        
        # Parameters frame
        self.params_frame = ttk.LabelFrame(control_frame, text="Tham số", padding=10)
        self.params_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=10)
        
        # Common parameters
        ttk.Label(self.params_frame, text="Khóa bí mật:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(self.params_frame, textvariable=self.key, width=10).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Algorithm-specific parameters
        self.create_wu_lee_params() # Default parameters
        
        # Action buttons
        action_frame = ttk.Frame(control_frame)
        action_frame.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=10)
        
        ttk.Button(action_frame, text="Nhúng thủy vân", command=self.embed_watermark).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Trích xuất thủy vân", command=self.extract_watermark).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Kiểm tra độ bền vững", command=self.test_robustness).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Lưu kết quả", command=self.save_results).pack(side=tk.LEFT, padx=5)
        
        # Right panel for image display
        display_frame = ttk.LabelFrame(main_frame, text="Hiển thị ảnh", padding=10)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs for different views
        self.tabs = ttk.Notebook(display_frame)
        self.tabs.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Images
        self.images_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.images_tab, text="Hình ảnh")
        
        # Image frames
        self.cover_frame = ttk.LabelFrame(self.images_tab, text="Ảnh gốc")
        self.cover_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.NSEW)
        self.cover_canvas = tk.Canvas(self.cover_frame, width=350, height=350, bg="#eeeeee")
        self.cover_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.watermark_frame = ttk.LabelFrame(self.images_tab, text="Thủy vân")
        self.watermark_frame.grid(row=0, column=1, padx=5, pady=5, sticky=tk.NSEW)
        self.watermark_canvas = tk.Canvas(self.watermark_frame, width=350, height=350, bg="#eeeeee")
        self.watermark_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.watermarked_frame = ttk.LabelFrame(self.images_tab, text="Ảnh đã nhúng thủy vân")
        self.watermarked_frame.grid(row=1, column=0, padx=5, pady=5, sticky=tk.NSEW)
        self.watermarked_canvas = tk.Canvas(self.watermarked_frame, width=350, height=350, bg="#eeeeee")
        self.watermarked_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.extracted_frame = ttk.LabelFrame(self.images_tab, text="Thủy vân trích xuất")
        self.extracted_frame.grid(row=1, column=1, padx=5, pady=5, sticky=tk.NSEW)
        self.extracted_canvas = tk.Canvas(self.extracted_frame, width=350, height=350, bg="#eeeeee")
        self.extracted_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        self.images_tab.grid_rowconfigure(0, weight=1)
        self.images_tab.grid_rowconfigure(1, weight=1)
        self.images_tab.grid_columnconfigure(0, weight=1)
        self.images_tab.grid_columnconfigure(1, weight=1)
        
        # Tab 2: Metrics
        self.metrics_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.metrics_tab, text="Đánh giá")
        
        self.metrics_frame = ttk.Frame(self.metrics_tab)
        self.metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Sẵn sàng")
        self.status_bar = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=2)
    
    def on_algorithm_change(self, event=None):
        # Xóa tham số cũ
        for widget in self.params_frame.winfo_children():
            if widget not in [self.params_frame.grid_slaves(row=0, column=0)[0], self.params_frame.grid_slaves(row=0, column=1)[0]]:
                widget.destroy()
        
        # Tạo tham số mới dựa trên thuật toán được chọn
        algorithm = self.algorithm.get()
        if algorithm == "Wu-Lee":
            self.create_wu_lee_params()
        elif algorithm == "LSB":
            self.create_lsb_params()
        elif algorithm == "DCT":
            self.create_dct_params()
        elif algorithm == "DWT":
            self.create_dwt_params()
        elif algorithm == "Spread Spectrum":
            self.create_sw_params()
    
    def create_wu_lee_params(self):
        ttk.Label(self.params_frame, text="Kích thước khối:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Scale(self.params_frame, from_=4, to=32, variable=self.block_size, orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, sticky=tk.W, pady=5)
        ttk.Label(self.params_frame, textvariable=self.block_size).grid(row=1, column=2, sticky=tk.W, pady=5)
        
        ttk.Label(self.params_frame, text="Alpha (Cường độ):").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Scale(self.params_frame, from_=1.0, to=20.0, variable=self.alpha, orient=tk.HORIZONTAL, length=200).grid(row=2, column=1, sticky=tk.W, pady=5)
        ttk.Label(self.params_frame, textvariable=self.alpha).grid(row=2, column=2, sticky=tk.W, pady=5)
    
    def create_lsb_params(self):
        ttk.Label(self.params_frame, text="Số bit LSB:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.lsb_bits = tk.IntVar(value=2)
        ttk.Scale(self.params_frame, from_=1, to=8, variable=self.lsb_bits, orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, sticky=tk.W, pady=5)
        ttk.Label(self.params_frame, textvariable=self.lsb_bits).grid(row=1, column=2, sticky=tk.W, pady=5)
        
        ttk.Label(self.params_frame, text="Chế độ:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.lsb_mode = tk.StringVar(value="Cơ bản")
        modes = ["Cơ bản", "Thích nghi"]
        ttk.Combobox(self.params_frame, textvariable=self.lsb_mode, values=modes, state="readonly", width=15).grid(row=2, column=1, sticky=tk.W, pady=5)
    
    def create_dct_params(self):
        ttk.Label(self.params_frame, text="Kích thước khối:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Scale(self.params_frame, from_=8, to=32, variable=self.block_size, orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, sticky=tk.W, pady=5)
        ttk.Label(self.params_frame, textvariable=self.block_size).grid(row=1, column=2, sticky=tk.W, pady=5)
        
        ttk.Label(self.params_frame, text="Hệ số nhúng:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.dct_factor = tk.DoubleVar(value=25.0)
        ttk.Scale(self.params_frame, from_=5.0, to=100.0, variable=self.dct_factor, orient=tk.HORIZONTAL, length=200).grid(row=2, column=1, sticky=tk.W, pady=5)
        ttk.Label(self.params_frame, textvariable=self.dct_factor).grid(row=2, column=2, sticky=tk.W, pady=5)
    
    def create_dwt_params(self):
        ttk.Label(self.params_frame, text="Mức phân tách:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.dwt_level = tk.IntVar(value=2)
        ttk.Scale(self.params_frame, from_=1, to=3, variable=self.dwt_level, orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, sticky=tk.W, pady=5)
        ttk.Label(self.params_frame, textvariable=self.dwt_level).grid(row=1, column=2, sticky=tk.W, pady=5)
        
        ttk.Label(self.params_frame, text="Wavelet:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.dwt_wavelet = tk.StringVar(value="haar")
        wavelets = ["haar", "db1", "db2", "sym2", "coif1"]
        ttk.Combobox(self.params_frame, textvariable=self.dwt_wavelet, values=wavelets, state="readonly", width=15).grid(row=2, column=1, sticky=tk.W, pady=5)
    
    def create_sw_params(self):
        ttk.Label(self.params_frame, text="Hệ số khuếch đại:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.sw_gain = tk.DoubleVar(value=0.1)
        ttk.Scale(self.params_frame, from_=0.01, to=1.0, variable=self.sw_gain, orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, sticky=tk.W, pady=5)
        ttk.Label(self.params_frame, textvariable=self.sw_gain).grid(row=1, column=2, sticky=tk.W, pady=5)
        
        ttk.Label(self.params_frame, text="Độ dài chuỗi PN:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.sw_length = tk.IntVar(value=1000)
        ttk.Scale(self.params_frame, from_=100, to=10000, variable=self.sw_length, orient=tk.HORIZONTAL, length=200).grid(row=2, column=1, sticky=tk.W, pady=5)
        ttk.Label(self.params_frame, textvariable=self.sw_length).grid(row=2, column=2, sticky=tk.W, pady=5)
    
    def browse_cover_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Tệp ảnh", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.cover_image_path.set(file_path)
            self.load_cover_image()
    
    def browse_watermark(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Tệp ảnh", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.watermark_path.set(file_path)
            self.load_watermark_image()
    
    def load_cover_image(self):
        try:
            self.status_var.set("Đang tải ảnh gốc...")
            self.root.update_idletasks()
            
            # Load the image with OpenCV
            self.cover_img = cv2.imread(self.cover_image_path.get())
            
            # Display the image
            self.display_image(self.cover_img, self.cover_canvas)
            
            self.status_var.set("Ảnh gốc đã tải thành công")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải ảnh gốc: {str(e)}")
            self.status_var.set("Lỗi khi tải ảnh gốc")
    
    def load_watermark_image(self):
        try:
            self.status_var.set("Đang tải ảnh thủy vân...")
            self.root.update_idletasks()
            
            # Load the image with OpenCV
            self.watermark_img = cv2.imread(self.watermark_path.get())
            
            # Display the image
            self.display_image(self.watermark_img, self.watermark_canvas)
            
            self.status_var.set("Ảnh thủy vân đã tải thành công")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải ảnh thủy vân: {str(e)}")
            self.status_var.set("Lỗi khi tải ảnh thủy vân")
    
    def display_image(self, img, canvas):
        if img is None:
            return
        
        # Convert from BGR to RGB for display
        if len(img.shape) == 3:
            display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Resize image to fit in canvas while preserving aspect ratio
        canvas_width = canvas.winfo_width() or 350
        canvas_height = canvas.winfo_height() or 350
        
        h, w = display_img.shape[:2]
        ratio = min(canvas_width / w, canvas_height / h)
        new_size = (int(w * ratio), int(h * ratio))
        
        display_img = cv2.resize(display_img, new_size)
        
        # Convert to PIL format
        pil_img = Image.fromarray(display_img)
        
        # Convert to PhotoImage
        tk_img = ImageTk.PhotoImage(pil_img)
        
        # Update canvas
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=tk_img)
        canvas.image = tk_img  # Keep a reference to prevent garbage collection
    
    def embed_watermark(self):
        if self.cover_img is None or self.watermark_img is None:
            messagebox.showerror("Lỗi", "Vui lòng tải cả ảnh gốc và ảnh thủy vân trước")
            return
        
        try:
            self.status_var.set(f"Đang nhúng thủy vân với thuật toán {self.algorithm.get()}...")
            self.progress_var.set(10)
            self.root.update_idletasks()
            
            # Get common parameters
            key = self.key.get()
            cover_name = os.path.splitext(os.path.basename(self.cover_image_path.get()))[0]
            
            self.progress_var.set(30)
            self.root.update_idletasks()
            
            # Nhúng theo thuật toán được chọn
            algorithm = self.algorithm.get()
            
            if algorithm == "Wu-Lee":
                block_size = self.block_size.get()
                alpha = self.alpha.get()
                self.watermarked_img = embed_wu_lee(self.cover_img, self.watermark_img, key, cover_name, block_size, alpha)
                
                # Tạo thư mục nếu chưa có
                os.makedirs("WU_LEE", exist_ok=True)
                
            elif algorithm == "LSB":
                lsb_bits = self.lsb_bits.get()
                mode = "adaptive" if self.lsb_mode.get() == "Thích nghi" else "simple"
                self.watermarked_img = embed_lsb(self.cover_img, self.watermark_img, key, lsb_bits, mode)
                
                # Tạo thư mục nếu chưa có
                os.makedirs("LSB", exist_ok=True)
                
            elif algorithm == "DCT":
                # Đảm bảo thủy vân được nhị phân hóa đúng cách
                if len(self.watermark_img.shape) > 2:
                    watermark_gray = cv2.cvtColor(self.watermark_img, cv2.COLOR_BGR2GRAY)
                else:
                    watermark_gray = self.watermark_img.copy()
                    
                _, watermark_bin = cv2.threshold(watermark_gray, 128, 255, cv2.THRESH_BINARY)
                
                # Lấy các tham số DCT
                alpha = self.dct_factor.get() / 100.0  # Chuyển đổi thành phạm vi 0.05-1.0
                
                # Gọi hàm với tham số phù hợp
                self.watermarked_img = dct_watermark_embed(
                    self.cover_img, 
                    watermark_bin, 
                    alpha=alpha,
                    key=key,
                    quantization=True,
                    block_selection='variance'
                )
                
                # Tạo thư mục nếu chưa có
                os.makedirs("DCT", exist_ok=True)
                
            elif algorithm == "DWT":
                level = self.dwt_level.get()
                wavelet = self.dwt_wavelet.get()
                
                # Đảm bảo thủy vân được nhị phân hóa đúng cách
                if len(self.watermark_img.shape) > 2:
                    watermark_gray = cv2.cvtColor(self.watermark_img, cv2.COLOR_BGR2GRAY)
                else:
                    watermark_gray = self.watermark_img.copy()
                    
                _, watermark_bin = cv2.threshold(watermark_gray, 128, 255, cv2.THRESH_BINARY)
                
                self.watermarked_img = embed_dwt(
                    self.cover_img, 
                    watermark_bin, 
                    alpha=0.1, 
                    wavelet=wavelet, 
                    level=level, 
                    key=key
                )
                
                # Tạo thư mục nếu chưa có
                os.makedirs("DWT", exist_ok=True)
                
            elif algorithm == "Spread Spectrum":
                gain = self.sw_gain.get()
                length = self.sw_length.get()
                self.watermarked_img = embed_sw(
                    self.cover_img, 
                    self.watermark_img, 
                    alpha=gain,
                    key=key,
                    chip_rate=length
                )
                
                # Tạo thư mục nếu chưa có
                os.makedirs("SW", exist_ok=True)
            
            self.progress_var.set(70)
            self.root.update_idletasks()
            
            # Display watermarked image
            self.display_image(self.watermarked_img, self.watermarked_canvas)
            
            self.progress_var.set(100)
            self.status_var.set(f"Nhúng thủy vân thành công với thuật toán {algorithm}")
            
            # Switch to the Images tab
            self.tabs.select(0)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể nhúng thủy vân: {str(e)}")
            self.status_var.set("Lỗi khi nhúng thủy vân")
        finally:
            self.progress_var.set(0)
    
    def extract_watermark(self):
        if self.watermarked_img is None:
            messagebox.showerror("Lỗi", "Vui lòng nhúng thủy vân trước hoặc tải ảnh đã nhúng thủy vân")
            return
        
        try:
            self.status_var.set(f"Đang trích xuất thủy vân với thuật toán {self.algorithm.get()}...")
            self.progress_var.set(10)
            self.root.update_idletasks()
            
            # Get parameters
            key = self.key.get()
            cover_name = os.path.splitext(os.path.basename(self.cover_image_path.get()))[0]
            
            self.progress_var.set(30)
            self.root.update_idletasks()
            
            # Trích xuất theo thuật toán đã chọn
            algorithm = self.algorithm.get()
            
            if algorithm == "Wu-Lee":
                self.extracted_watermark = extract_wu_lee(self.watermarked_img, key, cover_name)
            elif algorithm == "LSB":
                lsb_bits = self.lsb_bits.get()
                mode = "adaptive" if self.lsb_mode.get() == "Thích nghi" else "simple"
                self.extracted_watermark = extract_lsb(self.watermarked_img, key, lsb_bits, mode)
            elif algorithm == "DCT":
                # Lấy kích thước dự kiến của thủy vân
                wm_height = max(self.watermarked_img.shape[0] // 32, 16)
                wm_width = max(self.watermarked_img.shape[1] // 32, 16)
                watermark_shape = (wm_height, wm_width)
                
                # Lấy tham số DCT
                alpha = self.dct_factor.get() / 100.0  # Chuyển đổi thành phạm vi 0.05-1.0
                
                # Gọi hàm với tham số phù hợp
                self.extracted_watermark = dct_watermark_extract(
                    self.watermarked_img,
                    watermark_shape,
                    alpha=alpha,
                    key=key,
                    quantization=True,
                    block_selection='variance'
                )
            elif algorithm == "DWT":
                level = self.dwt_level.get()
                wavelet = self.dwt_wavelet.get()
                # Trích xuất thủy vân với các tham số chính xác
                
                # Lấy kích thước dự kiến của thủy vân
                h, w = self.watermarked_img.shape[:2] if len(self.watermarked_img.shape) == 3 else self.watermarked_img.shape
                wm_height = h // (2**level)
                wm_width = w // (2**level)
                watermark_shape = (wm_height, wm_width)
                
                self.extracted_watermark = extract_dwt(
                    self.watermarked_img,
                    original_watermark_shape=watermark_shape,
                    key=key,
                    wavelet=wavelet,
                    level=level
                )
            elif algorithm == "Spread Spectrum":
                length = self.sw_length.get()
                self.extracted_watermark = extract_sw(
                    self.watermarked_img,
                    key=key,
                    chip_rate=length
                )
            
            self.progress_var.set(70)
            self.root.update_idletasks()
            
            # Display extracted watermark
            self.display_image(self.extracted_watermark, self.extracted_canvas)
            
            self.progress_var.set(100)
            self.status_var.set(f"Trích xuất thủy vân thành công với thuật toán {algorithm}")
            
            # Switch to the Images tab
            self.tabs.select(0)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể trích xuất thủy vân: {str(e)}")
            self.status_var.set("Lỗi khi trích xuất thủy vân")
        finally:
            self.progress_var.set(0)
    
    def test_robustness(self):
        if self.cover_img is None or self.watermark_img is None:
            messagebox.showerror("Lỗi", "Vui lòng tải cả ảnh gốc và ảnh thủy vân trước")
            return
        
        try:
            self.status_var.set(f"Đang kiểm tra độ bền vững với thuật toán {self.algorithm.get()}...")
            self.progress_var.set(10)
            self.root.update_idletasks()
            
            # Get parameters
            key = self.key.get()
            cover_name = os.path.splitext(os.path.basename(self.cover_image_path.get()))[0]
            
            self.progress_var.set(30)
            self.root.update_idletasks()
            
            # Kiểm tra theo thuật toán đã chọn
            algorithm = self.algorithm.get()
            
            if algorithm == "Wu-Lee":
                block_size = self.block_size.get()
                alpha = self.alpha.get()
                results = test_wu_lee(self.cover_img, self.watermark_img, key, cover_name, block_size, alpha)
            elif algorithm == "LSB":
                lsb_bits = self.lsb_bits.get()
                mode = "adaptive" if self.lsb_mode.get() == "Thích nghi" else "simple"
                results = test_lsb(self.cover_img, self.watermark_img, key, lsb_bits, mode)
            elif algorithm == "DCT":
                # Đảm bảo thủy vân được nhị phân hóa đúng cách
                if len(self.watermark_img.shape) > 2:
                    watermark_gray = cv2.cvtColor(self.watermark_img, cv2.COLOR_BGR2GRAY)
                else:
                    watermark_gray = self.watermark_img.copy()
                    
                _, watermark_bin = cv2.threshold(watermark_gray, 128, 255, cv2.THRESH_BINARY)
                
                # Lấy tham số DCT
                alpha = self.dct_factor.get() / 100.0  # Chuyển đổi thành phạm vi 0.05-1.0
                
                # Gọi hàm kiểm tra độ bền vững
                test_results = test_dct(
                    self.cover_img, 
                    watermark_bin, 
                    alpha=alpha, 
                    key=key,
                    quantization=True,
                    block_selection='variance'
                )
                
                # Chuyển đổi kết quả sang định dạng chung để hiển thị
                results = {}
                for attack, data in test_results.items():
                    results[attack] = data['ber']
                
            elif algorithm == "DWT":
                level = self.dwt_level.get()
                wavelet = self.dwt_wavelet.get()
                results = test_dwt(
                    self.cover_img, 
                    self.watermark_img, 
                    alpha=0.1,
                    wavelet=wavelet, 
                    level=level, 
                    key=key
                )
            elif algorithm == "Spread Spectrum":
                gain = self.sw_gain.get()
                length = self.sw_length.get()
                results = test_sw(
                    self.cover_img, 
                    self.watermark_img, 
                    alpha=gain,
                    key=key,
                    chip_rate=length
                )
            
            self.progress_var.set(70)
            self.root.update_idletasks()
            
            # Display results as a bar chart
            self.display_robustness_results(results)
            
            self.progress_var.set(100)
            self.status_var.set(f"Kiểm tra độ bền vững hoàn tất với thuật toán {algorithm}")
            
            # Switch to the Metrics tab
            self.tabs.select(1)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể kiểm tra độ bền vững: {str(e)}")
            self.status_var.set("Lỗi khi kiểm tra độ bền vững")
        finally:
            self.progress_var.set(0)
    
    def display_robustness_results(self, results):
        # Clear previous metrics
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()
        
        # Create a figure for the bar chart
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Translate attack names to Vietnamese
        attack_translations = {
            "Ảnh gốc": "Ảnh gốc",
            "Nhiễu Gaussian": "Nhiễu Gaussian",
            "Nén JPEG": "Nén JPEG",
            "Lọc trung vị": "Lọc trung vị",
            "Cắt ảnh": "Cắt ảnh",
            "Xoay ảnh": "Xoay ảnh"
        }
        
        # Create bar chart
        attacks = [attack_translations.get(k, k) for k in results.keys()]
        ber_values = list(results.values())
        
        bars = ax.bar(attacks, ber_values, color='skyblue')
        
        # Add labels and title
        ax.set_ylabel('Tỉ lệ lỗi bit (BER)')
        ax.set_title(f'Độ bền vững của thủy vân {self.algorithm.get()} đối với các tấn công khác nhau')
        ax.set_ylim(0, max(ber_values) * 1.2)  # Set y-axis limit
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add text labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        
        # Create a canvas to display the figure
        canvas = FigureCanvasTkAgg(fig, master=self.metrics_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def save_results(self):
        if self.watermarked_img is None:
            messagebox.showerror("Lỗi", "Vui lòng nhúng thủy vân trước")
            return
        
        try:
            # Tạo thư mục tương ứng nếu chưa có
            algorithm_folder = {
                "Wu-Lee": "WU_LEE",
                "LSB": "LSB",
                "DCT": "DCT",
                "DWT": "DWT",
                "Spread Spectrum": "SW"
            }
            
            folder = algorithm_folder.get(self.algorithm.get())
            os.makedirs(folder, exist_ok=True)
            
            # Save watermarked image
            watermarked_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("Tệp PNG", "*.png"), ("Tất cả tệp", "*.*")],
                initialdir=folder,
                initialfile=f"{self.algorithm.get()}_anh_da_thuy_van.png",
                title="Lưu ảnh đã nhúng thủy vân"
            )
            
            if watermarked_path:
                cv2.imwrite(watermarked_path, self.watermarked_img)
                self.status_var.set(f"Ảnh đã nhúng thủy vân được lưu tại {watermarked_path}")
            
            # Save extracted watermark if available
            if self.extracted_watermark is not None:
                extracted_path = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("Tệp PNG", "*.png"), ("Tất cả tệp", "*.*")],
                    initialdir=folder,
                    initialfile=f"{self.algorithm.get()}_thuy_van_trich_xuat.png",
                    title="Lưu thủy vân đã trích xuất"
                )
                
                if extracted_path:
                    cv2.imwrite(extracted_path, self.extracted_watermark)
                    self.status_var.set(f"Thủy vân đã trích xuất được lưu tại {extracted_path}")
                
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu kết quả: {str(e)}")
            self.status_var.set("Lỗi khi lưu kết quả")

if __name__ == "__main__":
    root = tk.Tk()
    app = MainWatermarkingApp(root)
    root.mainloop() 