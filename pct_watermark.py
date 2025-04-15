import cv2
import numpy as np
import pywt
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

class PCTWatermarker:
    """
    PCT (Perceptible Content Transform) Watermarking Algorithm
    
    This class implements a robust watermarking technique using wavelet transforms
    to embed and extract watermarks from images.
    """
    
    def __init__(self, alpha=0.1, wavelet='haar', level=2):
        """
        Initialize the watermarker with parameters
        
        Args:
            alpha: Strength of the watermark (0.05-0.5)
            wavelet: Wavelet family to use (haar, db1, db2, etc.)
            level: Decomposition level
        """
        self.alpha = alpha
        self.wavelet = wavelet
        self.level = level
    
    def embed_watermark(self, image_path, watermark_path, output_path=None):
        """
        Embed a watermark image into a host image
        
        Args:
            image_path: Path to the host image
            watermark_path: Path to the watermark image
            output_path: Path to save the watermarked image (optional)
            
        Returns:
            Watermarked image as numpy array
        """
        # Read images
        host_img = cv2.imread(image_path)
        watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if images were loaded successfully
        if host_img is None or watermark is None:
            raise ValueError("Could not load host or watermark image")
        
        # Convert host image to YCrCb color space
        host_ycrcb = cv2.cvtColor(host_img, cv2.COLOR_BGR2YCrCb)
        y_channel = host_ycrcb[:,:,0].astype(float)
        
        # Resize watermark to match host dimensions
        watermark = cv2.resize(watermark, (y_channel.shape[1], y_channel.shape[0]))
        watermark = watermark / 255.0  # Normalize to [0,1]
        
        # Apply wavelet transform to Y channel
        coeffs = pywt.wavedec2(y_channel, self.wavelet, level=self.level)
        
        # Embed watermark in the detail coefficients of the chosen level
        # We embed in LH (horizontal), HL (vertical), and HH (diagonal) coefficients
        LH, HL, HH = coeffs[1]
        
        # Embedding formula: LH' = LH + alpha * watermark
        LH_mod = LH + self.alpha * watermark
        HL_mod = HL + self.alpha * watermark
        HH_mod = HH + self.alpha * watermark
        
        # Replace the original coefficients with modified ones
        coeffs[1] = (LH_mod, HL_mod, HH_mod)
        
        # Apply inverse wavelet transform
        watermarked_y = pywt.waverec2(coeffs, self.wavelet)
        
        # Ensure values are within valid range
        watermarked_y = np.clip(watermarked_y, 0, 255)
        
        # Replace Y channel with watermarked version
        host_ycrcb[:,:,0] = watermarked_y.astype(np.uint8)
        
        # Convert back to BGR
        watermarked_img = cv2.cvtColor(host_ycrcb, cv2.COLOR_YCrCb2BGR)
        
        # Save watermarked image if output path is provided
        if output_path:
            cv2.imwrite(output_path, watermarked_img)
            print(f"Watermarked image saved to {output_path}")
        
        return watermarked_img
    
    def extract_watermark(self, original_path, watermarked_path, output_path=None):
        """
        Extract watermark from a watermarked image
        
        Args:
            original_path: Path to the original host image
            watermarked_path: Path to the watermarked image
            output_path: Path to save the extracted watermark (optional)
            
        Returns:
            Extracted watermark as numpy array
        """
        # Read images
        original_img = cv2.imread(original_path)
        watermarked_img = cv2.imread(watermarked_path)
        
        # Check if images were loaded successfully
        if original_img is None or watermarked_img is None:
            raise ValueError("Could not load original or watermarked image")
        
        # Convert to YCrCb
        original_ycrcb = cv2.cvtColor(original_img, cv2.COLOR_BGR2YCrCb)
        watermarked_ycrcb = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2YCrCb)
        
        # Get Y channels
        y_original = original_ycrcb[:,:,0].astype(float)
        y_watermarked = watermarked_ycrcb[:,:,0].astype(float)
        
        # Apply wavelet transform to both Y channels
        coeffs_original = pywt.wavedec2(y_original, self.wavelet, level=self.level)
        coeffs_watermarked = pywt.wavedec2(y_watermarked, self.wavelet, level=self.level)
        
        # Extract watermark from the detail coefficients
        LH_original, HL_original, HH_original = coeffs_original[1]
        LH_watermarked, HL_watermarked, HH_watermarked = coeffs_watermarked[1]
        
        # Extraction formula: watermark = (LH_watermarked - LH_original) / alpha
        watermark_LH = (LH_watermarked - LH_original) / self.alpha
        watermark_HL = (HL_watermarked - HL_original) / self.alpha
        watermark_HH = (HH_watermarked - HH_original) / self.alpha
        
        # Average the three extracted watermarks for better results
        extracted_watermark = (watermark_LH + watermark_HL + watermark_HH) / 3
        
        # Normalize to [0,255] for visualization
        extracted_watermark = np.clip(extracted_watermark, 0, 1)
        extracted_watermark = (extracted_watermark * 255).astype(np.uint8)
        
        # Save extracted watermark if output path is provided
        if output_path:
            cv2.imwrite(output_path, extracted_watermark)
            print(f"Extracted watermark saved to {output_path}")
        
        return extracted_watermark
    
    def evaluate_performance(self, original_path, watermarked_path, watermark_path):
        """
        Evaluate the performance of the watermarking algorithm
        
        Args:
            original_path: Path to the original host image
            watermarked_path: Path to the watermarked image
            watermark_path: Path to the original watermark
            
        Returns:
            Dictionary containing PSNR and SSIM metrics
        """
        # Read images
        original_img = cv2.imread(original_path)
        watermarked_img = cv2.imread(watermarked_path)
        watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        
        # Calculate PSNR between original and watermarked
        mse = np.mean((original_img - watermarked_img) ** 2)
        if mse == 0:
            psnr = 100
        else:
            psnr = 10 * np.log10(255.0**2 / mse)
        
        # Extract watermark
        extracted_watermark = self.extract_watermark(original_path, watermarked_path)
        
        # Resize original watermark to match extracted one if needed
        watermark = cv2.resize(watermark, (extracted_watermark.shape[1], extracted_watermark.shape[0]))
        
        # Calculate Normalized Correlation (NC) between original and extracted watermark
        watermark_norm = watermark / 255.0
        extracted_norm = extracted_watermark / 255.0
        
        nc = np.sum(watermark_norm * extracted_norm) / np.sqrt(np.sum(watermark_norm**2) * np.sum(extracted_norm**2))
        
        return {
            "PSNR": psnr,
            "NC": nc
        }
    
    def display_results(self, original_path, watermarked_path, watermark_path, extracted_watermark_path=None):
        """
        Display original, watermarked images and watermarks
        
        Args:
            original_path: Path to the original host image
            watermarked_path: Path to the watermarked image
            watermark_path: Path to the original watermark
            extracted_watermark_path: Path to the extracted watermark (optional)
        """
        # Read images
        original_img = cv2.imread(original_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        watermarked_img = cv2.imread(watermarked_path)
        watermarked_img = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2RGB)
        
        watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        
        if extracted_watermark_path:
            extracted_watermark = cv2.imread(extracted_watermark_path, cv2.IMREAD_GRAYSCALE)
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        else:
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            # Extract watermark
            extracted_watermark = self.extract_watermark(original_path, watermarked_path)
        
        axs[0, 0].imshow(original_img)
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')
        
        axs[0, 1].imshow(watermarked_img)
        axs[0, 1].set_title('Watermarked Image')
        axs[0, 1].axis('off')
        
        axs[1, 0].imshow(watermark, cmap='gray')
        axs[1, 0].set_title('Original Watermark')
        axs[1, 0].axis('off')
        
        axs[1, 1].imshow(extracted_watermark, cmap='gray')
        axs[1, 1].set_title('Extracted Watermark')
        axs[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print performance metrics
        metrics = self.evaluate_performance(original_path, watermarked_path, watermark_path)
        print(f"PSNR: {metrics['PSNR']:.2f} dB")
        print(f"NC: {metrics['NC']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='PCT Watermarking Tool')
    parser.add_argument('--mode', choices=['embed', 'extract', 'evaluate'], required=True,
                        help='Operation mode: embed, extract, or evaluate')
    parser.add_argument('--image', required=True, help='Path to the host image')
    parser.add_argument('--watermark', help='Path to the watermark image (required for embed mode)')
    parser.add_argument('--watermarked', help='Path to the watermarked image (required for extract/evaluate modes)')
    parser.add_argument('--output', help='Path to save the output')
    parser.add_argument('--alpha', type=float, default=0.1, help='Watermark strength (default: 0.1)')
    parser.add_argument('--wavelet', default='haar', help='Wavelet family (default: haar)')
    parser.add_argument('--level', type=int, default=2, help='Decomposition level (default: 2)')
    parser.add_argument('--display', action='store_true', help='Display results')
    
    args = parser.parse_args()
    
    # Initialize watermarker
    watermarker = PCTWatermarker(alpha=args.alpha, wavelet=args.wavelet, level=args.level)
    
    if args.mode == 'embed':
        if not args.watermark:
            parser.error("--watermark is required for embed mode")
        if not args.output:
            args.output = f"watermarked_{Path(args.image).name}"
        
        watermarker.embed_watermark(args.image, args.watermark, args.output)
        
        if args.display:
            watermarker.display_results(args.image, args.output, args.watermark)
    
    elif args.mode == 'extract':
        if not args.watermarked:
            parser.error("--watermarked is required for extract mode")
        if not args.output:
            args.output = f"extracted_watermark_{Path(args.watermarked).name}"
        
        watermarker.extract_watermark(args.image, args.watermarked, args.output)
        
        if args.display and args.watermark:
            watermarker.display_results(args.image, args.watermarked, args.watermark, args.output)
    
    elif args.mode == 'evaluate':
        if not args.watermarked or not args.watermark:
            parser.error("--watermarked and --watermark are required for evaluate mode")
        
        metrics = watermarker.evaluate_performance(args.image, args.watermarked, args.watermark)
        print(f"PSNR: {metrics['PSNR']:.2f} dB")
        print(f"NC: {metrics['NC']:.4f}")
        
        if args.display:
            watermarker.display_results(args.image, args.watermarked, args.watermark)


if __name__ == "__main__":
    main()
