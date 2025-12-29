"""
根据两端截面判断圆柱体的首尾
假设：存在明显条纹的为头部，无任何条纹的为尾部
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Dict, List


class HeadTailDetector:
    """圆柱体首尾检测器"""
    
    def __init__(self, reverse_logic: bool = False):
        """
        初始化检测器
        
        Args:
            reverse_logic: 是否反转判断逻辑
                          False: 高分(多纹理) = 头部, 低分(光滑) = 尾部
                          True:  低分(光滑) = 头部, 高分(多纹理) = 尾部  
        """
        self.min_edge_count = 50  # 最小边缘点数阈值
        self.reverse_logic = reverse_logic
    
    def detect_cylinder_region(self, gray: np.ndarray) -> Tuple[int, int, int, int]:
        """
        检测圆柱体的大致位置
        返回: (x, y, w, h) 边界框
        """
        # 使用Otsu二值化找到物体
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # 如果没找到，返回中央80%区域
            h, w = gray.shape
            margin_w = int(w * 0.1)
            margin_h = int(h * 0.1)
            return margin_w, margin_h, w - 2*margin_w, h - 2*margin_h
        
        # 找到最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        
        return x, y, w, h
    
    def extract_end_regions(self, image: np.ndarray, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取圆柱体两端的截面区域
        基于图像中圆柱体的实际位置
        """
        h, w = image.shape[:2]
        
        # 检测圆柱体位置
        cx, cy, cw, ch = self.detect_cylinder_region(gray)
        
        # 判断圆柱体在图像中的朝向
        # 如果圆柱体宽度 > 高度，说明是水平放置
        if cw > ch * 1.2:
            # 水平放置的圆柱体
            # 提取圆柱体内部左右两侧的区域
            region_width = max(int(cw * 0.3), 50)  # 提取30%宽度区域
            
            # 左侧截面区域
            left_x = cx
            left_region = image[cy:cy+ch, left_x:left_x+region_width]
            left_gray = gray[cy:cy+ch, left_x:left_x+region_width]
            
            # 右侧截面区域  
            right_x = cx + cw - region_width
            right_region = image[cy:cy+ch, right_x:cx+cw]
            right_gray = gray[cy:cy+ch, right_x:cx+cw]
            
            return (left_region, left_gray), (right_region, right_gray)
        else:
            # 垂直放置的圆柱体
            # 提取圆柱体内部上下两侧的区域
            region_height = max(int(ch * 0.3), 50)  # 提取30%高度区域
            
            # 上侧截面区域
            top_y = cy
            top_region = image[top_y:top_y+region_height, cx:cx+cw]
            top_gray = gray[top_y:top_y+region_height, cx:cx+cw]
            
            # 下侧截面区域
            bottom_y = cy + ch - region_height
            bottom_region = image[bottom_y:cy+ch, cx:cx+cw]
            bottom_gray = gray[bottom_y:cy+ch, cx:cx+cw]
            
            return (top_region, top_gray), (bottom_region, bottom_gray)
    
    def detect_patterns(self, gray: np.ndarray) -> Dict:
        """
        检测图像中的条纹/纹理特征
        返回：边缘强度、方向性、纹理复杂度等指标
        """
        # 1. 边缘检测
        edges = cv2.Canny(gray, 30, 100)
        edge_count = np.sum(edges > 0)
        edge_density = edge_count / (gray.shape[0] * gray.shape[1])
        
        # 2. 梯度分析
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        avg_gradient = np.mean(gradient_magnitude)
        
        # 3. 标准差（纹理复杂度）
        std_dev = np.std(gray)
        
        # 4. 频域分析（检测周期性条纹）
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)
        # 计算高频成分的能量
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        # 去除中心低频部分
        mask = np.ones((h, w), dtype=bool)
        mask[center_h-10:center_h+10, center_w-10:center_w+10] = False
        high_freq_energy = np.mean(magnitude_spectrum[mask])
        
        # 5. LBP（局部二值模式）- 纹理特征
        # 简化版：计算像素变化频率
        diff_x = np.abs(np.diff(gray.astype(np.float32), axis=1))
        diff_y = np.abs(np.diff(gray.astype(np.float32), axis=0))
        texture_variation = np.mean(diff_x) + np.mean(diff_y)
        
        return {
            'edge_count': edge_count,
            'edge_density': edge_density,
            'avg_gradient': avg_gradient,
            'std_dev': std_dev,
            'high_freq_energy': high_freq_energy,
            'texture_variation': texture_variation
        }
    
    def calculate_pattern_score(self, features: Dict) -> float:
        """
        计算条纹/纹理得分
        得分越高，表示条纹越明显
        """
        # 各项特征的权重
        score = 0.0
        
        # 边缘密度（条纹会有更多边缘）
        score += features['edge_density'] * 1000
        
        # 梯度强度（条纹有明显的强度变化）
        score += features['avg_gradient'] * 0.5
        
        # 标准差（纹理复杂度）
        score += features['std_dev'] * 0.3
        
        # 高频能量（周期性条纹）
        score += features['high_freq_energy'] * 0.01
        
        # 纹理变化
        score += features['texture_variation'] * 0.2
        
        return score
    
    def detect_ring_stripes(self, image: np.ndarray, gray: np.ndarray, cx: int, cy: int, radius: int) -> Dict:
        """
        检测圆环上的条纹（用于端面视图）
        
        Args:
            image: 原始BGR图像
            gray: 灰度图像
            cx, cy: 圆心坐标
            radius: 圆半径
            
        Returns:
            包含条纹信息的字典
        """
        # 创建环形掩码（只包含纸筒壁外表面区域，排除内部）
        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 外圆半径（稍微扩大）
        outer_radius = int(radius * 1.02)
        # 内圆半径（增大以只保留纸筒壁外表面，排除内部空洞和阴影）
        inner_radius = int(radius * 0.85)
        
        # 绘制环形掩码
        cv2.circle(mask, (cx, cy), outer_radius, 255, -1)
        cv2.circle(mask, (cx, cy), inner_radius, 0, -1)
        
        # 提取环形区域
        ring_region = cv2.bitwise_and(image, image, mask=mask)
        
        # 转换到HSV色彩空间检测彩色条纹
        hsv = cv2.cvtColor(ring_region, cv2.COLOR_BGR2HSV)
        
        # 定义条纹颜色范围（检测彩色和黑色条纹）
        stripe_colors = {
            'yellow': {'lower': np.array([20, 80, 80]), 'upper': np.array([35, 255, 255])},
            'blue': {'lower': np.array([100, 80, 80]), 'upper': np.array([130, 255, 255])},
            'red': None,  # 红色需要特殊处理
            'green': {'lower': np.array([40, 80, 80]), 'upper': np.array([80, 255, 255])},
            'black': {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 100, 50])}  # 深黑色条纹
        }
        
        # 计算环形区域的有效像素数
        ring_pixels = np.sum(mask > 0)
        
        # 检测各种颜色
        color_percentages = {}
        for color_name, color_range in stripe_colors.items():
            if color_name == 'red':
                # 红色跨越HSV色轮
                mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
                mask2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
                color_mask = cv2.bitwise_or(mask1, mask2)
            else:
                color_mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            
            # 只在环形区域内统计
            color_mask = cv2.bitwise_and(color_mask, mask)
            color_pixels = np.sum(color_mask > 0)
            percentage = (color_pixels / ring_pixels) * 100 if ring_pixels > 0 else 0
            
            # 彩色条纹阈值5%，黑色条纹阈值10%（在外表面检测更严格）
            threshold = 10 if color_name == 'black' else 5
            if percentage > threshold:
                color_percentages[color_name] = percentage
        
        # 判断是否有条纹
        has_stripes = len(color_percentages) > 0
        
        return {
            'has_stripes': has_stripes,
            'colors': color_percentages,
            'ring_mask': mask
        }
    
    def determine_head_tail(self, image: np.ndarray) -> Dict:
        """
        判断图像中圆柱体截面是HEAD还是TAIL
        只分析一个截面：有明显条纹=HEAD，无条纹=TAIL
        """
        # 预处理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 首先检查是否为端面视图（圆形）
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=30,
            maxRadius=min(gray.shape[0], gray.shape[1]) // 2
        )
        
        # 如果检测到圆形，分析圆环上是否有条纹
        if circles is not None and len(circles[0]) > 0:
            # 找到最大的圆
            max_circle = max(circles[0], key=lambda c: c[2])
            cx, cy, radius = int(max_circle[0]), int(max_circle[1]), int(max_circle[2])
            
            # 检查圆的大小是否合理
            min_image_dim = min(gray.shape[0], gray.shape[1])
            if radius > 50 and radius < min_image_dim * 0.4:
                # 检测圆环上的条纹
                ring_info = self.detect_ring_stripes(image, gray, cx, cy, radius)
                
                if ring_info['has_stripes']:
                    # 圆环上有彩色条纹 → HEAD
                    return {
                        'type': "HEAD",
                        'pattern_score': 100.0,
                        'confidence': 0.9,
                        'threshold': 50.0,
                        'features': {},
                        'cylinder_region': (0, 0, gray.shape[1], gray.shape[0]),
                        'is_end_view': True,
                        'circle_info': (cx, cy, radius),
                        'ring_colors': ring_info['colors']
                    }
                else:
                    # 圆环上无条纹 → TAIL
                    return {
                        'type': "TAIL",
                        'pattern_score': 0.0,
                        'confidence': 0.9,
                        'threshold': 50.0,
                        'features': {},
                        'cylinder_region': (0, 0, gray.shape[1], gray.shape[0]),
                        'is_end_view': True,
                        'circle_info': (cx, cy, radius),
                        'ring_colors': {}
                    }
        
        # 如果不是圆形，按照原来的侧面视图逻辑处理
        # 检测圆柱体位置
        cx, cy, cw, ch = self.detect_cylinder_region(gray)
        
        # 提取圆柱体截面区域（整个圆柱体内部区域）
        cylinder_gray = gray[cy:cy+ch, cx:cx+cw]
        
        # 检测条纹特征
        features = self.detect_patterns(cylinder_gray)
        
        # 计算条纹得分
        pattern_score = self.calculate_pattern_score(features)
        
        # 判断是HEAD还是TAIL
        # 设定阈值：得分超过50认为有明显条纹
        threshold = 50.0
        
        if pattern_score > threshold:
            # 得分高 = 有条纹 = HEAD
            result_type = "HEAD"
            confidence = min((pattern_score - threshold) / threshold, 1.0)
        else:
            # 得分低 = 无条纹 = TAIL
            result_type = "TAIL"
            confidence = min((threshold - pattern_score) / threshold, 1.0)
        
        return {
            'type': result_type,
            'pattern_score': pattern_score,
            'confidence': confidence,
            'threshold': threshold,
            'features': features,
            'cylinder_region': (cx, cy, cw, ch),
            'is_end_view': False
        }
    
    def visualize(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """
        可视化首尾检测结果 - 只显示HEAD或TAIL标签
        """
        vis_image = image.copy()
        h, w = vis_image.shape[:2]
        
        result_type = result['type']
        confidence = result['confidence']
        pattern_score = result['pattern_score']
        
        # 字体设置 - 更大更醒目
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3.0
        font_thickness = 8
        
        # 显示HEAD或TAIL标签在右上角
        text = result_type
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # 根据类型选择颜色
        if result_type == "HEAD":
            bg_color = (0, 0, 255)  # 红色
        else:
            bg_color = (0, 255, 0)  # 绿色
        
        # 显示在右上角
        cv2.rectangle(vis_image, (w-text_w-30, 10), (w-10, 30 + text_h), bg_color, -1)
        cv2.putText(vis_image, text, (w-text_w-20, 25 + text_h), 
                   font, font_scale, (255, 255, 255), font_thickness)
        
        # 添加置信度信息
        info_texts = [
            f"Type: {result['type']}",
            f"Pattern Score: {result['pattern_score']:.1f}",
            f"Threshold: {result['threshold']:.1f}",
            f"Confidence: {result['confidence']:.2%}"
        ]
        
        text_y = h - 150
        for text in info_texts:
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(vis_image, (5, text_y - text_height - 5), 
                         (15 + text_width, text_y + baseline), (0, 0, 0), -1)
            cv2.putText(vis_image, text, (10, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            text_y += 35
        
        return vis_image


def process_head_tail_batch(input_dir: str = "data/input_is_head", 
                            output_dir: str = "data/output/head",
                            reverse_logic: bool = False):
    """
    批量处理首尾检测
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录  
        reverse_logic: 是否反转判断逻辑（False=有纹理为头部，默认）
    """
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # 递归查找所有图像文件（包括子目录）
    image_files = []
    input_path = Path(input_dir)
    
    for ext in image_extensions:
        image_files.extend(input_path.rglob(f"*{ext}"))
        image_files.extend(input_path.rglob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ 在目录 {input_dir} 中未找到图像文件")
        return
    
    print(f"📁 找到 {len(image_files)} 张图像")
    print("=" * 70)
    
    # 创建检测器（使用反转逻辑）
    detector = HeadTailDetector(reverse_logic=reverse_logic)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计信息
    results = []
    
    # 处理每张图像
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] 处理: {image_path.name}")
        
        # 读取图像（支持中文路径）
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), 
                            cv2.IMREAD_COLOR)
        
        if image is None:
            print(f"   ❌ 无法读取图像")
            continue
        
        # 检测首尾
        result = detector.determine_head_tail(image)
        
        # 可视化
        visualized = detector.visualize(image, result)
        
        # 保存结果
        output_path = os.path.join(output_dir, 
                                  f"{image_path.stem}_head_tail.jpg")
        is_success, buffer = cv2.imencode('.jpg', visualized)
        if is_success:
            buffer.tofile(output_path)
        
        # 打印结果
        print(f"   ✅ 类型: {result['type']}")
        print(f"   📊 条纹得分: {result['pattern_score']:.1f}")
        print(f"   📏 阈值: {result['threshold']:.1f}")
        print(f"   📈 置信度: {result['confidence']:.2%}")
        
        # 保存到结果列表
        results.append({
            'file': image_path.name,
            'subfolder': image_path.parent.name,
            'type': result['type'],
            'pattern_score': result['pattern_score'],
            'threshold': result['threshold'],
            'confidence': result['confidence']
        })
    
    print("\n" + "=" * 70)
    print(f"📊 处理完成! 共处理 {len(results)} 张图像")
    print(f"💾 结果保存在: {output_dir}")
    
    # 保存CSV报告
    if results:
        csv_path = os.path.join(output_dir, "head_tail_report.csv")
        with open(csv_path, 'w', encoding='utf-8-sig') as f:
            f.write("文件名,子目录,类型,条纹得分,阈值,置信度\n")
            for r in results:
                f.write(f"{r['file']},{r['subfolder']},{r['type']},"
                       f"{r['pattern_score']:.2f},{r['threshold']:.2f},"
                       f"{r['confidence']:.4f}\n")
        print(f"📄 报告已保存: {csv_path}")


def process_single_head_tail(image_path: str, output_dir: str = "data/output/head",
                            reverse_logic: bool = False):
    """处理单张图像的首尾检测
    
    Args:
        image_path: 图像路径
        output_dir: 输出目录
        reverse_logic: 是否反转判断逻辑（False=有纹理为头部，默认）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取图像（支持中文路径）
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 
                        cv2.IMREAD_COLOR)
    
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return None
    
    print(f"📸 处理图像: {os.path.basename(image_path)}")
    print(f"   图像尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # 创建检测器
    detector = HeadTailDetector(reverse_logic=reverse_logic)
    
    # 检测首尾
    result = detector.determine_head_tail(image)
    
    # 可视化
    visualized = detector.visualize(image, result)
    
    # 打印检测信息
    print(f"   ✅ 检测成功!")
    print(f"   🔍 类型: {result['type']}")
    print(f"   📊 条纹得分: {result['pattern_score']:.1f}")
    print(f"   📏 阈值: {result['threshold']:.1f}")
    print(f"   📈 置信度: {result['confidence']:.2%}")
    
    # 保存结果
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{filename}_head_tail.jpg")
    
    is_success, buffer = cv2.imencode('.jpg', visualized)
    if is_success:
        buffer.tofile(output_path)
        print(f"   💾 结果已保存: {output_path}")
    
    # 显示结果
    display_height = 800
    aspect_ratio = image.shape[1] / image.shape[0]
    display_width = int(display_height * aspect_ratio)
    
    display_image = cv2.resize(visualized, (display_width, display_height))
    cv2.imshow("Head-Tail Detection (Press any key to close)", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("=" * 70)
        print("🔍 圆柱体首尾检测工具")
        print("=" * 70)
        print("\n功能说明:")
        print("  根据两端截面的条纹特征判断圆柱体的首尾")
        print("  - 存在明显条纹的为头部")
        print("  - 无任何条纹的为尾部")
        print("\n使用方法:")
        print("  python head_tail_detector.py <image_path>      # 处理单张图像")
        print("  python head_tail_detector.py --batch           # 批量处理")
        print("\n示例:")
        print("  python head_tail_detector.py \"data/input_is_head/1/h (1).jpg\"")
        print("  python head_tail_detector.py --batch")
    elif sys.argv[1] == "--batch" or sys.argv[1] == "-b":
        process_head_tail_batch()
    else:
        process_single_head_tail(sys.argv[1])
