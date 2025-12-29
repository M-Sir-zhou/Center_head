"""
智能圆柱体中心检测器
自动识别视角（端面/侧面）并使用合适的检测方法
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import os


class SmartCylinderDetector:
    """智能圆柱体检测器 - 自动识别视角"""
    
    def __init__(self):
        self.min_circle_radius = 20
        self.max_circle_radius = 1000
        self.min_contour_area = 500
        
        # 定义常见条纹颜色范围（HSV色彩空间）
        self.color_ranges = {
            'black': {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 50])},
            'white': {'lower': np.array([0, 0, 200]), 'upper': np.array([180, 30, 255])},
            'yellow': {'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255])},
            'blue': {'lower': np.array([100, 100, 100]), 'upper': np.array([130, 255, 255])},
            'red': {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
            'green': {'lower': np.array([40, 40, 40]), 'upper': np.array([80, 255, 255])},
            'brown': {'lower': np.array([10, 50, 20]), 'upper': np.array([20, 255, 200])},
            'orange': {'lower': np.array([10, 100, 100]), 'upper': np.array([20, 255, 255])}
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred
    
    def detect_circles(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """检测圆形（端面视图）- 改进版"""
        # 尝试多组参数
        param_sets = [
            # (minDist, param1, param2, minRadius, maxRadius)
            (50, 100, 30, 20, 500),   # 原始参数
            (40, 80, 25, 30, 400),    # 更宽松
            (60, 120, 35, 40, 600),   # 更严格
            (30, 60, 20, 20, 300),    # 最宽松
        ]
        
        all_circles = []
        
        for minDist, param1, param2, minRadius, maxRadius in param_sets:
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=minDist,
                param1=param1,
                param2=param2,
                minRadius=minRadius,
                maxRadius=min(maxRadius, min(gray.shape) // 2)
            )
            
            if circles is not None:
                all_circles.extend(circles[0])
        
        if not all_circles:
            return None
        
        # 去重：合并相近的圆
        unique_circles = []
        all_circles = sorted(all_circles, key=lambda c: c[2], reverse=True)  # 按半径排序
        
        # 图像尺寸
        max_dim = max(gray.shape)
        
        for circle in all_circles:
            cx, cy, r = circle
            
            # 过滤不合理的圆：
            # 1. 半径不能太大（超过图像20%）
            # 2. 圆心不能太靠近边缘
            # 3. 半径不能太小（小于30px）
            margin = max(r * 0.3, 50)  # 至少50px的边距
            if (r > max_dim * 0.20 or  # 半径太大（改为20%）
                cx < margin or cy < margin or  # 太靠边
                cx > gray.shape[1] - margin or cy > gray.shape[0] - margin or
                r < 30):  # 太小
                continue
            
            is_duplicate = False
            
            for existing in unique_circles:
                ex, ey, er = existing
                dist = np.sqrt((cx - ex)**2 + (cy - ey)**2)
                
                # 如果圆心很接近，认为是同一个圆
                if dist < min(r, er) * 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_circles.append(circle)
        
        if not unique_circles:
            return None
        
        # 返回为霍夫圆变换格式
        return np.array([unique_circles], dtype=np.float32)
    
    def detect_rectangle_contours(self, image: np.ndarray, gray: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """检测矩形轮廓（侧面视图）- 改进版，避免检测背景"""
        h, w = gray.shape
        image_area = h * w
        
        # 多种二值化方法
        methods = []
        
        # 方法1: Otsu阈值（正向和反向）
        _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        methods.append(('otsu_inv', binary1))
        
        _, binary1_normal = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append(('otsu_normal', binary1_normal))
        
        # 方法2: 自适应阈值
        binary2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
        methods.append(('adaptive_inv', binary2))
        
        # 方法3: Canny边缘
        edges = cv2.Canny(gray, 30, 100)
        # 膨胀边缘以形成闭合轮廓
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        methods.append(('canny', edges))
        
        best_contour = None
        best_score = 0
        best_method = None
        
        for method_name, binary in methods:
            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            binary_processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            binary_processed = cv2.morphologyEx(binary_processed, cv2.MORPH_OPEN, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(binary_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # 过滤条件
                if area < self.min_contour_area:
                    continue
                
                # 排除占据整个图像的轮廓（可能是背景）
                area_ratio = area / image_area
                if area_ratio > 0.7:  # 如果占据超过70%的图像，肯定是背景
                    continue
                
                # 排除中等大但可能是背景的轮廓
                if area_ratio > 0.25:  # 占25%-70%，需要进一步检查
                    # 如果是大轮廓且贴近边缘，很可能是背景
                    margin = 20
                    if (x < margin or y < margin or 
                        x + w > image.shape[1] - margin or 
                        y + h > image.shape[0] - margin):
                        continue  # 大且贴边，是背景
                
                # 计算边界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 排除贴边的大轮廓（可能是背景/白纸）
                margin = 10
                if (x < margin and y < margin and 
                    x + w > image.shape[1] - margin and 
                    y + h > image.shape[0] - margin):
                    continue  # 四边都贴边，可能是背景
                
                rect_area = w * h
                rectangularity = area / rect_area if rect_area > 0 else 0
                aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
                
                # 计算周长
                perimeter = cv2.arcLength(contour, True)
                
                # 计算紧凑度（越接近1越规则）
                if perimeter > 0:
                    compactness = (4 * np.pi * area) / (perimeter * perimeter)
                else:
                    compactness = 0
                
                # 综合评分改进v2：
                # 重点：正确识别长条形圆柱体（如p3）和正方形圆柱体（如p1）
                
                # 面积分数：优先5%-25%的中等面积
                if 0.05 < area_ratio < 0.20:
                    area_score = 2.0  # 最佳范围
                elif 0.02 < area_ratio < 0.05:
                    area_score = 1.2  # 稍小但可接受
                elif 0.20 < area_ratio < 0.30:
                    area_score = 0.8  # 偏大
                else:
                    area_score = 0.2  # 太小(<2%)或太大(>30%)
                
                # 形状分数：针对不同宽高比
                if 0.8 < aspect_ratio < 1.5:
                    shape_score = 3.0  # 接近正方形，优先级最高（p1类型）
                elif 2.0 < aspect_ratio < 4.0:
                    shape_score = 2.5  # 长条形，优先级很高（p3类型）
                elif 1.5 < aspect_ratio < 2.0:
                    shape_score = 2.0  # 稍长
                elif 4.0 < aspect_ratio < 8.0:
                    shape_score = 1.5  # 很长的条形
                else:
                    shape_score = 0.5  # 太极端
                
                # 矩形度分数：必须大于0.4才考虑
                if rectangularity < 0.4:
                    rect_score = 0.1  # 太不规则
                else:
                    rect_score = rectangularity ** 1.5
                
                # 紧凑度分数
                compact_score = 1.0 + compactness * 0.3
                
                # 综合分数（使用面积开方降低大轮廓优势）
                score = (area ** 0.6) * rect_score * area_score * shape_score * compact_score
                
                if score > best_score:
                    best_score = score
                    best_contour = contour
                    best_method = method_name
        
        info = {
            'method': best_method,
            'score': best_score
        }
        
        return best_contour, info
    
    def calculate_contour_center(self, contour: np.ndarray) -> Tuple[int, int]:
        """计算轮廓的中心点 - 使用边界框中心（更稳定）"""
        # 对于矩形物体，边界框的几何中心比质心更准确
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2
        cy = y + h // 2
        return cx, cy
    
    def detect_orientation(self, contour: np.ndarray) -> str:
        """检测方向"""
        x, y, w, h = cv2.boundingRect(contour)
        if w > h * 1.2:
            return "horizontal"
        elif h > w * 1.2:
            return "vertical"
        else:
            return "square"
    
    def detect_stripe_colors(self, image: np.ndarray, contour: np.ndarray = None, 
                            circle_info: tuple = None) -> Dict:
        """
        检测条纹颜色
        
        Args:
            image: 原始BGR图像
            contour: 轮廓（侧面视图）
            circle_info: (x, y, radius) 圆形信息（端面视图）
        
        Returns:
            包含检测到的颜色信息的字典
        """
        # 获取感兴趣区域
        if contour is not None:
            # 侧面视图：使用轮廓区域
            x, y, w, h = cv2.boundingRect(contour)
            roi = image[y:y+h, x:x+w]
        elif circle_info is not None:
            # 端面视图：使用圆形区域
            cx, cy, radius = circle_info
            x = max(0, int(cx - radius))
            y = max(0, int(cy - radius))
            w = int(radius * 2)
            h = int(radius * 2)
            roi = image[y:y+h, x:x+w]
        else:
            # 使用整个图像
            roi = image
        
        if roi.size == 0:
            return {'colors': [], 'dominant_color': None}
        
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 检测每种颜色
        color_percentages = {}
        total_pixels = roi.shape[0] * roi.shape[1]
        
        for color_name, color_range in self.color_ranges.items():
            # 创建颜色掩码
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            
            # 计算颜色占比
            color_pixels = np.sum(mask > 0)
            percentage = (color_pixels / total_pixels) * 100
            
            if percentage > 5:  # 只记录占比超过5%的颜色
                color_percentages[color_name] = percentage
        
        # 特殊处理红色（跨越HSV色轮）
        if 'red' in self.color_ranges:
            mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
            red_mask = cv2.bitwise_or(mask1, mask2)
            red_pixels = np.sum(red_mask > 0)
            red_percentage = (red_pixels / total_pixels) * 100
            if red_percentage > 5:
                color_percentages['red'] = red_percentage
        
        # 条纹颜色（排除纸筒本色）
        # 彩色条纹：黄色、蓝色、红色、绿色（占比>8%）
        # 黑色条纹：黑色（占比>15%，避免误判阴影）
        color_stripe_colors = ['yellow', 'blue', 'red', 'green']
        black_stripe_colors = ['black']
        
        # 保留彩色条纹和黑色条纹
        stripe_color_percentages = {}
        for color_name, percentage in color_percentages.items():
            if color_name in color_stripe_colors and percentage > 8:
                stripe_color_percentages[color_name] = percentage
            elif color_name in black_stripe_colors and percentage > 15:
                stripe_color_percentages[color_name] = percentage
        
        # 如果没有检测到条纹，返回空结果
        if not stripe_color_percentages:
            return {
                'colors': [],
                'colors_cn': [],
                'dominant_color': None,
                'dominant_color_cn': None,
                'has_stripes': False
            }
        
        # 如果过滤后没有颜色，返回空结果
        if not stripe_color_percentages:
            return {
                'colors': [],
                'colors_cn': [],
                'dominant_color': None,
                'dominant_color_cn': None,
                'has_stripes': False
            }
        
        # 按占比排序
        sorted_colors = sorted(stripe_color_percentages.items(), key=lambda x: x[1], reverse=True)
        
        # 获取主要颜色
        dominant_color = sorted_colors[0][0] if sorted_colors else None
        
        # 中文颜色名称映射
        color_names_cn = {
            'black': '黑色',
            'white': '白色',
            'yellow': '黄色',
            'blue': '蓝色',
            'red': '红色',
            'green': '绿色'
        }
        
        # 转换为中文名称
        colors_cn = [(color_names_cn.get(c, c), p) for c, p in sorted_colors]
        
        return {
            'colors': sorted_colors,
            'colors_cn': colors_cn,
            'dominant_color': dominant_color,
            'dominant_color_cn': color_names_cn.get(dominant_color, dominant_color) if dominant_color else None,
            'has_stripes': True
        }
    
    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        智能处理图像
        自动判断是圆形（端面）还是矩形（侧面）视图
        """
        result_image = image.copy()
        gray = self.preprocess_image(image)
        
        # 尝试圆形检测
        circles = self.detect_circles(gray)
        
        # 尝试矩形轮廓检测
        best_contour, contour_info = self.detect_rectangle_contours(image, gray)
        
        # 决策：使用哪种方法
        circle_detected = circles is not None and len(circles[0]) > 0
        contour_detected = best_contour is not None
        
        if not circle_detected and not contour_detected:
            return result_image, {"error": "未检测到圆柱体"}
        
        # 智能决策
        use_circle = False
        
        if circle_detected and contour_detected:
            # 两种都检测到，需要判断哪个更可靠
            circle = circles[0][0]
            circle_x, circle_y, circle_radius = circle[0], circle[1], circle[2]
            circle_area = np.pi * circle_radius ** 2
            contour_area = cv2.contourArea(best_contour)
            
            # 计算轮廓的宽高比
            x, y, w, h = cv2.boundingRect(best_contour)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            
            # 计算轮廓占图像的比例
            image_area = gray.shape[0] * gray.shape[1]
            contour_ratio = contour_area / image_area
            circle_ratio = circle_area / image_area
            
            min_image_dim = min(gray.shape[0], gray.shape[1])
            
            # 新的决策逻辑：
            # 如果轮廓的宽高比接近1（正方形），且面积合理（3%-15%），
            # 很可能是圆柱体的侧面视图（水平放置）
            # 这种情况下轮廓检测更可靠
            
            # 条件1：轮廓接近正方形且面积合理 -> 优先使用轮廓
            if 0.8 < aspect_ratio < 1.5 and 0.02 < contour_ratio < 0.15:
                use_circle = False
                
            # 条件2：明显的长条形 -> 侧面视图，用轮廓
            elif aspect_ratio > 2.0 and 0.05 < contour_ratio < 0.8:
                use_circle = False
                
            # 条件3：圆形半径太大 -> 可能误检，用轮廓
            elif circle_radius > min_image_dim * 0.2:
                use_circle = False
                
            # 条件4：圆形很小但轮廓很大 -> 用轮廓
            elif circle_radius < 100 and contour_area > circle_area * 1.5:
                use_circle = False
                
            # 条件5：轮廓很小但圆形合理 -> 端面视图
            elif contour_ratio < 0.02 and 100 < circle_radius < min_image_dim * 0.18:
                use_circle = True
                
            # 条件6：圆形合理且轮廓比例不是特别合适 -> 端面视图
            elif 100 < circle_radius < min_image_dim * 0.18 and (contour_ratio > 0.3 or contour_ratio < 0.01):
                use_circle = True
                
            # 默认：使用轮廓（侧面视图更常见）
            else:
                use_circle = False
                
        elif circle_detected:
            # 只检测到圆形，检查是否合理
            circle = circles[0][0]
            circle_radius = circle[2]
            min_image_dim = min(gray.shape[0], gray.shape[1])
            
            # 圆形半径不能太大或太小
            if 40 < circle_radius < min_image_dim * 0.2:
                use_circle = True
            else:
                return result_image, {"error": "未检测到合适的圆柱体"}
        else:
            use_circle = False
        
        # 根据决策返回结果
        if use_circle:
            # 使用圆形检测结果
            circles = np.uint16(np.around(circles))
            # 选择最大的圆
            max_circle = max(circles[0], key=lambda c: c[2])
            
            # 端面视图不检测颜色（内部黑色是空洞不是条纹）
            color_info = {
                'colors': [],
                'colors_cn': [],
                'dominant_color': None,
                'dominant_color_cn': None,
                'has_stripes': False
            }
            
            info = {
                "type": "circle",
                "center": (int(max_circle[0]), int(max_circle[1])),
                "radius": int(max_circle[2]),
                "view": "端面视图 (End View)",
                "num_circles": len(circles[0]),
                "color_info": color_info
            }
        else:
            # 使用轮廓检测结果
            cx, cy = self.calculate_contour_center(best_contour)
            x, y, w, h = cv2.boundingRect(best_contour)
            orientation = self.detect_orientation(best_contour)
            
            # 检查是否为接近正方形的形状（可能是圆形端面）
            # 如果宽高比接近1，跳过颜色检测
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            if aspect_ratio < 1.3:
                # 接近正方形，可能是端面视图，跳过颜色检测
                color_info = {
                    'colors': [],
                    'colors_cn': [],
                    'dominant_color': None,
                    'dominant_color_cn': None,
                    'has_stripes': False
                }
            else:
                # 检测颜色（只针对侧面视图）
                color_info = self.detect_stripe_colors(image, contour=best_contour)
            
            info = {
                "type": "rectangle",
                "center": (cx, cy),
                "contour": best_contour,
                "bounding_box": (x, y, w, h),
                "orientation": orientation,
                "view": f"侧面视图 (Side View - {orientation})",
                "method": contour_info['method'],
                "area": cv2.contourArea(best_contour),
                "color_info": color_info
            }
        
        return result_image, info
    
    def visualize(self, image: np.ndarray, info: Dict) -> np.ndarray:
        """可视化检测结果"""
        result = image.copy()
        
        if "error" in info:
            cv2.putText(result, info["error"], (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return result
        
        cx, cy = info['center']
        
        if info['type'] == 'circle':
            # 圆形检测可视化
            radius = info['radius']
            
            # 绘制圆形轮廓
            cv2.circle(result, (cx, cy), radius, (0, 255, 0), 3)
            
            # 绘制中心点
            cv2.circle(result, (cx, cy), 8, (0, 0, 255), -1)
            cv2.circle(result, (cx, cy), 12, (255, 0, 0), 2)
            
            # 绘制十字线
            cross_length = min(50, radius // 2)
            cv2.line(result, (cx - cross_length, cy), (cx + cross_length, cy), (255, 0, 0), 3)
            cv2.line(result, (cx, cy - cross_length), (cx, cy + cross_length), (255, 0, 0), 3)
            
            # 绘制直径线
            for angle in [0, 45, 90, 135]:
                rad = np.radians(angle)
                x1 = int(cx + radius * np.cos(rad))
                y1 = int(cy + radius * np.sin(rad))
                x2 = int(cx - radius * np.cos(rad))
                y2 = int(cy - radius * np.sin(rad))
                cv2.line(result, (x1, y1), (x2, y2), (255, 255, 0), 1)
            
            # 文本信息
            texts = [
                f"View: {info['view']}",
                f"Center: ({cx}, {cy})",
                f"Radius: {radius} px",
                f"Diameter: {radius * 2} px"
            ]
            
            # 添加颜色信息（只有检测到条纹时才显示）
            if 'color_info' in info and info['color_info'].get('has_stripes', False):
                colors_str = ", ".join([c for c, _ in info['color_info']['colors'][:3]])
                texts.append(f"Colors: {colors_str}")
        
        else:
            # 矩形检测可视化
            x, y, w, h = info['bounding_box']
            contour = info['contour']
            
            # 绘制轮廓
            cv2.drawContours(result, [contour], -1, (0, 255, 0), 3)
            
            # 绘制边界框
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 255), 2)
            
            # 绘制中心点
            cv2.circle(result, (cx, cy), 10, (0, 0, 255), -1)
            cv2.circle(result, (cx, cy), 15, (255, 0, 0), 3)
            
            # 绘制十字线（穿过整个图像）
            cv2.line(result, (0, cy), (result.shape[1], cy), (255, 0, 0), 2)
            cv2.line(result, (cx, 0), (cx, result.shape[0]), (255, 0, 0), 2)
            
            # 绘制局部十字线
            cross_length = 40
            cv2.line(result, (cx - cross_length, cy), (cx + cross_length, cy), (0, 255, 255), 3)
            cv2.line(result, (cx, cy - cross_length), (cx, cy + cross_length), (0, 255, 255), 3)
            
            # 文本信息
            texts = [
                f"View: {info['view']}",
                f"Center: ({cx}, {cy})",
                f"Size: {w}x{h} px",
                f"Area: {int(info['area'])} px^2",
                f"Method: {info['method']}"
            ]
            
            # 添加颜色信息（只有检测到条纹时才显示）
            if 'color_info' in info and info['color_info'].get('has_stripes', False):
                colors_str = ", ".join([c for c, _ in info['color_info']['colors'][:3]])
                texts.append(f"Colors: {colors_str}")
        
        # 绘制文本背景和文字
        text_y = 40
        for text in texts:
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(result, (5, text_y - text_height - 8), 
                         (20 + text_width, text_y + baseline), (0, 0, 0), -1)
            cv2.putText(result, text, (10, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            text_y += 40
        
        return result


def process_single_image(image_path: str, output_dir: str = "data/output/center"):
    """处理单张图像"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取图像（支持中文路径）
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return None
    
    print(f"📸 处理图像: {os.path.basename(image_path)}")
    print(f"   图像尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # 创建检测器
    detector = SmartCylinderDetector()
    
    # 处理图像
    _, info = detector.process_image(image)
    
    # 可视化
    visualized = detector.visualize(image, info)
    
    # 打印检测信息
    if "error" in info:
        print(f"   ❌ {info['error']}")
        return None
    else:
        cx, cy = info['center']
        view_type = info['type']
        view_name = info['view']
        
        print(f"   ✅ 检测成功!")
        print(f"   🔍 视图类型: {view_name}")
        print(f"   📍 中心坐标: ({cx}, {cy})")
        
        if view_type == 'circle':
            print(f"   📏 半径: {info['radius']} 像素")
            print(f"   📐 直径: {info['radius'] * 2} 像素")
        else:
            x, y, w, h = info['bounding_box']
            print(f"   📏 边界框: ({x}, {y}) - 尺寸: {w}x{h}")
            print(f"   📐 方向: {info['orientation']}")
            print(f"   📊 面积: {int(info['area'])} 像素²")
    
    # 保存结果
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{filename}_smart_result.jpg")
    
    is_success, buffer = cv2.imencode('.jpg', visualized)
    if is_success:
        buffer.tofile(output_path)
        print(f"   💾 结果已保存: {output_path}")
    
    # 显示结果
    display_height = 800
    aspect_ratio = image.shape[1] / image.shape[0]
    display_width = int(display_height * aspect_ratio)
    
    display_image = cv2.resize(visualized, (display_width, display_height))
    cv2.imshow("Smart Detection Result (Press any key to close)", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return info


def batch_process(input_dir: str = "data/input", output_dir: str = "data/output/center"):
    """批量处理"""
    from pathlib import Path
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ 在目录 {input_dir} 中未找到图像文件")
        return
    
    print(f"📁 找到 {len(image_files)} 张图像")
    print("=" * 70)
    
    detector = SmartCylinderDetector()
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    results = []
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] 处理: {image_path.name}")
        
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            print(f"   ❌ 无法读取图像")
            fail_count += 1
            continue
        
        _, info = detector.process_image(image)
        visualized = detector.visualize(image, info)
        
        # 保存结果
        output_path = os.path.join(output_dir, f"{image_path.stem}_smart_result.jpg")
        is_success, buffer = cv2.imencode('.jpg', visualized)
        if is_success:
            buffer.tofile(output_path)
        
        if "error" in info:
            print(f"   ❌ {info['error']}")
            fail_count += 1
        else:
            cx, cy = info['center']
            view_type = info['type']
            view_name = info['view']
            
            print(f"   ✅ {view_name}")
            print(f"      中心: ({cx}, {cy})")
            
            success_count += 1
            
            result_data = {
                'file': image_path.name,
                'view': view_name,
                'center_x': cx,
                'center_y': cy,
                'type': view_type
            }
            
            if view_type == 'circle':
                result_data['radius'] = info['radius']
                result_data['diameter'] = info['radius'] * 2
            else:
                x, y, w, h = info['bounding_box']
                result_data['width'] = w
                result_data['height'] = h
                result_data['orientation'] = info['orientation']
            
            results.append(result_data)
    
    print("\n" + "=" * 70)
    print(f"📊 处理完成!")
    print(f"   ✅ 成功: {success_count}")
    print(f"   ❌ 失败: {fail_count}")
    print(f"   💾 结果保存在: {output_dir}")
    
    # 保存CSV报告
    if results:
        csv_path = os.path.join(output_dir, "smart_detection_report.csv")
        with open(csv_path, 'w', encoding='utf-8-sig') as f:
            f.write("文件名,视图类型,中心X,中心Y,半径,直径,宽度,高度,方向\n")
            for r in results:
                radius = r.get('radius', '')
                diameter = r.get('diameter', '')
                width = r.get('width', '')
                height = r.get('height', '')
                orientation = r.get('orientation', '')
                f.write(f"{r['file']},{r['view']},{r['center_x']},{r['center_y']},{radius},{diameter},{width},{height},{orientation}\n")
        print(f"   📄 报告已保存: {csv_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("=" * 70)
        print("🔍 智能圆柱体中心检测工具")
        print("=" * 70)
        print("\n使用方法:")
        print("  python smart_detector.py <image_path>          # 处理单张图像")
        print("  python smart_detector.py --batch               # 批量处理")
        print("\n示例:")
        print("  python smart_detector.py \"data/input/p(1).jpg\"")
        print("  python smart_detector.py --batch")
    elif sys.argv[1] == "--batch" or sys.argv[1] == "-b":
        batch_process()
    else:
        process_single_image(sys.argv[1])
