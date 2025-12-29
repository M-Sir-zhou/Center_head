"""
批量处理圆柱体中心检测
使用智能检测器，自动识别端面/侧面视图
"""

import cv2
import numpy as np
import os
from pathlib import Path
from smart_detector import SmartCylinderDetector


def process_batch(input_dir: str = "data/input", output_dir: str = "data/output/center", show_all: bool = False):
    """
    批量处理目录中的所有图像
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        show_all: 是否在可视化中显示所有检测到的圆（端面视图时有效）
    """
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # 获取所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ 在目录 {input_dir} 中未找到图像文件")
        return
    
    print(f"📁 找到 {len(image_files)} 张图像")
    print("=" * 60)
    
    # 创建智能检测器
    detector = SmartCylinderDetector()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计信息
    success_count = 0
    fail_count = 0
    results = []
    
    # 处理每张图像
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] 处理: {image_path.name}")
        
        # 读取图像（支持中文路径）
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            print(f"   ❌ 无法读取图像")
            fail_count += 1
            continue
        
        # 处理图像
        _, info = detector.process_image(image)
        
        # 可视化
        visualized = detector.visualize(image, info)
        
        # 保存结果
        output_path = os.path.join(output_dir, f"{image_path.stem}_result.jpg")
        is_success, buffer = cv2.imencode('.jpg', visualized)
        if is_success:
            buffer.tofile(output_path)
        
        # 打印结果
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
                print(f"      半径: {info['radius']}px, 直径: {info['radius'] * 2}px")
            else:
                x, y, w, h = info['bounding_box']
                result_data['width'] = w
                result_data['height'] = h
                result_data['orientation'] = info['orientation']
                print(f"      尺寸: {w}x{h}px, 方向: {info['orientation']}")
            
            results.append(result_data)
    
    # 打印总结
    print("\n" + "=" * 60)
    print(f"📊 处理完成!")
    print(f"   ✅ 成功: {success_count}")
    print(f"   ❌ 失败: {fail_count}")
    print(f"   💾 结果保存在: {output_dir}")
    
    # 保存CSV报告
    if results:
        csv_path = os.path.join(output_dir, "detection_report.csv")
        with open(csv_path, 'w', encoding='utf-8-sig') as f:
            f.write("文件名,视图类型,中心X,中心Y,半径(px),直径(px),宽度(px),高度(px),方向\n")
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
    
    if "--help" in sys.argv or "-h" in sys.argv:
        print("批量圆形中心检测工具")
        print("\n使用方法:")
        print("  python batch_process.py                    # 使用默认路径")
        print("  python batch_process.py <input_dir>        # 指定输入目录")
        print("  python batch_process.py <input_dir> <output_dir>  # 指定输入输出目录")
        print("  python batch_process.py --show-all         # 显示所有检测到的圆")
        print("\n示例:")
        print("  python batch_process.py")
        print("  python batch_process.py data/input data/output")
    else:
        show_all = "--show-all" in sys.argv
        
        if len(sys.argv) >= 2 and not sys.argv[1].startswith("--"):
            input_dir = sys.argv[1]
            output_dir = sys.argv[2] if len(sys.argv) >= 3 and not sys.argv[2].startswith("--") else "data/output/center"
        else:
            input_dir = "data/input"
            output_dir = "data/output/center"
        
        process_batch(input_dir, output_dir, show_all)
