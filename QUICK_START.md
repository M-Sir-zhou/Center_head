# 快速使用指南

## 🎯 最简单的使用方法

### 1. 安装依赖
```bash
pip install opencv-python numpy
```

### 2. 批量处理所有图像（推荐）

#### 智能中心检测
```bash
python batch_process.py
```
- ✅ 自动处理 `data/input/` 目录下的所有图像
- ✅ 自动识别端面视图和侧面视图
- ✅ 结果保存到 `data/output/center/`
- ✅ 生成详细的 CSV 报告

#### 首尾识别检测
```bash
python head_tail_detector.py --batch
```
- ✅ 自动处理 `data/input_is_head/` 目录下的所有图像
- ✅ 根据两端截面条纹特征判断首尾
- ✅ 结果保存到 `data/output/head/`
- ✅ 生成 `head_tail_report.csv` 报告

### 3. 处理单张图像

#### 中心检测
```bash
python smart_detector.py "data/input/your_image.jpg"
```

#### 首尾识别
```bash
python head_tail_detector.py "data/input_is_head/1/h (1).jpg"
```

## 📸 支持的功能类型

### 中心检测

#### 侧面视图（Side View）
- 从侧面拍摄的圆柱体
- 看到的是矩形外形
- 可以是水平或垂直放置
- 自动检测轮廓并计算中心

#### 端面视图（End View）  
- 从上方或前方拍摄的圆柱体
- 看到的是圆形端面
- 自动检测圆形并计算圆心
- 测量半径和直径

### 首尾识别（新功能）

根据两端截面的条纹特征自动判断圆柱体的头部和尾部：
- **头部**：存在明显条纹的一端
- **尾部**：无明显条纹的一端
- 支持水平和垂直放置
- 提供置信度评分

**检测特征：**
- 边缘密度分析
- 梯度强度检测
- 纹理复杂度评估
- 频域周期性分析

## 🎨 检测结果说明

### 中心检测可视化标注
- 🔴 红色点：中心位置
- 🟢 绿色线：轮廓/圆形
- 🔵 蓝色十字：中心标记线
- 📝 白色文字：检测信息

### 首尾识别可视化标注
- 🔴 HEAD 红色：头部位置及箭头
- 🟢 TAIL 绿色：尾部位置及箭头
- 📝 白色文字：方向、置信度、得分信息

### 输出文件
- **中心检测**（保存在 `data/output/center/`）：
  - `[文件名]_result.jpg` - 可视化结果图像
  - `detection_report.csv` - 批量处理报告
  
- **首尾识别**（保存在 `data/output/head/`）：
  - `[文件名]_head_tail.jpg` - 可视化结果图像
  - `head_tail_report.csv` - 批量处理报告

## ⚠️ 注意事项

### Windows 用户
文件名包含括号或空格时，必须用引号：
```bash
# ✅ 正确
python smart_detector.py "data/input/p(1).jpg"

# ❌ 错误  
python smart_detector.py data/input/p(1).jpg
```

### 获得最佳检测效果
1. ✅ 确保图像清晰
2. ✅ 圆柱体与背景有明显对比
3. ✅ 光照均匀
4. ✅ 避免复杂背景

## 🔧 故障排除

### 检测不到中心？
1. 检查图像是否清晰
2. 尝试裁剪掉无关背景
3. 提高图像对比度
4. 确保圆柱体完整可见

### 路径错误？
- 确保使用引号包裹路径
- 使用绝对路径或相对路径
- 检查文件名拼写

## 📞 获取帮助

查看完整文档：
```bash
# 查看 README.md 获取详细信息
```

运行示例：
```bash
# 批量处理
python batch_process.py

# 单张处理  
python smart_detector.py "data/input/p(1).jpg"
```
