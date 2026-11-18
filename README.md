# 时序笔触重建（Three.js）

本项目把一段参考视频离线解析为可执行的绘图指令，在浏览器运行时用 Three.js 的 GPU 实例化图元重新绘制画面。每条笔触都有几何、颜色、笔宽、透明度，以及随时间变化的位移、缩放、旋转、亮度和透明度参数；原始视频不参与网页运行时播放。

参考视频为 1504×704、24 fps、124 帧；分析统一缩放到 752×352。离线分析先估计逐帧仿射相机变换并生成稳定的时域中值画面，再拟合 SLIC 椭圆、Sobel 线段/二次贝塞尔笔触、残差软笔刷与时序闪动粒子。播放器合成低频基底和全部几何指令，在 shader 中按 13 组关键帧逐笔触插值动画。

## 文件

- `index.html` / `styles.css`：播放器界面
- `src/player.js`：Three.js 图元绘图函数、相机变换、关键帧插值与时序动画
- `src/main.js`：播放控制、错误诊断与统计信息
- `src/instructions.js`：浏览器和 Node 共用的二进制指令解析器
- `src/three.module.js` / `src/three.core.js`：Three.js r180（MIT）
- `assets/strokes.bin`：从参考视频提取的 23,664 条二进制绘图指令
- `assets/base.jpg`：视频时域中值经边缘保持平滑生成的低频基底
- `assets/strokes.manifest.json`：指令格式和相机关键帧的可读说明
- `tools/extract_strokes.py`：视频分析、矢量化、时间参数拟合与指令打包
- `tools/prepare_frames.ps1`：使用 ffmpeg 解码参考视频

## 本地运行

这是静态 ES Module 工程，需要通过 HTTP 服务打开：

```powershell
node tools/serve.js
```

默认访问 `http://127.0.0.1:8765/`。可用 `PORT=8791 node tools/serve.js` 指定端口。播放器支持暂停、拖动时间轴和从起点重播。

## 指令格式

`assets/strokes.bin` 以 `VSTK0003` 开头，头部记录版本、关键帧数、JSON 长度和笔触数；随后是相机逆仿射矩阵和每条笔触的指令记录：

- `kind`：椭圆、线段、二次贝塞尔、闪动、软笔刷
- 12 个几何参数：中心、半径、角度、控制点、端点、线宽等
- RGB 颜色和基础透明度
- 13 组时间关键帧：`dx, dy, scale, rotate, delta_alpha, brightness, time, wobble`

当前数据包含 10,176 个椭圆、538 条线段、2,150 条贝塞尔曲线、9,000 个软笔刷和 1,800 个闪动笔触。每条记录都有独立的时间参数；相机也在 13 个时间点之间平滑插值。具体格式见 `assets/strokes.manifest.json`。

```powershell
node tools/check-instructions.mjs assets/strokes.bin
```

该命令检查指令头、每条笔触的长度、图层数量、缩放/旋转范围和全部浮点参数。

## 重新生成指令

安装 Python 依赖并解码本地参考视频：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy opencv-python-headless scipy scikit-image pillow
powershell -File tools\prepare_frames.ps1 -Video "C:\path\to\reference.mp4"
.\.venv\Scripts\python.exe tools\extract_strokes.py --video "C:\path\to\reference.mp4" --work analysis --output assets\strokes.bin --ellipses 12000 --edges 9000 --brushes 9000
```

`tools/prepare_frames.ps1` 需要系统 PATH 中可用的 ffmpeg。抽取器会同时更新 `assets/base.jpg`、`assets/strokes.manifest.json` 和 `analysis/reconstruction_t0.png` 预览。

## 还原边界

参考视频是高精度数字插画，包含人物细节、纹理与半透明特效。有限数量的几何笔触可保留构图、配色、轮廓与整体运动，但无法逐像素复现压缩视频里的全部纹理。低频基底用于填补有限图元之间的空隙，线条、笔刷和闪动则由运行时绘图函数叠加。增加 `--ellipses`、`--edges` 和 `--brushes` 可以提高细节，代价是更大的指令文件和更高的 GPU 填充率。

## 许可

Three.js 使用 MIT 许可，许可证文本见 `assets/THREE-LICENSE.txt`。参考视频及由其生成的绘图数据仅作为本项目的输入和演示资产。
