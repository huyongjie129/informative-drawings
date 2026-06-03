# draw_robot — Phase 1 baseline

绘画机器人路径优化项目（UltraArm P340）。Phase 1 实现：机器人运动标定 + 经典 TSP 重排基线。

完整方案见 [/Users/hujiao/.claude/plans/swirling-petting-crayon.md](/Users/hujiao/.claude/plans/swirling-petting-crayon.md)。

## 安装

```bash
pip install -r requirements.txt
```

`elkai`（LKH-3 Python 绑定）为可选；默认走 OR-Tools，安装失败不影响。

## 本地干跑（Mock 机器人，无需真机）

```bash
# 1. 标定（mock 模式：会用一组已知参数模拟，最后拟合结果应接近）
python -m calibration.measure_motion --mock --output motion_params.json

# 2. 跑 baseline：在样例 SVG 上比较默认顺序 vs TSP 重排
python -m baseline.lkh_reorder \
    --svg data/sample_fragmented.svg \
    --motion-params motion_params.json \
    --output outputs/reordered.gcode
```

输出会打印改善百分比。

## 真机模式

```bash
# 需先连上 UltraArm P340（USB serial，通常是 /dev/ttyUSB0 或 /dev/cu.usbserial-*）
python -m calibration.measure_motion --port /dev/ttyUSB0 --output motion_params.json
```

需要 `pymycobot` 包（`pip install pymycobot`）。本仓库默认不强依赖，避免无机器人环境下安装失败。

## 单元测试

```bash
pytest tests/
```

## 目录

- `common/` — 共享：motion model、stroke 数据结构
- `robot/` — 机械臂接口（Real + Mock）、gcode 生成
- `calibration/` — Phase 1：运动参数标定
- `baseline/` — Phase 1：SVG 解析、TSP 求解、端到端重排
- `tests/` — 单元测试
- `data/` — 样例 SVG
- `outputs/` — 生成的 gcode、报告（gitignore）
