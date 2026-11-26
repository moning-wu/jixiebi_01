"""
main.py —— 入口脚本

一条命令跑通：python main.py <视频文件>

做的事情：读视频 → 检测动作切点 → 输出两个结果：
  1. 一个 CSV（默认 actions.csv），每条记录 start_frame / end_frame /
     duration_sec / confidence；
  2. 一张运动量曲线图（默认 motion_curve.png），把检测到的切点用颜色标出来。

你只需要改 detector.py 顶部的参数，不需要动本文件的逻辑。
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

# 无窗口环境下（例如远程跑、CI）用非交互后端，避免弹不出窗口而报错
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np

from detector import (  # noqa: E402
    MOTION_THRESHOLD,
    SMOOTH_WINDOW,
    MIN_ACTION_FRAMES,
    MIN_GAP_FRAMES,
    detect_actions,
    estimate_threshold,
)


def frames_to_seconds(frame: int, fps: float) -> float:
    """帧号换算成秒。用于输出更符合人直觉的 duration_sec。"""
    return frame / fps


def build_records(segments, fps):
    """
    把检测结果整理成列表字典，方便同时写 CSV 和打印到屏幕。

    字段说明：
    - start_frame / end_frame：起止帧号（0 基，含端点）
    - duration_sec：动作持续秒数
    - confidence：置信度 0~1，越高越可信
    """
    records = []
    for seg in segments:
        records.append(
            {
                "start_frame": seg.start_frame,
                "end_frame": seg.end_frame,
                "duration_sec": round(frames_to_seconds(seg.end_frame - seg.start_frame + 1, fps), 3),
                "confidence": seg.confidence,
            }
        )
    return records


def write_csv(records, output_path):
    """把切点列表写成 CSV。"""
    fieldnames = ["start_frame", "end_frame", "duration_sec", "confidence"]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def plot_motion(raw, smoothed, segments, threshold, fps, output_path):
    """
    画运动量曲线，并把切点标注出来。

    为什么同时画 raw 和 smoothed：让用户直观看到平滑前后的差异，
    理解阈值是在哪条线上生效的。切点用浅色区间背景 + 起止竖线标出。
    """
    secs = np.arange(len(smoothed)) / fps  # 横轴用秒，比帧号更直观

    fig, ax = plt.subplots(figsize=(12, 5))

    # 原始曲线：半透明，展示噪声有多大
    ax.plot(secs, raw, color="gray", alpha=0.4, linewidth=0.6, label="raw motion")
    # 平滑后的曲线：主体，阈值作用在这条线上
    ax.plot(secs, smoothed, color="#1f77b4", linewidth=1.4, label="smoothed motion")

    # 阈值线：一条水平虚线，直观展示"过线=动"
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1, label=f"threshold = {threshold}")

    # 逐个动作画背景色块 + 起止线
    for i, seg in enumerate(segments):
        s, e = seg.start_frame / fps, seg.end_frame / fps
        ax.axvspan(s, e, color="#ff7f0e", alpha=0.18)
        ax.axvline(s, color="#d62728", linestyle=":", linewidth=1)
        ax.axvline(e, color="#2ca02c", linestyle=":", linewidth=1)

    # 只在有切点时标注一次图例说明（避免 legend 重复）
    if segments:
        ax.axvspan(0, 0, color="#ff7f0e", alpha=0.18, label="action segment")
        ax.axvline(0, color="#d62728", linestyle=":", linewidth=1, label="start")
        ax.axvline(0, color="#2ca02c", linestyle=":", linewidth=1, label="end")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("motion (mean absdiff)")
    ax.set_title(f"Motion curve & detected action segments ({len(segments)} actions)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="从固定机位视频中检测动作切点（起止帧号）"
    )
    parser.add_argument("video", help="输入视频路径（mp4 / avi）")
    parser.add_argument("--csv", default="actions.csv", help="切点输出 CSV 路径")
    parser.add_argument("--plot", default="motion_curve.png", help="曲线图输出路径")
    parser.add_argument("--threshold", type=float, default=MOTION_THRESHOLD,
                        help="运动量阈值（默认自动估计，可显式覆盖）")
    parser.add_argument("--smooth", type=int, default=SMOOTH_WINDOW, help="平滑窗口（帧）")
    parser.add_argument("--min-frames", type=int, default=MIN_ACTION_FRAMES, help="最小动作时长（帧）")
    args = parser.parse_args(argv)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"错误：找不到视频文件 {video_path}", file=sys.stderr)
        return 1

    print(f"开始处理：{video_path}")
    segments, raw, smoothed, fps = detect_actions(
        str(video_path),
        threshold=args.threshold,
        smooth_window=args.smooth,
        min_frames=args.min_frames,
    )

    # detect_actions 内部可能在 threshold 为 None 时自动估了一个值，
    # 这里再算一次同样的值，纯粹是为了画图时画出那条阈值线、并打印给用户看。
    actual_threshold = args.threshold if args.threshold is not None else estimate_threshold(smoothed)

    print(f"视频帧率：{fps:.2f} fps，共 {len(raw)} 帧")
    print(f"运动量阈值：{actual_threshold:.3f}" + ("（自动估计）" if args.threshold is None else "（手动指定）"))
    print(f"检测到 {len(segments)} 个动作切点：\n")

    records = build_records(segments, fps)
    for r in records:
        print(
            f"  动作 {r['start_frame']:>6} → {r['end_frame']:>6} 帧"
            f"（{r['duration_sec']:.2f} 秒），置信度 {r['confidence']}"
        )

    write_csv(records, args.csv)
    print(f"\n切点已写入：{args.csv}")

    plot_motion(raw, smoothed, segments, actual_threshold, fps, args.plot)
    print(f"曲线图已保存：{args.plot}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
