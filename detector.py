"""
detector.py —— 动作切点检测核心算法

思路：一段固定机位的视频里，主体一旦开始动，相邻两帧的画面差异就会变大；
主体停下来，画面差异又回到噪声水平。所以我们用「帧差法」把每一帧的
"运动量"算成一个标量，得到一条随时间变化的曲线。曲线升起 = 开始动，
曲线回落 = 停下来。剩下的工作就是把"升起"和"回落"这两个时刻找出来。

本文件只负责算法，不碰命令行、不碰文件读写之外的 IO。
一个函数只干一件事，方便单测和讲解。
"""

from dataclasses import dataclass

import cv2
import numpy as np

# ============================================================================
# 可调参数 —— 全部集中在这里，方便你按自己的视频特性去调
# ============================================================================

# 运动量平滑窗口（帧）。相邻帧差值抖动很大，直接找切点会误报，所以先做
# 一个滑动平均把毛刺压掉。窗口越大曲线越光滑，但也越"迟钝"，动作边界会被
# 拉宽。30fps 视频取 5 帧 ≈ 0.17 秒，是一个比较折中的值。
SMOOTH_WINDOW = 5

# 运动量阈值：曲线超过这个值，认为"正在动"。
#
# 帧差法输出的是"灰度差的全局平均"，量级通常在 0~几之间（取决于噪声和目标
# 占画面比例），不同视频差异很大，没法给一个放之四海皆准的常数。
#
# 所以这里用 None 表示"自动估计"：程序会先算一条静止基线（取整条曲线的
# 低分位数，代表"没在动"的水平），再乘以一个余量系数作为阈值。这样对任意
# 视频都能给出一个合理起点，不需要你先跑一遍看量级。估计值偏保守（偏灵敏），
# 宁可多检出让你删，也不要漏掉动作。看完曲线图后，你仍然可以用命令行
# --threshold 显式覆盖成任何值。
MOTION_THRESHOLD = None

# 自动估计阈值时，"静止基线"取曲线的这个分位数（越低越接近纯静止噪声）。
# 一个视频即使动作占比不低，静止帧的噪声也总是集中在分布的低处，所以取
# 一个较低的分位（20%）就能稳定地代表"没在动"的水平，而不会被动作帧拉高。
BASELINE_PERCENTILE = 20

# 自动估计阈值时的倍数系数：阈值 = 静止基线 × 这个倍数。
# 含义：运动量要达到"静止噪声水平"的 2 倍，才认为真的在动。
# 为什么用"基线×倍数"而不是"基线+某个固定值"：噪声是乘性的（压缩、增益
# 都会整体放大噪声），用倍数能在不同噪声水平的视频之间自动换算，更通用。
# 2.0 是一个偏灵敏的起点，宁可多检出让你删，也不要漏掉动作。
THRESHOLD_MARGIN = 2.0

# 最小动作时长（帧）。低于这个长度的"动了一下"大概率是噪声或抖动，不算
# 一个动作。30fps 下 15 帧 ≈ 0.5 秒。
MIN_ACTION_FRAMES = 15

# 两个动作之间的最小间隔（帧）。相邻两个动作如果靠得太近，会被合并成
# 一个。这是为了防止一次动作因为中间短暂停顿被切成两段。
MIN_GAP_FRAMES = 5


@dataclass
class ActionSegment:
    """一个动作切点：从 start_frame 开始动，到 end_frame 停下（均为帧号，含端点）。"""
    start_frame: int
    end_frame: int
    confidence: float


def compute_motion_per_frame(video_path: str) -> tuple[np.ndarray, float]:
    """
    逐帧读视频，用帧差法计算每一帧的"运动量"。

    帧差法：把当前帧转成灰度图，和上一帧做 absdiff，取所有像素差值的
    平均。主体动得越厉害，这个平均值越大；完全静止时接近 0（只剩传感器噪声）。

    为什么用灰度而不是彩色：动作检测只看"有没有变化"，不看颜色，
    灰度图算起来快 3 倍，CPU 上很划算。

    为什么取平均而不是求和：取平均对分辨率不敏感，同一个动作在 640x480
    和 1920x1080 下算出来的值量级接近，阈值更好复用。

    返回 (motion, fps)，motion 是长度为帧数的数组，第 0 帧没有上一帧，
    运动量记为 0；fps 是视频真实帧率（用于后面把帧号换算成秒）。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        # 有些编码的视频 fps 读不到，兜底按 30 处理，避免除零
        fps = 30.0

    motion = []
    prev_gray = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            # 第一帧没有可比较的上一帧，运动量记 0
            motion.append(0.0)
        else:
            diff = cv2.absdiff(gray, prev_gray)
            # diff 是 uint8，直接 .mean() 会溢出/丢精度，先转 float 再求均值
            motion.append(float(diff.astype(np.float32).mean()))

        prev_gray = gray

    cap.release()
    return np.array(motion, dtype=np.float32), float(fps)


def smooth_motion(motion: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """
    用滑动平均平滑运动量曲线，压掉相邻帧的抖动。

    为什么必须平滑：absdiff 对噪声极敏感，哪怕画面完全静止，相邻两帧
    也会因为传感器噪声产生微小的非零差值，曲线像锯齿一样上下跳。直接
    在这些锯齿上找"过阈值"的点，会检出大量假动作。先平均一下，把锯齿
    抹平，真正的动作起伏才会凸现出来。

    用 numpy 的卷积做滑动平均，窗口取奇数；两端用 'same' 保持长度不变，
    边界处因为 padding 会轻微失真，但不影响切点判断（切点不会在视频头尾）。
    """
    if window < 1:
        window = 1
    # 用均匀核做卷积，等价于对每个点取周围 window 帧的平均
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(motion, kernel, mode="same").astype(np.float32)


def find_active_ranges(motion: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """
    找出曲线上所有"高于阈值"的连续区间，返回 [(start, end), ...] 的帧号区间。

    这是找切点的第一步：先把"动"和"不动"二值化（过阈值 = 动），
    然后找出所有连续的"动"区间。后续再对这些区间做过滤和合并。
    """
    active = motion > threshold
    ranges: list[tuple[int, int]] = []
    start = None
    for i, is_active in enumerate(active):
        if is_active and start is None:
            start = i
        elif not is_active and start is not None:
            ranges.append((start, i - 1))
            start = None
    # 视频结尾还处于"动"状态的话，补上最后一段
    if start is not None:
        ranges.append((start, len(motion) - 1))
    return ranges


def filter_and_merge_ranges(
    ranges: list[tuple[int, int]],
    min_frames: int = MIN_ACTION_FRAMES,
    min_gap: int = MIN_GAP_FRAMES,
) -> list[tuple[int, int]]:
    """
    过滤掉过短的动作区间，并合并靠得太近的相邻区间。

    为什么过滤过短的：一次真实的"拿起-放下"动作至少持续零点几秒，
    曲线只冒头一两帧的尖峰多半是噪声，应该丢掉。

    为什么合并靠得近的：一个动作中间如果有个短暂停顿（曲线刚好跌回
    阈值以下又立刻升上来），会被切成两段。把间隔小于 min_gap 的两段
    合并，还原成完整的单个动作。
    """
    if not ranges:
        return []

    result: list[tuple[int, int]] = []
    for start, end in ranges:
        if end - start + 1 < min_frames:
            continue  # 太短，视为噪声丢弃
        if result and start - result[-1][1] <= min_gap:
            # 和前一段贴得太近，合并（前一段已经通过时长过滤）
            result[-1] = (result[-1][0], end)
        else:
            result.append((start, end))
    return result


def score_confidence(motion: np.ndarray, start: int, end: int, threshold: float) -> float:
    """
    给一个动作区间算置信度（0~1）。

    思路：动作期间曲线超出阈值越多，说明越"真"、越不像噪声。这里用一个
    简单直观的映射：把「峰值 / 阈值」这个比值压进 0~1 区间。
      - 峰值刚好等于阈值（比值 1.0）→ 置信度 0，勉强压线，最不可信；
      - 峰值是阈值的 2 倍以上 → 置信度趋近 1，非常可信。
    公式用 (x-1)/x，x=峰值/阈值：x=1 时得 0，x 越大越接近 1，单调且不超界。

    为什么需要 confidence：帧差法本质是启发式，一定会误报。给每个切点
    一个置信度，用户后续人工复核时可以优先看低分的，省时间。
    """
    seg = motion[start : end + 1]
    peak = float(seg.max())
    if threshold <= 0:
        return 0.0
    ratio = peak / threshold  # 峰值相对阈值的倍数，>=1（区间内必然有超阈值的帧）
    # (x-1)/x：x=1 → 0，x=2 → 0.5，x 越大越接近 1
    confidence = (ratio - 1.0) / ratio
    return round(float(np.clip(confidence, 0.0, 1.0)), 3)


def estimate_threshold(
    motion: np.ndarray,
    percentile: int = BASELINE_PERCENTILE,
    margin: float = THRESHOLD_MARGIN,
) -> float:
    """
    自动估计一个合理的运动量阈值。

    为什么这么估：一个固定机位视频里，静止帧的运动量（传感器噪声）总是
    集中在分布的低处，而动作帧的运动会明显更高。所以：
      1. 取一个低分位（默认 20%）作为"静止基线"，它不会被动作帧拉高；
      2. 阈值 = 基线 × 倍数（默认 2.0）。

    为什么用「基线×倍数」而不是「基线+固定值」：噪声大体是乘性的——画面
    更亮、增益更高、压缩更强，噪声会整体按比例放大，动作信号也会同比例
    放大。用倍数能自动适应这种整体缩放，而加一个固定值不行。

    为什么取分位数而不是最小值/平均值：最小值可能被个别"极安静"的帧拉低
    （甚至开头第一帧被人工填 0）；平均值又会被动作段拉高。低分位（20%）
    既躲开离群点、又稳定代表静止噪声水平，是最稳的锚点。

    局限：如果动作信号本身很弱（信噪比接近 1，动作峰值只比噪声高一点点），
    任何自动阈值都救不回来，那种视频需要人工用 --threshold 精调，或干脆
    换光流等更强的运动估计。这是帧差法的物理边界，不是调参能解决的。
    """
    if motion.size == 0:
        return 0.0
    baseline = float(np.percentile(motion, percentile))
    # 兜底：极端干净的合成视频基线可能是 0，给一个最小正数避免阈值为 0
    baseline = max(baseline, 1e-3)
    return baseline * margin


def detect_actions(
    video_path: str,
    threshold: float | None = MOTION_THRESHOLD,
    smooth_window: int = SMOOTH_WINDOW,
    min_frames: int = MIN_ACTION_FRAMES,
    min_gap: int = MIN_GAP_FRAMES,
) -> tuple[list[ActionSegment], np.ndarray, np.ndarray, float]:
    """
    主入口：读视频 → 算运动量 → 平滑 → 找切点。

    阈值可以显式指定；传 None（默认）则自动估计，见 estimate_threshold。

    返回 (segments, raw_motion, smoothed_motion, fps)：
    - segments：检测到的动作切点列表
    - raw_motion / smoothed_motion：原始和平滑后的运动量曲线（给画图用）
    - fps：视频帧率（给换算秒数用）
    """
    raw, fps = compute_motion_per_frame(video_path)
    smoothed = smooth_motion(raw, smooth_window)

    if threshold is None:
        threshold = estimate_threshold(smoothed)

    ranges = find_active_ranges(smoothed, threshold)
    ranges = filter_and_merge_ranges(ranges, min_frames, min_gap)

    segments = [
        ActionSegment(
            start_frame=start,
            end_frame=end,
            confidence=score_confidence(smoothed, start, end, threshold),
        )
        for start, end in ranges
    ]
    return segments, raw, smoothed, fps
