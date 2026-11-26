"""
make_test_video.py —— 生成一段用于验收的合成测试视频

为什么要有这个脚本：真实标注视频往往涉及隐私/版权，不方便放进仓库，
而且真实视频不好控制"到底有几个动作"这个 ground truth。合成视频的好处是：
我们知道正确答案（哪几帧在动），可以用来验证 detector 检出的切点是否合理。

生成的视频是一个固定机位的"机械臂"抽象场景：
- 灰色背景 + 一些静态纹理（模拟真实背景的噪声）
- 一个深色方块（机械臂末端）做 4 段来回移动，中间有明显的停顿

动作时间表（30fps，每段之间留足静止间隔）：
  段1：第 30~90 帧  （1.0~3.0 秒）
  段2：第 150~210 帧（5.0~7.0 秒）
  段3：第 270~330 帧（9.0~11.0 秒）
  段4：第 390~450 帧（13.0~15.0 秒）

这样验收时应该至少检出 4 个切点，且曲线图能看到 4 个明显的起落。
"""

import cv2
import numpy as np

WIDTH, HEIGHT = 640, 480
FPS = 30
TOTAL_SECONDS = 18
TOTAL_FRAMES = FPS * TOTAL_SECONDS

OUTPUT = "test_video.mp4"

# (起始帧, 结束帧) —— 机械臂"在动"的时间段，这就是 ground truth
ACTION_RANGES = [
    (30, 90),
    (150, 210),
    (270, 330),
    (390, 450),
]


def make_background():
    """生成一张静态背景：浅灰底 + 固定纹理，模拟真实监控画面的低频细节。"""
    bg = np.full((HEIGHT, WIDTH, 3), 120, dtype=np.uint8)
    # 加一些固定条纹和圆，让画面有结构（否则 absdiff 结果太"干净"）
    cv2.rectangle(bg, (50, 50), (590, 430), (100, 100, 100), 2)
    cv2.circle(bg, (320, 240), 150, (135, 135, 135), 1)
    for x in range(0, WIDTH, 40):
        cv2.line(bg, (x, 0), (x, HEIGHT), (110, 110, 110), 1)
    return bg


def arm_position(frame):
    """
    计算机械臂末端（方块）在某一帧的位置。

    用一个简单的往返运动：在动作段内，方块从左边匀速移动到右边。
    动作段之外，方块停在上一段的终点（静止）。
    """
    # 找到当前帧属于哪个动作段
    for start, end in ACTION_RANGES:
        if start <= frame <= end:
            # 归一化进度 0~1，映射到水平位置
            progress = (frame - start) / (end - start)
            x = int(80 + progress * (WIDTH - 160))
            y = int(HEIGHT / 2)
            return x, y
    # 不在任何动作段内：停在最后一次动作的终点（或初始位置）
    if frame < ACTION_RANGES[0][0]:
        return 80, HEIGHT // 2
    # 找到最近一个已经结束的动作段
    last = ACTION_RANGES[0]
    for s, e in ACTION_RANGES:
        if frame > e:
            last = (s, e)
    return WIDTH - 80, HEIGHT // 2


def main():
    bg = make_background()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT, fourcc, FPS, (WIDTH, HEIGHT))

    if not writer.isOpened():
        raise RuntimeError("无法创建视频文件，请确认已安装 opencv 且磁盘可写")

    for f in range(TOTAL_FRAMES):
        frame = bg.copy()
        x, y = arm_position(f)
        # 机械臂末端：一个深色方块（和背景对比强烈，方便 absdiff 检出）
        cv2.rectangle(frame, (x - 20, y - 20), (x + 20, y + 20), (30, 30, 30), -1)
        # 加一点高斯噪声，模拟真实传感器的随机噪声（让"静止"时曲线也不全为 0）
        noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    print(f"已生成测试视频：{OUTPUT}（{TOTAL_FRAMES} 帧，{FPS} fps，{TOTAL_SECONDS} 秒）")
    print(f"真实动作段（ground truth）：{ACTION_RANGES}")


if __name__ == "__main__":
    main()
