import pyautogui
import cv2
import numpy as np
import time
import os
import sys
import tempfile
import shutil


# 辅助函数：获取资源文件路径（适应打包环境）
def get_resource_path(relative_path):
    """获取资源文件的绝对路径，无论是在开发环境还是打包后的环境"""
    try:
        # PyInstaller创建临时文件夹，并将路径存储在_MEIPASS中
        base_path = sys._MEIPASS
        # 确保路径是ASCII编码，避免中文乱码问题
        if hasattr(sys, '_MEIPASS'):
            # 如果是打包环境，将资源文件复制到临时目录
            temp_dir = tempfile.gettempdir()
            resource_temp_path = os.path.join(temp_dir, os.path.basename(relative_path))

            # 如果文件不存在，则从MEIPASS复制
            if not os.path.exists(resource_temp_path):
                try:
                    src_path = os.path.join(base_path, relative_path)
                    shutil.copy2(src_path, resource_temp_path)
                    print(f"已复制资源文件到: {resource_temp_path}")
                except Exception as e:
                    print(f"复制资源文件失败: {e}")
                    # 回退到直接使用MEIPASS路径
                    return os.path.join(base_path, relative_path)

            return resource_temp_path
        else:
            return os.path.join(base_path, relative_path)

    except Exception:
        # 开发环境
        base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)


class MultiScalePatternClicker:
    def __init__(self, pattern_path, confidence=0.8, check_interval=1.0,
                 scale_range=(0.5, 2.0), scale_steps=20):
        """
        初始化多尺度图案点击器

        :param pattern_path: 要识别的图案文件路径
        :param confidence: 匹配置信度阈值 (0-1)
        :param check_interval: 检查间隔时间(秒)
        :param scale_range: 缩放范围 (最小比例, 最大比例)
        :param scale_steps: 缩放步数
        """
        # 确保文件存在
        if not os.path.exists(pattern_path):
            raise ValueError(f"图案文件不存在: {pattern_path}")

        # 加载目标图案（使用资源路径处理函数）
        self.pattern = cv2.imread(pattern_path)
        if self.pattern is None:
            # 尝试使用不同的读取方式
            try:
                with open(pattern_path, 'rb') as f:
                    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                    self.pattern = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            except Exception as e:
                raise ValueError(f"无法加载图案文件: {pattern_path}, 错误: {e}")

        if self.pattern is None:
            raise ValueError(f"无法加载图案文件: {pattern_path}")

        self.confidence = confidence
        self.check_interval = check_interval
        self.is_running = False
        self.original_height, self.original_width = self.pattern.shape[:2]

        # 多尺度参数
        self.scale_range = scale_range
        self.scale_steps = scale_steps

        # 获取屏幕尺寸
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"屏幕尺寸: {self.screen_width}x{self.screen_height}")
        print(f"原始图案尺寸: {self.original_width}x{self.original_height}")
        print(f"缩放范围: {scale_range[0]} - {scale_range[1]}, 步数: {scale_steps}")

    def find_pattern_multiscale(self, screenshot):
        """
        多尺度模板匹配

        :param screenshot: 屏幕截图
        :return: (x, y, 置信度, 缩放比例) 或 (None, None, 0, 1.0)
        """
        best_match = None
        best_confidence = 0
        best_scale = 1.0

        # 生成多个缩放比例
        for scale in np.linspace(self.scale_range[0], self.scale_range[1], self.scale_steps):
            try:
                # 缩放模板图像
                if scale != 1.0:
                    new_width = int(self.original_width * scale)
                    new_height = int(self.original_height * scale)

                    # 确保缩放后的尺寸合理
                    if new_width < 10 or new_height < 10:
                        continue
                    if new_width > screenshot.shape[1] or new_height > screenshot.shape[0]:
                        continue

                    resized_pattern = cv2.resize(self.pattern, (new_width, new_height),
                                                 interpolation=cv2.INTER_AREA)
                else:
                    resized_pattern = self.pattern

                # 模板匹配
                result = cv2.matchTemplate(screenshot, resized_pattern, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                # 更新最佳匹配
                if max_val > best_confidence and max_val >= self.confidence:
                    best_confidence = max_val
                    best_match = max_loc
                    best_scale = scale

            except Exception as e:
                print(f"缩放比例 {scale:.2f} 时出错: {e}")
                continue

        if best_match and best_confidence >= self.confidence:
            # 计算中心位置
            scaled_width = int(self.original_width * best_scale)
            scaled_height = int(self.original_height * best_scale)

            center_x = best_match[0] + scaled_width // 2
            center_y = best_match[1] + scaled_height // 2

            return center_x, center_y, best_confidence, best_scale

        return None, None, 0, 1.0

    def click_pattern(self, x, y):
        """
        点击指定位置
        """
        try:
            pyautogui.click(x, y)
            time.sleep(0.1)  # 短暂暂停以避免过快点击
            pyautogui.click(x, y)
            # 确保点击生效
            print(f"已点击位置: ({x}, {y})")
            return True
        except Exception as e:
            print(f"点击时出错: {e}")
            return False

    def start_detection(self):
        """开始持续检测"""
        self.is_running = True
        detection_count = 0
        found_count = 0

        print("开始多尺度图案检测... (按Ctrl+C停止)")

        try:
            while self.is_running:
                detection_count += 1

                # 截取屏幕
                screenshot = pyautogui.screenshot()
                screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

                # 多尺度查找图案
                x, y, confidence, scale = self.find_pattern_multiscale(screenshot_cv)

                if x is not None and y is not None:
                    found_count += 1
                    print(
                        f"[{detection_count}] 找到图案! 置信度: {confidence:.3f}, 缩放: {scale:.2f}x, 位置: ({x}, {y})")
                    # 点击图案
                    self.click_pattern(x, y)

                    # 点击后短暂暂停
                    time.sleep(0.5)
                else:
                    # 每10次检测输出一次状态
                    if detection_count % 10 == 0:
                        print(f"[{detection_count}] 检测中... 已发现 {found_count} 次图案")

                # 等待下一次检测
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("检测已手动停止")
        finally:
            self.is_running = False
            print(f"检测结束. 总共检测: {detection_count} 次, 发现图案: {found_count} 次")


# 使用示例
if __name__ == "__main__":
    try:
        # 获取图案文件的正确路径（使用资源路径处理函数）
        pattern_path = get_resource_path("box.png")
        print(f"使用图案文件: {pattern_path}")

        # 初始化多尺度图案点击器
        clicker = MultiScalePatternClicker(
            pattern_path=pattern_path,
            confidence=0.7,  # 置信度阈值（多尺度匹配可以适当降低）
            check_interval=1.0,  # 每1秒检查一次
            scale_range=(0.5, 2.0),  # 缩放范围：50% 到 200%
            scale_steps=15  # 缩放步数
        )

        # 开始检测
        clicker.start_detection()
    except Exception as e:
        print(f"程序启动失败: {e}")
        import traceback

        traceback.print_exc()  # 打印详细错误信息
        input("按回车键退出...")  # 保持窗口打开以便查看错误