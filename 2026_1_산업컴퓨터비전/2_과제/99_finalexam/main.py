"""Matplotlib GUI launcher for the mosquito glue-tape analyzer.

동작 방식
1) 프로그램 시작 시 빈 흰 화면만 표시합니다.
2) 상단 Matplotlib toolbar의 '📂 열기' 버튼을 누르면 이미지 파일 선택창이 열립니다.
3) 선택한 이미지를 분석하고 결과 overlay를 같은 창에 표시합니다.
4) ESC 또는 Ctrl+O를 누르면 다시 이미지 파일 선택창이 열립니다.
"""
from __future__ import annotations

from pathlib import Path
import sys

import cv2

# Tk 기반 Matplotlib 창을 명시적으로 사용합니다.
# PyCharm/Windows 환경에서 Figure toolbar를 사용하기 위한 설정입니다.
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    # 이미 backend가 초기화된 경우 무시합니다.
    pass


def setup_korean_matplotlib_font() -> None:
    """Matplotlib 제목/텍스트에서 한글이 깨지지 않도록 폰트를 설정합니다."""
    from matplotlib import font_manager

    font_paths = [
        Path("C:/Windows/Fonts/malgun.ttf"),          # Windows: 맑은 고딕
        Path("C:/Windows/Fonts/malgunbd.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    ]

    for font_path in font_paths:
        if font_path.exists():
            try:
                font_manager.fontManager.addfont(str(font_path))
                font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
                matplotlib.rcParams["font.family"] = font_name
                matplotlib.rcParams["axes.unicode_minus"] = False
                return
            except Exception:
                continue

    # 폰트 파일을 못 찾은 경우에도 일반적으로 설치된 한글 폰트명을 시도합니다.
    matplotlib.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


setup_korean_matplotlib_font()

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from config import AnalyzerConfig, IMAGE_EXTS, SAMPLE_DIR
from image_analyzer import analyze_image, count_results, read_image_bgr


try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:  # pragma: no cover - GUI 없는 환경 대비
    tk = None
    filedialog = None
    messagebox = None


class MosquitoAnalyzerApp:
    """Matplotlib toolbar에 파일 열기 버튼을 추가한 GUI 앱."""

    def __init__(self) -> None:
        self.config = AnalyzerConfig(
            hole_pitch_mm=None,
            hole_diameter_mm=5.0,      # 실제 홀 지름 5mm를 기준자로 사용
            show_result_window=False,  # cv2.imshow 사용하지 않고 matplotlib 창에 표시
        )

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("Mosquito Glue Tape Analyzer")
        self.ax.axis("off")
        self.fig.patch.set_facecolor("white")
        self.ax.set_facecolor("white")
        self.root = self.fig.canvas.get_tk_widget().winfo_toplevel()

        self._setup_toolbar_open_button()
        self._setup_key_events()
        self._show_blank_canvas()

        print("Mosquito Glue Tape Analyzer - Improved Accuracy Version")
        print("- 상단 toolbar의 '📂 열기' 버튼을 눌러 모기 이미지를 선택하세요.")
        print("- ESC 또는 Ctrl+O를 누르면 다른 이미지를 다시 선택합니다.")
        print("- 결과는 outputs/overlay, outputs/crops, outputs/masks, outputs/reports 에 저장됩니다.\n")

    def _setup_toolbar_open_button(self) -> None:
        """기본 Matplotlib toolbar에 파일 열기 버튼을 추가합니다."""
        if tk is None:
            return

        # Matplotlib FigureManager가 생성한 toolbar를 찾습니다.
        manager = self.fig.canvas.manager
        toolbar = getattr(manager, "toolbar", None)

        # 환경에 따라 toolbar가 아직 없을 수 있으므로 직접 생성합니다.
        if toolbar is None:
            toolbar = NavigationToolbar2Tk(self.fig.canvas, self.root, pack_toolbar=True)
            manager.toolbar = toolbar

        # 구분선
        sep = tk.Frame(toolbar, width=2, bg="#b0b0b0")
        sep.pack(side=tk.LEFT, padx=4, pady=2, fill=tk.Y)

        # 파일 열기 버튼. 이미지 파일 없이도 동작하도록 텍스트 아이콘을 사용합니다.
        open_btn = tk.Button(
            master=toolbar,
            text="📂 열기",
            command=self.open_and_analyze_image,
            relief=tk.FLAT,
            padx=6,
            pady=2,
        )
        open_btn.pack(side=tk.LEFT, padx=2)

        toolbar.update()

    def _setup_key_events(self) -> None:
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    def _on_key_press(self, event) -> None:
        if event.key in {"escape", "ctrl+o", "cmd+o"}:
            self.open_and_analyze_image()

    def _show_blank_canvas(self) -> None:
        """프로그램 시작 시 아무 이미지도 표시하지 않는 흰 여백 상태."""
        self.ax.clear()
        self.ax.axis("off")
        self.ax.set_facecolor("white")
        self.fig.patch.set_facecolor("white")
        self.fig.canvas.draw_idle()

    def _select_image_file(self) -> Path | None:
        if filedialog is None:
            print("파일 선택창을 열 수 없습니다. 콘솔에서 이미지 경로를 입력하세요.")
            text = input("이미지 경로: ").strip().strip('"')
            return Path(text) if text else None

        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("All files", "*.*"),
        ]
        initial_dir = SAMPLE_DIR if SAMPLE_DIR.exists() else Path.cwd()
        filename = filedialog.askopenfilename(
            parent=self.root if tk is not None else None,
            title="분석할 모기 이미지를 선택하세요",
            initialdir=str(initial_dir),
            filetypes=filetypes,
        )
        return Path(filename) if filename else None

    def open_and_analyze_image(self) -> None:
        """파일을 선택하고 분석 후 overlay 결과를 현재 Matplotlib 창에 표시합니다."""
        selected = self._select_image_file()
        if selected is None:
            return

        if selected.suffix.lower() not in IMAGE_EXTS:
            msg = f"지원하지 않는 이미지 확장자입니다: {selected.suffix}"
            print(msg)
            if messagebox is not None:
                messagebox.showwarning("Unsupported file", msg, parent=self.root)
            return

        try:
            results, overlay_path = analyze_image(selected, config=self.config)
            self._display_overlay(overlay_path, selected.name, results)
        except Exception as exc:
            msg = f"오류: {exc}"
            print(msg)
            if messagebox is not None:
                messagebox.showerror("Analysis error", msg, parent=self.root)

    def _display_overlay(self, overlay_path: Path, source_name: str, results) -> None:
        overlay_bgr = read_image_bgr(overlay_path)
        if overlay_bgr is None:
            raise RuntimeError(f"결과 이미지를 읽을 수 없습니다: {overlay_path}")

        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        counts = count_results(results)

        self.ax.clear()
        self.ax.imshow(overlay_rgb)
        self.ax.axis("off")
        self.ax.set_title(
            f"{source_name}  |  한국숲모기: {counts['korean_forest']}  "
            f"갈색모기: {counts['brown_mosquito']}  Unknown/Review: {counts['unknown_review']}",
            fontsize=12,
        )
        self.fig.canvas.draw_idle()


def main() -> None:
    if tk is None:
        print("Tkinter GUI를 사용할 수 없습니다. Python 설치 상태를 확인하세요.")
        sys.exit(1)

    app = MosquitoAnalyzerApp()
    plt.show()


if __name__ == "__main__":
    main()
