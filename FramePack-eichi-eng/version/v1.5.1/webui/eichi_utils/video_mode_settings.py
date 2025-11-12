# -*- coding: utf-8 -*-
"""
FramePack-eichi 動画モード設定モジュール
柔軟なモード設定のための構造化データと関連ユーティリティ関数を提供します
"""

import math
import gradio as gr

# モードタイプの定数定義
MODE_TYPE_NORMAL = "通常"
MODE_TYPE_LOOP = "ループ"

# ビデオモード設定の定義
# このデータ構造がモード選択の中心となります
VIDEO_MODE_SETTINGS = {
    "1秒": {
        "frames": 30,                   # 1秒×30FPS
        "sections": 1,                  # 必要セクション数（正確な計算に基づく）
        "display_seconds": 1.0,         # UI表示用秒数
        "important_keyframes": [0],     # 重要なキーフレームのインデックス（0始まり）
        "copy_patterns": {
            "通常": {
                "0": [],                # 短いためコピー不要
            },
            "ループ": {
                "0": [],                # 短いためコピー不要
            }
        }
    },
    "6秒": {
        "frames": 180,                  # 6秒×30FPS
        "sections": 6,                  # 必要セクション数（正確な計算に基づく）
        "display_seconds": 6.0,         # UI表示用秒数
        "important_keyframes": [0],     # 重要なキーフレームのインデックス（0始まり）
        "copy_patterns": {
            "通常": {
                "0": [1, 2, 3],         # キーフレーム0→1,2,3にコピー
            },
            "ループ": {
                "0": [1, 2, 3],         # キーフレーム0→1,2,3にコピー
            }
        }
    },
    "8秒": {
        "frames": 240,                  # 8秒×30FPS
        "sections": 8,                  # 必要セクション数
        "display_seconds": 8.0,         # UI表示用秒数
        "important_keyframes": [0],     # 重要なキーフレームのインデックス（0始まり）
        "copy_patterns": {
            "通常": {
                "0": [1, 2, 3, 4, 5],   # キーフレーム0→1,2,3,4,5にコピー
            },
            "ループ": {
                "0": [1, 2, 3, 4, 5],   # キーフレーム0→1,2,3,4,5にコピー
            }
        }
    },
    "10(5x2)秒": {
        "frames": 324,                  # 10.8秒×30FPS
        "sections": 10,                 # 必要セクション数
        "display_seconds": 10.8,        # UI表示用秒数
        "important_keyframes": [0, 4],  # 重要なキーフレームのインデックス
        "copy_patterns": {
            "通常": {
                "0": [1, 2, 3],         # キーフレーム0→1,2,3にコピー
                "4": [5, 6, 7],         # キーフレーム4→5,6,7にコピー
            },
            "ループ": {
                "0": [1, 2, 3],         # キーフレーム0→1,2,3にコピー
                "4": [5, 6, 7],         # キーフレーム4→5,6,7にコピー
            }
        }
    },
    "12(4x3)秒": {
        "frames": 360,                  # 12秒×30FPS
        "sections": 12,                 # 必要セクション数
        "display_seconds": 12.0,        # UI表示用秒数
        "important_keyframes": [0, 3, 6], # 重要なキーフレームのインデックス
        "copy_patterns": {
            "通常": {
                "0": [1, 2],            # キーフレーム0→1,2にコピー
                "3": [4, 5],            # キーフレーム3→4,5にコピー
                "6": [7, 8],            # キーフレーム6→7,8にコピー
            },
            "ループ": {
                "0": [1, 2],            # キーフレーム0→1,2にコピー
                "3": [4, 5],            # キーフレーム3→4,5にコピー
                "6": [7, 8],            # キーフレーム6→7,8にコピー
            }
        }
    },
    # 新しい16(4x4)秒モード
    "16(4x4)秒": {
        "frames": 480,                  # 16秒×30FPS
        "sections": 15,                 # 必要セクション数（計算に基づく）
        "display_seconds": 16.0,        # UI表示用秒数
        "important_keyframes": [0, 3, 6, 9], # 重要なキーフレームのインデックス
        "copy_patterns": {
            "通常": {
                "0": [1, 2],            # キーフレーム0→1,2にコピー
                "3": [4, 5],            # キーフレーム3→4,5にコピー
                "6": [7, 8],            # キーフレーム6→7,8にコピー
                "9": [10, 11],          # キーフレーム9→10,11にコピー
            },
            "ループ": {
                "0": [1, 2],            # キーフレーム0→1,2にコピー
                "3": [4, 5],            # キーフレーム3→4,5にコピー
                "6": [7, 8],            # キーフレーム6→7,8にコピー
                "9": [10, 11],          # キーフレーム9→10,11にコピー
            }
        }
    },
    # 新しい20(4x5)秒モード
    "20(4x5)秒": {
        "frames": 624,                  # 20.8秒×30FPS
        "sections": 19,                 # 必要セクション数（計算に基づく）
        "display_seconds": 20.8,        # UI表示用秒数
        "important_keyframes": [0, 3, 6, 9, 12], # 重要なキーフレームのインデックス
        "copy_patterns": {
            "通常": {
                "0": [1, 2],            # キーフレーム0→1,2にコピー
                "3": [4, 5],            # キーフレーム3→4,5にコピー
                "6": [7, 8],            # キーフレーム6→7,8にコピー
                "9": [10, 11],          # キーフレーム9→10,11にコピー
                "12": [13, 14],         # キーフレーム12→13,14にコピー
            },
            "ループ": {
                "0": [1, 2],            # キーフレーム0→1,2にコピー
                "3": [4, 5],            # キーフレーム3→4,5にコピー
                "6": [7, 8],            # キーフレーム6→7,8にコピー
                "9": [10, 11],          # キーフレーム9→10,11にコピー
                "12": [13, 14],         # キーフレーム12→13,14にコピー
            }
        }
    }
}

# HTMLキャッシュ関連
_html_cache = {}

def clear_html_cache():
    """HTMLキャッシュをクリアする"""
    global _html_cache
    _html_cache = {}

# ユーティリティ関数
def get_video_modes():
    """利用可能なビデオモードのリストを取得"""
    return list(VIDEO_MODE_SETTINGS.keys())


def get_video_frames(mode_key):
    """モード名から総フレーム数を取得"""
    if mode_key not in VIDEO_MODE_SETTINGS:
        raise ValueError(f"Unknown video mode: {mode_key}")
    return VIDEO_MODE_SETTINGS[mode_key]["frames"]


def get_video_seconds(mode_key):
    """モード名から表示用秒数を取得"""
    if mode_key not in VIDEO_MODE_SETTINGS:
        raise ValueError(f"Unknown video mode: {mode_key}")
    return VIDEO_MODE_SETTINGS[mode_key]["display_seconds"]


def get_important_keyframes(mode_key):
    """重要なキーフレームのインデックスを取得"""
    if mode_key not in VIDEO_MODE_SETTINGS:
        raise ValueError(f"Unknown video mode: {mode_key}")
    return VIDEO_MODE_SETTINGS[mode_key]["important_keyframes"]


def get_total_sections(mode_key):
    """モード名からセクション数を取得"""
    if mode_key not in VIDEO_MODE_SETTINGS:
        raise ValueError(f"Unknown video mode: {mode_key}")
    return VIDEO_MODE_SETTINGS[mode_key]["sections"]


def get_copy_targets(mode, mode_key, keyframe_index):
    """指定キーフレームからのコピー先を取得"""
    if mode_key not in VIDEO_MODE_SETTINGS:
        raise ValueError(f"Unknown video mode: {mode_key}")
    
    if mode not in VIDEO_MODE_SETTINGS[mode_key]["copy_patterns"]:
        return []
    
    str_keyframe_index = str(keyframe_index)
    if str_keyframe_index not in VIDEO_MODE_SETTINGS[mode_key]["copy_patterns"][mode]:
        return []
    
    return VIDEO_MODE_SETTINGS[mode_key]["copy_patterns"][mode][str_keyframe_index]


def get_max_keyframes_count():
    """設定で使用されている最大キーフレーム数を取得"""
    # 動的に計算するメソッド
    max_kf = 0
    for mode_key in VIDEO_MODE_SETTINGS:
        # まず重要なキーフレームの最大値をチェック
        if "important_keyframes" in VIDEO_MODE_SETTINGS[mode_key]:
            important_kfs = VIDEO_MODE_SETTINGS[mode_key]["important_keyframes"]
            if important_kfs and max(important_kfs) > max_kf:
                max_kf = max(important_kfs)
        
        # 次にコピーパターンの中の最大値をチェック
        for mode_type in ["通常", "ループ"]:
            if mode_type not in VIDEO_MODE_SETTINGS[mode_key]["copy_patterns"]:
                continue
                
            for src_kf_str in VIDEO_MODE_SETTINGS[mode_key]["copy_patterns"][mode_type]:
                src_kf = int(src_kf_str)
                if src_kf > max_kf:
                    max_kf = src_kf
                    
                targets = VIDEO_MODE_SETTINGS[mode_key]["copy_patterns"][mode_type][src_kf_str]
                if targets and max(targets) > max_kf:
                    max_kf = max(targets)
    
   # 計算されたセクション数の最大値も考慮
    for mode_key in VIDEO_MODE_SETTINGS:
        if "sections" in VIDEO_MODE_SETTINGS[mode_key]:
            sections = VIDEO_MODE_SETTINGS[mode_key]["sections"]
            if sections > max_kf:
                max_kf = sections
    
    # 固定値として50を設定（または計算値と50の大きい方を使用。100だとUIコンポーネント超過エラー）
    return max(max_kf + 1, 50)  # 0始まりなので+1、かつ最小50
    

def generate_keyframe_guide_html():
    """キーフレームガイドのHTML生成"""
    # キャッシュがあれば使用
    global _html_cache
    if "keyframe_guide" in _html_cache:
        return _html_cache["keyframe_guide"]
    
    html = """
    <div style="margin: 5px 0; padding: 10px; border-radius: 5px; border: 1px solid #ddd; background-color: #f9f9f9;">
        <span style="display: inline-block; margin-bottom: 5px; font-weight: bold;">■ キーフレーム画像設定ガイド:</span>
        <ul style="margin: 0; padding-left: 20px;">
    """
    
    # 各モードの説明を動的に生成
    for length, settings in VIDEO_MODE_SETTINGS.items():
        if length == "6秒" or length == "8秒":
            continue  # 基本モードは説明不要
        
        important_kfs = [kf+1 for kf in settings["important_keyframes"]]  # 1始まりに変換
        html += f'<li><span style="color: #ff3860; font-weight: bold;">{length}</span> モードでは、'
        html += f'<span style="color: #ff3860; font-weight: bold;">キーフレーム画像{", ".join(map(str, important_kfs))}</span> が重要です</li>'
    
    html += """
        <li><span style="color: #ff3860; font-weight: bold;">ループモード</span>では、常に<span style="color: #ff3860; font-weight: bold;">キーフレーム画像1</span>が重要です</li>
        </ul>
        <div style="margin-top: 5px; font-size: 0.9em; color: #666;">※ 重要なキーフレームは赤枠で強調表示されます</div>
    </div>
    """
    
    # キャッシュに保存
    _html_cache["keyframe_guide"] = html
    return html


# 動画モードの追加が容易になるようサポート関数を追加
def add_video_mode(mode_name, frames, sections, display_seconds, important_keyframes, copy_patterns):
    """
    新しい動画モードを設定に追加する関数
    
    Args:
        mode_name: モード名（例: "6秒"）
        frames: フレーム数
        sections: セクション数
        display_seconds: 表示用秒数
        important_keyframes: 重要なキーフレームのインデックス（0始まり）のリスト
        copy_patterns: コピーパターン辞書 {"通常": {...}, "ループ": {...}}
    """
    VIDEO_MODE_SETTINGS[mode_name] = {
        "frames": frames,
        "sections": sections,
        "display_seconds": display_seconds,
        "important_keyframes": important_keyframes,
        "copy_patterns": copy_patterns
    }
    
    # ガイドHTML等のキャッシュをクリア
    clear_html_cache()


def handle_mode_length_change(mode, length, section_number_inputs):
    """モードと動画長の変更時のUI更新処理"""
    # 基本要素のクリア（Image, Final Frame）
    base_updates = [gr.update(value=None) for _ in range(2)]
    
    # キーフレーム画像の更新リスト生成
    keyframe_updates = []
    max_keyframes = get_max_keyframes_count()
    for i in range(max_keyframes):
        keyframe_updates.append(gr.update(value=None, elem_classes=""))
    
    # セクション番号ラベルのリセット
    for i in range(max_keyframes):
        if i < len(section_number_inputs):
            section_number_inputs[i].elem_classes = ""
    
    # 重要なキーフレームの強調表示
    important_kfs = get_important_keyframes(length)
    for idx in important_kfs:
        if idx < len(keyframe_updates):
            keyframe_updates[idx] = gr.update(value=None, elem_classes="highlighted-keyframe")
            if idx < len(section_number_inputs):
                section_number_inputs[idx].elem_classes = "highlighted-label"
    
    # ループモードの場合はキーフレーム1も重要
    if mode == MODE_TYPE_LOOP:
        keyframe_updates[0] = gr.update(value=None, elem_classes="highlighted-keyframe")
        if 0 < len(section_number_inputs):
            section_number_inputs[0].elem_classes = "highlighted-label"
    
    # 動画長の設定
    video_length = get_video_seconds(length)
    
    # 結果を返す
    return base_updates + keyframe_updates + [gr.update(value=video_length)]


def process_keyframe_change(keyframe_idx, img, mode, length, enable_copy=True):
    """キーフレーム画像変更時の処理（汎用版）"""
    if img is None or not enable_copy:
        # 更新なし
        max_keyframes = get_max_keyframes_count()
        return [gr.update() for _ in range(max_keyframes - keyframe_idx - 1)]
    
    # コピー先の取得
    targets = get_copy_targets(mode, length, keyframe_idx)
    if not targets:
        max_keyframes = get_max_keyframes_count()
        return [gr.update() for _ in range(max_keyframes - keyframe_idx - 1)]
    
    # コピー先に対するアップデートを生成
    max_keyframes = get_max_keyframes_count()
    updates = []
    for i in range(keyframe_idx + 1, max_keyframes):
        # ターゲットインデックスへの相対位置を計算
        if (i - keyframe_idx) in targets:
            updates.append(gr.update(value=img))
        else:
            updates.append(gr.update())
    
    return updates


def print_settings_summary(enable_debug=False):
    """設定の概要をコンソールに出力（デバッグ用）"""
    if not enable_debug:
        return
        
    print("\n==== ビデオモード設定の概要 ====")
    for mode_key in VIDEO_MODE_SETTINGS:
        settings = VIDEO_MODE_SETTINGS[mode_key]
        print(f"\nモード: {mode_key}")
        print(f"  フレーム数: {settings['frames']}")
        print(f"  セクション数: {settings['sections']}")
        print(f"  表示秒数: {settings['display_seconds']}")
        print(f"  重要キーフレーム: {settings['important_keyframes']}")
        print("  コピーパターン:")
        for mode_type in settings["copy_patterns"]:
            print(f"    {mode_type}:")
            for src, targets in settings["copy_patterns"][mode_type].items():
                print(f"      キーフレーム{src} → {targets}")
    
    max_kf = get_max_keyframes_count()
    print(f"\n最大キーフレーム数: {max_kf}")
    print("============================\n")


# 将来の拡張用に保持 - 現在は未使用
"""
def calculate_frames_per_section(latent_window_size=9):
    \"""1セクションあたりのフレーム数を計算\"""
    return latent_window_size * 4 - 3


def calculate_sections_from_frames(total_frames, latent_window_size=9):
    \"""フレーム数から必要なセクション数を計算\"""
    frames_per_section = calculate_frames_per_section(latent_window_size)
    return math.ceil(total_frames / frames_per_section)


def calculate_total_frame_count(sections, latent_window_size=9):
    \"""セクション数から総フレーム数を計算\"""
    frames_per_section = calculate_frames_per_section(latent_window_size)
    return sections * frames_per_section


def calculate_total_second_length(frames, fps=30):
    \"""フレーム数から秒数を計算\"""
    return frames / fps
"""
