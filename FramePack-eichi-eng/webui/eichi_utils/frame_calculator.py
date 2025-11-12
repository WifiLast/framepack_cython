"""
フレームサイズに基づくセクション数計算モジュール
0.5秒/1秒モードに対応したセクション数の計算機能を提供します
"""

import math
from eichi_utils.video_mode_settings import VIDEO_MODE_SETTINGS

def calculate_frames_per_section(latent_window_size=9):
    """1セクションあたりのフレーム数を計算"""
    return latent_window_size * 4 - 3


def calculate_sections_from_frames(total_frames, latent_window_size=9):
    """フレーム数から必要なセクション数を計算"""
    frames_per_section = calculate_frames_per_section(latent_window_size)
    return math.ceil(total_frames / frames_per_section)


def calculate_total_frame_count(sections, latent_window_size=9):
    """セクション数から総フレーム数を計算"""
    frames_per_section = calculate_frames_per_section(latent_window_size)
    return sections * frames_per_section


def calculate_total_second_length(frames, fps=30):
    """フレーム数から秒数を計算"""
    return frames / fps


def calculate_sections_for_mode_and_size(mode_key, frame_size_setting="1秒 (33フレーム)"):
    """動画モードとフレームサイズ設定から必要なセクション数を計算"""
    # 動画モードからフレーム数を取得
    if mode_key not in VIDEO_MODE_SETTINGS:
        return 15  # デフォルト値
    
    total_frames = VIDEO_MODE_SETTINGS[mode_key]["frames"]
    
    # フレームサイズ設定からlatent_window_sizeを判定
    if frame_size_setting == "0.5秒 (17フレーム)":
        latent_window_size = 5  # 0.5秒モード
    else:
        latent_window_size = 9  # 1秒モードがデフォルト
    
    # 必要なセクション数を計算
    required_sections = calculate_sections_from_frames(total_frames, latent_window_size)
    
    # デバッグ情報
    frames_per_section = calculate_frames_per_section(latent_window_size)
    print(f"計算詳細: モード={mode_key}, フレームサイズ={frame_size_setting}, 総フレーム数={total_frames}, セクションあたり={frames_per_section}フレーム, 必要セクション数={required_sections}")
    
    # 結果を返す
    return required_sections
