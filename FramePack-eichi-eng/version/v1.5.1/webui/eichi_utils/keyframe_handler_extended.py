"""
拡張キーフレーム処理モジュール
キーフレーム処理に関する拡張機能を提供します
"""

import gradio as gr
from eichi_utils.video_mode_settings import (
    get_total_sections,
    get_important_keyframes,
    get_video_seconds,
    MODE_TYPE_LOOP
)
from eichi_utils.keyframe_handler import code_to_ui_index, get_max_keyframes_count
from eichi_utils.frame_calculator import calculate_sections_for_mode_and_size

def extended_mode_length_change_handler(mode, length, section_number_inputs, section_row_groups=None, frame_size_setting="1秒 (33フレーム)"):
    """モードと動画長の変更を統一的に処理する関数（セクション行の表示/非表示制御を追加）
    
    Args:
        mode: モード ("通常" or "ループ")
        length: 動画長 ("6秒", "8秒", "10(5x2)秒", "12(4x3)秒", "16(4x4)秒")
        section_number_inputs: セクション番号入力欄のリスト
        section_row_groups: セクション行のUIグループリスト（オプション）
        frame_size_setting: フレームサイズ設定 ("1秒 (33フレーム)" or "0.5秒 (17フレーム)")
        
    Returns:
        更新リスト: 各UI要素の更新情報のリスト
    """
    # 基本要素のクリア（入力画像と終了フレーム）
    updates = [gr.update(value=None) for _ in range(2)]
    
    # すべてのキーフレーム画像をクリア
    section_image_count = get_max_keyframes_count()
    for _ in range(section_image_count):
        updates.append(gr.update(value=None, elem_classes=""))
    
    # セクション番号ラベルをリセット
    for i in range(len(section_number_inputs)):
        section_number_inputs[i].elem_classes = ""
    
    # 重要なキーフレームを強調表示
    important_kfs = get_important_keyframes(length)
    for idx in important_kfs:
        ui_idx = code_to_ui_index(idx)
        update_idx = ui_idx + 1  # 入力画像と終了フレームの2つを考慮
        if update_idx < len(updates):
            updates[update_idx] = gr.update(value=None, elem_classes="highlighted-keyframe")
            if idx < len(section_number_inputs):
                section_number_inputs[idx].elem_classes = "highlighted-label"
    
    # ループモードの場合はキーフレーム0も強調（まだ強調されていない場合）
    if mode == MODE_TYPE_LOOP and 0 not in important_kfs:
        updates[2] = gr.update(value=None, elem_classes="highlighted-keyframe")
        if 0 < len(section_number_inputs):
            section_number_inputs[0].elem_classes = "highlighted-label"
    
    # 動画長の設定
    video_length = get_video_seconds(length)
    
    # 最終的な動画長設定を追加
    updates.append(gr.update(value=video_length))
    
    # セクション行の表示/非表示を制御
    if section_row_groups is not None:
        # 動画モードとフレームサイズから必要なセクション数を計算
        required_sections = calculate_sections_for_mode_and_size(length, frame_size_setting)
        print(f"動画モード '{length}' とフレームサイズ '{frame_size_setting}' で必要なセクション数: {required_sections}")
        
        # 各セクション行の表示/非表示を設定
        row_updates = []
        for i, _ in enumerate(section_row_groups):
            if i < required_sections:
                row_updates.append(gr.update(visible=True))
            else:
                row_updates.append(gr.update(visible=False))
        
        # 更新リストに行の表示/非表示設定を追加
        updates.extend(row_updates)
    
    return updates
