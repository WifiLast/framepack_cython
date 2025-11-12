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
    """モードと動画長の変更を統一的に処理する関数（セクション行の表示/非表示制御とセクション番号の自動設定を追加）
    
    Args:
        mode: モード ("通常" or "ループ")
        length: 動画長 ("6秒", "8秒", "10(5x2)秒", "12(4x3)秒", "16(4x4)秒")
        section_number_inputs: セクション番号入力欄のリスト
        section_row_groups: セクション行のUIグループリスト（オプション）
        frame_size_setting: フレームサイズ設定 ("1秒 (33フレーム)" or "0.5秒 (17フレーム)")
        
    Returns:
        更新リスト: 各UI要素の更新情報のリスト
    """
    # 基本要素の更新（値を保持し、visible状態のみ更新）
    updates = [gr.update(), gr.update()]
    
    # すべてのキーフレーム画像をクリア
    section_image_count = get_max_keyframes_count()
    section_image_updates = []
    for _ in range(section_image_count):
        section_image_updates.append(gr.update(value=None, elem_classes=""))
    
    # 更新リストに画像更新を追加
    updates.extend(section_image_updates)
    
    # セクション番号ラベルをリセット
    for i in range(len(section_number_inputs)):
        section_number_inputs[i].elem_classes = ""
    
    # 動画モードとフレームサイズから必要なセクション数を計算
    required_sections = calculate_sections_for_mode_and_size(length, frame_size_setting)
    print(f"動画モード '{length}' とフレームサイズ '{frame_size_setting}' で必要なセクション数: {required_sections}")
    
    # セクション番号の更新用リスト
    section_number_updates = []
    
    # セクション番号を降順で自動設定 (実際のセクション数に合わせて設定)
    for i in range(len(section_number_inputs)):
        if i < required_sections - 1:  # 実際のセクション数-1まで表示
            # セクション番号をrequired_sections-1から0に設定
            section_number = (required_sections - 1) - i  # 例: 7セクションなら6,5,4,3,2,1,0に
            section_number_updates.append(gr.update(value=section_number))
            # print(f"[デバッグ] セクション番号自動設定: セクションインデックス{i} → 値{section_number}")
        else:
            # 非表示のセクションは変更なし
            section_number_updates.append(gr.update())
    
    # 重要なキーフレームの強調表示を削除
    # キーフレーム画像には赤枠などの強調表示を適用しない
    
    # 動画長の設定
    video_length = get_video_seconds(length)
    
    # 最終的な動画長設定を追加
    updates.append(gr.update(value=video_length))
    
    # セクション番号更新を更新リストに追加
    updates.extend(section_number_updates)
    
    # セクション行の表示/非表示を制御
    if section_row_groups is not None:
        # 各セクション行の表示/非表示を設定
        row_updates = []
        # print(f"[デバッグ] キーフレームハンドラー拡張: セクション行表示判定: required_sections={required_sections}, 表示条件=(i < (required_sections - 1))")
        for i, _ in enumerate(section_row_groups):
            # 実際のセクション数に合わせて表示/非表示を制御
            is_visible = i < (required_sections - 1)  # セクション数が7の場合はセクション番号6～0を表示
            # print(f"[デバッグ] セクション行{i}: 表示={'YES' if is_visible else 'NO'}")
            if is_visible:
                row_updates.append(gr.update(visible=True))
            else:
                row_updates.append(gr.update(visible=False))
        
        # 更新リストに行の表示/非表示設定を追加
        updates.extend(row_updates)
    
    return updates
