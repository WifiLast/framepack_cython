"""
FramePack-eichi LoRA管理モジュール
LoRAの読み込みと管理機能を提供します。
フック方式は廃止され、直接適用方式のみサポートしています。
"""

import os
import torch
import logging
import traceback
from typing import Dict, List, Optional, Union, Tuple, Any

# LoRAローダーをインポート
from .lora_loader import (load_lora_weights, detect_lora_format, convert_diffusers_lora_to_framepack,
                        check_for_musubi, filter_lora_weights_by_important_layers, 
                        filter_lora_weights_by_block_type, prune_lora_weights, PRESET_BLOCKS)

# ロギング設定
logger = logging.getLogger("dynamic_swap_lora")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class DynamicSwapLoRAManager:
    """
    LoRA管理クラス（互換性のために名前は維持）
    注意: フック方式は廃止され、直接適用方式のみサポート
    """
    
    def __init__(self):
        # LoRAの状態を保持
        self.lora_state_dict = None
        # 適用スケール
        self.scale = 0.0
        # LoRAの適用有無フラグ
        self.is_active = False
    
    def load_lora(self, lora_path: str, is_diffusers: bool = False, selective_application: bool = True, pruning: bool = True, pruning_threshold: float = 0.0005, hierarchical_pruning: bool = True, blocks_type: str = "all") -> None:
        """
        LoRAファイルを読み込む
        
        Args:
            lora_path: LoRAファイルへのパス
            is_diffusers: Diffusersフォーマットかどうか
            selective_application: 選択的なLoRA適用を行うかどうか
            pruning: LoRAパラメータのプルーニングを行うかどうか
            pruning_threshold: プルーニングの閾値
            hierarchical_pruning: 階層的プルーニングを使用するか
            blocks_type: 適用するブロックタイプ
        """
        # LoRAの読み込み
        try:
            lora_state_dict = load_lora_weights(lora_path)
            
            # フォーマット変換（必要な場合）
            format_type = detect_lora_format(lora_state_dict)
            logger.info(f"検出されたLoRAフォーマット: {format_type}")

            if format_type == 'diffusers' or is_diffusers:
                lora_state_dict = convert_diffusers_lora_to_framepack(lora_state_dict)
            elif format_type == 'musubi':
                lora_state_dict = check_for_musubi(lora_state_dict)
            
            logger.info(f"元のLoRA: {len(lora_state_dict)} パラメータ")
            
            # ブロックタイプに基づくフィルタリング
            if blocks_type != "all" and lora_state_dict:
                try:
                    logger.info(f"ブロックタイプ '{blocks_type}' に基づくフィルタリングを適用中...")
                    filtered_weights = filter_lora_weights_by_block_type(lora_state_dict, blocks_type)
                    # 空でない場合のみ更新
                    if filtered_weights:  
                        lora_state_dict = filtered_weights
                    else:
                        # 空の場合はLoRAを適用しない
                        logger.warning(f"ブロックタイプ '{blocks_type}' によるフィルタリング後にパラメータが見つからないため、LoRAは無効化されます")
                        self.lora_state_dict = None
                        self.is_active = False
                        return
                except Exception as e:
                    logger.warning(f"ブロックフィルタリング中にエラーが発生しました: {e}")
                    logger.warning(traceback.format_exc())
                    logger.info("ブロックフィルタリングをスキップし、標準的な選択的適用を使用します")
                    
                    # エラー発生時は通常の選択的適用にフォールバック
                    if selective_application:
                        try:
                            filtered_weights = filter_lora_weights_by_important_layers(lora_state_dict)
                            if filtered_weights:
                                lora_state_dict = filtered_weights
                            else:
                                logger.warning("選択的適用によりパラメータが見つからないため、LoRAは無効化されます")
                                self.lora_state_dict = None
                                self.is_active = False
                                return
                        except Exception as e:
                            logger.warning(f"選択的適用中にエラーが発生しました: {e}")
                            logger.warning(traceback.format_exc())
            
            # 標準的な選択的適用（ブロックフィルタリングを使用しない場合）
            elif selective_application and blocks_type == "all" and lora_state_dict:
                try:
                    filtered_weights = filter_lora_weights_by_important_layers(lora_state_dict)
                    # 空でない場合のみ更新
                    if filtered_weights:  
                        lora_state_dict = filtered_weights
                    else:
                        # 空の場合はLoRAを適用しない
                        logger.warning("選択的適用によりパラメータが見つからないため、LoRAは無効化されます")
                        self.lora_state_dict = None
                        self.is_active = False
                        return
                except Exception as e:
                    logger.warning(f"選択的適用中にエラーが発生しました: {e}")
                    logger.warning(traceback.format_exc())
                    logger.info("選択的適用をスキップします")
            
            # LoRAパラメータのプルーニング
            if pruning and lora_state_dict:
                try:
                    # プルーニング実行
                    pruned_weights = prune_lora_weights(lora_state_dict, pruning_threshold)
                    if pruned_weights:  # 空でない場合のみ更新
                        lora_state_dict = pruned_weights
                except Exception as prune_error:
                    logger.warning(f"プルーニング中にエラーが発生しました: {prune_error}")
                    logger.warning(traceback.format_exc())
                    logger.info("プルーニングをスキップします")
            
            # 最終的な状態設定
            try:
                if lora_state_dict and len(lora_state_dict) > 0:
                    self.lora_state_dict = lora_state_dict
                    logger.info(f"LoRA読み込み完了: {len(lora_state_dict)} パラメータ")
                    self.is_active = True
                else:
                    logger.warning("LoRAパラメータが空です。LoRAは有効化されません。")
                    self.lora_state_dict = None
                    self.is_active = False
            except Exception as e:
                logger.error(f"最終状態設定中にエラーが発生しました: {e}")
                logger.error(traceback.format_exc())
                self.lora_state_dict = None
                self.is_active = False
            
        except Exception as e:
            logger.error(f"LoRA読み込みエラー: {e}")
            logger.error(traceback.format_exc())
            self.lora_state_dict = None
            self.is_active = False
            raise
    
    def set_scale(self, scale: float) -> None:
        """
        LoRA適用スケールを設定
        
        Args:
            scale: 適用スケール (0.0-1.0)
        """
        self.scale = max(0.0, min(1.0, scale))
        logger.info(f"LoRA適用スケールを設定: {self.scale}")
    
    def install_hooks(self, model: torch.nn.Module) -> None:
        """
        互換性のための空のメソッド（実際には何も行わない）
        警告メッセージを出力
        
        Args:
            model: モデル
        """
        logger.warning("フック方式は廃止されました。代わりに load_and_apply_lora 関数を使用してください。")
        logger.info("このメソッドは互換性のために残されていますが、実際には何も行いません。")
