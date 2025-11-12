"""
FramePack-eichi LoRA診断ヘルパーモジュール
LoRAの適用状況を診断するための機能を提供します。
最小限の診断機能のみを実装。
"""

import torch
import logging
from typing import Dict, List, Optional, Any, Union, Tuple

# ロギング設定
logger = logging.getLogger("lora_check_helper")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def check_lora_applied(model: torch.nn.Module) -> Tuple[bool, str]:
    """
    モデルにLoRAが適用されているかをチェック
    
    Args:
        model: チェック対象のモデル
        
    Returns:
        Tuple[bool, str]: 適用されているかどうかとその方法
    """
    # _lora_appliedフラグの確認
    has_flag = hasattr(model, '_lora_applied') and model._lora_applied
    source = getattr(model, '_lora_source', 'unknown') if has_flag else 'none'
    
    return has_flag, source

def diagnose_lora_application_failure(model: torch.nn.Module, lora_state_dict: Dict[str, torch.Tensor]) -> str:
    """
    LoRA適用に失敗した原因を診断
    
    Args:
        model: 診断対象のモデル
        lora_state_dict: LoRAの状態辞書
        
    Returns:
        str: 診断結果
    """
    # フラグチェック
    has_flag, source = check_lora_applied(model)
    if has_flag:
        return f"LoRAフラグは正常に設定されています（適用方法: {source}）"
    
    # LoRAキーの存在チェック
    lora_keys = [k for k in lora_state_dict.keys() if '.lora_down' in k or '.lora_up' in k]
    if len(lora_keys) == 0:
        return "LoRA状態辞書に有効なLoRAキーが見つかりません"
    
    # モデルパラメータとLoRAキーの対応チェック
    model_params = dict(model.named_parameters())
    matching_count = 0
    
    for lora_key in lora_keys:
        if '.lora_down' in lora_key:
            param_name = lora_key.replace('.lora_down', '')
            if param_name in model_params:
                matching_count += 1
    
    if matching_count == 0:
        return "モデルパラメータとLoRAキーが一致しません。モデル構造の違いや名前空間の不一致が考えられます。"
    
    # その他の一般的な問題
    return "不明な原因でLoRAの適用に失敗しました。詳細なログを確認してください。"

def create_lora_stats_report(model: torch.nn.Module, lora_name: str = "", lora_state_dict: Optional[Dict[str, torch.Tensor]] = None) -> str:
    """
    LoRA適用状況の統計レポートを生成
    
    Args:
        model: 診断対象のモデル
        lora_name: LoRAの名前
        lora_state_dict: LoRAの状態辞書
        
    Returns:
        str: 診断レポート
    """
    # LoRA適用チェック
    has_lora, source = check_lora_applied(model)
    
    report = [f"--- LoRA適用状況レポート: {lora_name} ---"]
    report.append(f"LoRA適用フラグ: {'有効' if has_lora else '無効'}")
    report.append(f"適用方法: {source}")
    
    # LoRA辞書のチェック
    if lora_state_dict is not None:
        total_params = sum(v.numel() for k, v in lora_state_dict.items() if isinstance(v, torch.Tensor) and ('lora_down' in k or 'lora_up' in k))
        down_keys = [k for k in lora_state_dict.keys() if 'lora_down' in k]
        up_keys = [k for k in lora_state_dict.keys() if 'lora_up' in k]
        
        report.append(f"LoRA辞書: {len(lora_state_dict)} エントリ")
        report.append(f"lora_downキー: {len(down_keys)}")
        report.append(f"lora_upキー: {len(up_keys)}")
        report.append(f"合計パラメータ数: {total_params}")
    
    # 適用に失敗した場合の診断
    if not has_lora and lora_state_dict is not None:
        diagnosis = diagnose_lora_application_failure(model, lora_state_dict)
        report.append(f"問題診断: {diagnosis}")
    
    return "\n".join(report)
