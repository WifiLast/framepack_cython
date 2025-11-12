"""
LoRA適用状況の確認を行うユーティリティ
"""

import torch
import logging
import traceback
from collections import Counter
from typing import Dict, Tuple, List, Any, Optional, Set

# ロギング設定
logger = logging.getLogger("lora_check")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def check_lora_applied(model: torch.nn.Module) -> Tuple[int, int]:
    """
    モデルにLoRAが適用されているかチェック
    
    Args:
        model: チェック対象のモデル
        
    Returns:
        Tuple[int, int]: (適用されたパラメータ数, 対象となるパラメータの総数)
    """
    applied_count = 0
    total_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_count += 1
            if hasattr(param, '_lora_applied') and param._lora_applied:
                applied_count += 1
    
    # 適用率のログ出力
    if total_count > 0:
        application_rate = (applied_count / total_count) * 100.0
        logger.info(f"LoRA適用確認: {applied_count}/{total_count} パラメータ ({application_rate:.2f}%)")
    else:
        logger.warning("チェック可能なパラメータが見つかりません")
    
    return applied_count, total_count

def diagnose_lora_application_failure(model: torch.nn.Module, lora_state_dict: Optional[Dict[str, torch.Tensor]] = None) -> str:
    """
    LoRA適用失敗の原因を詳細に診断
    
    Args:
        model: チェック対象のモデル
        lora_state_dict: LoRAの状態辞書（オプション）
        
    Returns:
        str: 診断レポート
    """
    report = "\n==== LoRA適用失敗診断 ====\n"
    
    # モデルのパラメータ構造を調査
    model_params = list(model.named_parameters())
    if not model_params:
        report += "エラー: モデルにパラメータが見つかりません。\n"
        return report
    
    # モデルパラメータの構造を調査
    model_param_stats = {}
    model_keys = []
    for name, param in model_params[:20]:  # 最初の20パラメータをサンプル
        model_keys.append(name)
        if param.requires_grad:
            model_param_stats[name] = {
                "shape": param.shape,
                "has_lora_flag": hasattr(param, '_lora_applied'),
                "lora_applied": hasattr(param, '_lora_applied') and param._lora_applied,
                "dtype": param.dtype,
                "device": param.device
            }
    
    report += f"モデルパラメータサンプル（最初の5つ）: {model_keys[:5]}\n"
    
    # LoRA辞書の調査（オプション）
    if lora_state_dict is not None:
        # LoRAキーの分析
        lora_keys = list(lora_state_dict.keys())
        lora_keys_sample = lora_keys[:min(10, len(lora_keys))]
        
        report += f"\nLoRAキーサンプル（最初の5つ）: {lora_keys_sample[:5]}\n"
        
        # キーのパターン分析
        key_patterns = []
        for key in lora_keys:
            parts = key.split('.')
            if len(parts) > 2:
                pattern = '.'.join(parts[:-2])  # 最後の2要素を除外したパターン
                key_patterns.append(pattern)
        
        # 最も一般的なパターン
        pattern_counter = Counter(key_patterns)
        common_patterns = pattern_counter.most_common(3)
        report += f"\n最も一般的なLoRAキーパターン: {common_patterns}\n"
        
        # LoRAキーからモデルキーへの変換を試行
        potential_matches = []
        for lora_key in lora_keys_sample:
            if ".lora_A" in lora_key or ".lora_B" in lora_key or ".lora_up" in lora_key or ".lora_down" in lora_key:
                # LoRAサフィックスを削除してベースパスを取得
                base_key = lora_key.replace(".lora_A", "").replace(".lora_B", "")
                base_key = base_key.replace(".lora_up", "").replace(".lora_down", "")
                base_key = base_key.replace(".weight", "")
                
                # potentialなモデルキーを生成
                for model_key in model_keys:
                    if base_key in model_key or model_key in base_key:
                        potential_matches.append((lora_key, model_key))
                        break
        
        if potential_matches:
            report += "\n潜在的なキーマッピング（LoRAキー -> モデルキー）:\n"
            for lora_key, model_key in potential_matches[:5]:
                report += f"  {lora_key} -> {model_key}\n"
        else:
            report += "\n警告: LoRAキーとモデルキーの間に潜在的な一致が見つかりません。\n"
            report += "これはキー命名規則の違いによる可能性があります。\n"
    
    # 形状の不一致を検出
    shape_mismatches = []
    if lora_state_dict is not None:
        for lora_key, lora_tensor in lora_state_dict.items():
            for model_key, model_param in model_params:
                if model_key in lora_key or lora_key in model_key:
                    if lora_tensor.shape != model_param.shape:
                        shape_mismatches.append((lora_key, lora_tensor.shape, model_key, model_param.shape))
    
    if shape_mismatches:
        report += "\n形状の不一致（最初の3つ）:\n"
        for lora_key, lora_shape, model_key, model_shape in shape_mismatches[:3]:
            report += f"  LoRA: {lora_key} {lora_shape} != モデル: {model_key} {model_shape}\n"
    
    # _lora_appliedフラグの存在を確認
    lora_flag_count = sum(1 for name, param in model_params if hasattr(param, '_lora_applied'))
    report += f"\n_lora_appliedフラグを持つパラメータ数: {lora_flag_count}/{len(model_params)}\n"
    
    return report

def log_key_mapping_attempts(model: torch.nn.Module, lora_state_dict: Dict[str, torch.Tensor], max_samples: int = 10) -> str:
    """
    LoRAキーとモデルキーのマッピング試行をログに記録
    
    Args:
        model: チェック対象のモデル
        lora_state_dict: LoRAの状態辞書
        max_samples: ログに記録するサンプル数
        
    Returns:
        str: マッピング試行レポート
    """
    try:
        report = "\n==== LoRAキーマッピング診断 ====\n"
        
        # モデルのキーとLoRAキーのサンプルを収集
        model_keys = [name for name, _ in model.named_parameters()]
        lora_keys = list(lora_state_dict.keys())
        
        # LoRAキーのパターンを分析
        lora_up_keys = [k for k in lora_keys if '.lora_up' in k or '.lora_B' in k]
        lora_down_keys = [k for k in lora_keys if '.lora_down' in k or '.lora_A' in k]
        
        if not lora_up_keys or not lora_down_keys:
            report += "警告: 有効なLoRAキーペア（up/down または A/B）が見つかりません。\n"
            return report
        
        # サンプルサイズを制限
        lora_up_samples = lora_up_keys[:min(max_samples, len(lora_up_keys))]
        model_key_samples = model_keys[:min(max_samples * 2, len(model_keys))]
        
        # マッピング試行のシミュレーション
        successful_mappings = []
        failed_mappings = []
        
        for lora_up_key in lora_up_samples:
            # ベースパスを抽出
            base_path = lora_up_key.replace('.lora_up', '').replace('.lora_B', '')
            base_path = base_path.replace('.weight', '')
            
            # 対応するdown/Aキーを探す
            lora_down_key = None
            for down_key in lora_down_keys:
                if base_path in down_key:
                    lora_down_key = down_key
                    break
            
            if lora_down_key is None:
                failed_mappings.append((lora_up_key, "対応するdown/Aキーが見つかりません"))
                continue
            
            # モデルキーとのマッピングを試行
            matched_model_key = None
            for model_key in model_key_samples:
                # 単純なサブストリングマッチ
                if base_path in model_key or model_key in base_path:
                    matched_model_key = model_key
                    break
            
            if matched_model_key:
                # 形状チェック
                param = dict(model.named_parameters())[matched_model_key]
                lora_up = lora_state_dict[lora_up_key]
                lora_down = lora_state_dict[lora_down_key]
                
                # 形状の比較と期待される展開形状の計算
                try:
                    expected_shape = param.shape
                    lora_shape = torch.matmul(lora_up, lora_down).shape
                    
                    if lora_shape == expected_shape:
                        successful_mappings.append((lora_up_key, matched_model_key, f"形状一致: {lora_shape}"))
                    else:
                        failed_mappings.append((lora_up_key, matched_model_key, f"形状不一致: {lora_shape} != {expected_shape}"))
                except Exception as e:
                    failed_mappings.append((lora_up_key, matched_model_key, f"形状計算エラー: {str(e)}"))
            else:
                failed_mappings.append((lora_up_key, "対応するモデルキーが見つかりません"))
        
        # 結果をレポートに追加
        if successful_mappings:
            report += "\n成功したマッピング（最大5つ）:\n"
            for mapping in successful_mappings[:5]:
                report += f"  {mapping[0]} -> {mapping[1]}: {mapping[2]}\n"
        
        if failed_mappings:
            report += "\n失敗したマッピング（最大5つ）:\n"
            for mapping in failed_mappings[:5]:
                if len(mapping) == 3:
                    report += f"  {mapping[0]} -> {mapping[1]}: {mapping[2]}\n"
                else:
                    report += f"  {mapping[0]}: {mapping[1]}\n"
        
        # 成功率の計算
        total_attempts = len(successful_mappings) + len(failed_mappings)
        if total_attempts > 0:
            success_rate = len(successful_mappings) / total_attempts * 100
            report += f"\nマッピング成功率: {len(successful_mappings)}/{total_attempts} ({success_rate:.2f}%)\n"
        
        return report
    except Exception as e:
        return f"キーマッピング診断中にエラーが発生しました: {str(e)}\n{traceback.format_exc()}"

def create_lora_stats_report(model: torch.nn.Module, lora_name: str = "Unknown", lora_state_dict: Optional[Dict[str, torch.Tensor]] = None, applied_params: Optional[int] = None) -> str:
    """
    LoRA適用状況の詳細レポートを生成
    
    Args:
        model: チェック対象のモデル
        lora_name: LoRAの名前または識別子
        lora_state_dict: LoRAの状態辞書（オプション）
        
    Returns:
        str: 適用状況の詳細レポート
    """
    # 適用済みパラメータ数が指定されている場合は再チェックをスキップ
    if applied_params is not None:
        applied_count = applied_params
        # 一致確認のためにtotal_countは必要
        _, total_count = check_lora_applied(model)
    else:
        # 従来のパラメータチェック
        applied_count, total_count = check_lora_applied(model)
    
    if total_count == 0:
        report = f"LoRA適用状況: {lora_name} - パラメータが見つかりません"
        # 詳細診断情報を追加
        if applied_count == 0:
            report += "\n\n" + diagnose_lora_application_failure(model, lora_state_dict)
        return report
    
    application_rate = (applied_count / total_count) * 100.0
    
    # 適用状況のカテゴリ分類
    status_category = "未適用"
    if application_rate > 95:
        status_category = "完全適用"
    elif application_rate > 50:
        status_category = "部分適用"
    elif application_rate > 0:
        status_category = "最小適用"
    
    # 詳細レポートの生成
    report = f"LoRA適用状況: {lora_name}\n"
    report += f"{applied_count}/{total_count} パラメータ ({application_rate:.2f}%)\n"
    report += f"適用評価: {status_category}"
    
    # 適用率が0%の場合、詳細診断情報を追加
    if applied_count == 0 and lora_state_dict is not None:
        try:
            report += "\n\n" + diagnose_lora_application_failure(model, lora_state_dict)
            report += "\n\n" + log_key_mapping_attempts(model, lora_state_dict)
        except Exception as e:
            report += f"\n\n診断中にエラーが発生しました: {str(e)}\n{traceback.format_exc()}"
    
    return report
