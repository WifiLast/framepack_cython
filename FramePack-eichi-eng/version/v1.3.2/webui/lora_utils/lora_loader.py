"""FramePack-eichi LoRA loader module
Provides LoRA application functionality for HunyuanVideo model
Added memory usage optimization features and selective block loading
"""

import os
import torch
import logging
import traceback
import safetensors.torch as sf
from typing import Dict, List, Union, Optional, Any, Tuple
from time import time

# ロギング設定
logger = logging.getLogger("lora_loader")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ブロック選択の定義
BSINGLE = "single_blocks"
BDOUBLE = "double_blocks"
PRESET_BLOCKS = { # Name = single / double, accepted layers.
    "all": (None, None),  # すべてのブロックを対象
    "single_blocks": (BSINGLE, None),  # すべてのsingle_blocks
    "double_blocks": (BDOUBLE, None),  # すべてのdouble_blocks
    "db0-9": (BDOUBLE, list(range(0, 10))),  # double_blocksの0-9
    "db10-19": (BDOUBLE, list(range(10, 20))),  # double_blocksの10-19
    "sb0-9": (BSINGLE, list(range(0, 10))),  # single_blocksの0-9
    "sb10-19": (BSINGLE, list(range(10, 20))),  # single_blocksの10-19
    "important": (None, None)  # 重要なレイヤーのみ（別途フィルタリング）
}

# LoRAファイルのキャッシュ
_lora_cache = {}

def convert_diffusers_lora_to_hunyuan(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    後方互換性のための旧関数名を維持
    
    Args:
        state_dict: Diffusersフォーマットの状態辞書
    
    Returns:
        Dict[str, torch.Tensor]: FramePack内部形式の状態辞書
    """
    logger.info("後方互換性のための旧関数を使用")
    return convert_diffusers_lora_to_framepack(state_dict)

def load_lora_weights(lora_path: str) -> Dict[str, torch.Tensor]:
    """
    LoRAファイルから重みを読み込む（キャッシュ機能付き）
    
    Args:
        lora_path: LoRAファイルへのパス (.safetensors, .pt, .bin)
    
    Returns:
        Dict[str, torch.Tensor]: LoRAの状態辞書
    """
    global _lora_cache
    
    # キャッシュチェック
    if lora_path in _lora_cache:
        logger.info(f"キャッシュからLoRAを読み込み中: {os.path.basename(lora_path)}")
        return _lora_cache[lora_path]
    
    logger.info(f"LoRAファイルを読み込み中: {lora_path}")
    
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRAファイルが見つかりません: {lora_path}")
    
    _, ext = os.path.splitext(lora_path.lower())
    
    try:
        if ext == '.safetensors':
            state_dict = sf.load_file(lora_path)
            logger.info(f"safetensorsフォーマットからロード完了")
        else:  # .pt, .bin
            state_dict = torch.load(lora_path, map_location='cpu')
            if isinstance(state_dict, torch.nn.Module):
                state_dict = state_dict.state_dict()
            logger.info(f"Torchフォーマットからロード完了")
        
        # キーの種類を確認してフォーマットを自動判定
        format_type = detect_lora_format(state_dict)
        logger.info(f"検出されたLoRAフォーマット: {format_type}")
        
        # キャッシュに保存
        _lora_cache[lora_path] = state_dict
        
        return state_dict
    except Exception as e:
        logger.error(f"LoRAファイル読み込みエラー: {e}")
        logger.error(traceback.format_exc())
        raise

def detect_lora_format(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    状態辞書からLoRAフォーマットを検出する
    
    Args:
        state_dict: LoRAの状態辞書
    
    Returns:
        str: フォーマットタイプ ('diffusers', 'hunyuan', 'kohya', 'musubi', 'unknown')
    """
    # キーのサンプルを取得
    keys = list(state_dict.keys())
    sample_keys = keys[:min(10, len(keys))]
    
    # Diffusersフォーマットの検出
    diffusers_pattern = any(['lora_A' in k or 'lora_B' in k for k in sample_keys])
    
    # Hunyuanフォーマットの検出
    hunyuan_pattern = any(['hunyuan_video' in k.lower() for k in sample_keys])
    
    # Kohyaフォーマットの検出
    kohya_pattern = any(['lora_down' in k or 'lora_up' in k for k in sample_keys])
    
    # Musubiフォーマットの検出
    musubi_pattern = any(['lora_unet_' in k and ('alpha' in k or 'lora_down' in k or 'lora_up' in k) for k in sample_keys])
    
    if musubi_pattern:
        return 'musubi'
    elif diffusers_pattern:
        return 'diffusers'
    elif hunyuan_pattern:
        return 'hunyuan'
    elif kohya_pattern:
        return 'kohya'
    else:
        return 'unknown'

def convert_diffusers_lora_to_framepack(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    DiffusersフォーマットのLoRAをFramePack内部形式に変換
    
    Args:
        state_dict: Diffusersフォーマットの状態辞書
    
    Returns:
        Dict[str, torch.Tensor]: FramePack内部形式の状態辞書
    """
    logger.info("DiffusersフォーマットからFramePack内部フォーマットへ変換中...")
    converted_dict = {}
    
    # フォーマット変換ロジック
    for key, value in state_dict.items():
        if 'lora_A' in key:
            new_key = key.replace('lora_A', 'lora_down')
            converted_dict[new_key] = value
        elif 'lora_B' in key:
            new_key = key.replace('lora_B', 'lora_up')
            converted_dict[new_key] = value
        else:
            converted_dict[key] = value
    
    logger.info(f"変換完了: {len(converted_dict)} パラメータ")
    return converted_dict

def check_for_musubi(lora_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Musubiフォーマットを検出して変換
    
    Args:
        lora_weights: LoRA重み
    
    Returns:
        Dict[str, torch.Tensor]: 変換後のLoRA重み
    """
    prefix = "lora_unet_"
    musubi = False
    lora_alphas = {}
    
    # Musubiフォーマット検出
    for key, value in lora_weights.items():
        if key.startswith(prefix):
            lora_name = key.split(".", 1)[0]
            if lora_name not in lora_alphas and "alpha" in key:
                lora_alphas[lora_name] = value
                musubi = True
    
    if not musubi:
        return lora_weights
        
    logger.info("Musubiチューナーフォーマットのロード中...")
    converted_lora = {}
    
    for key, weight in lora_weights.items():
        if key.startswith(prefix):
            if "alpha" in key:
                continue
            lora_name = key.split(".", 1)[0]
            module_name = lora_name[len(prefix):]  # remove "lora_unet_"
            module_name = module_name.replace("_", ".")  # replace "_" with "."
            module_name = module_name.replace("double.blocks.", "double_blocks.")  # fix double blocks
            module_name = module_name.replace("single.blocks.", "single_blocks.")  # fix single blocks
            module_name = module_name.replace("img.", "img_")  # fix img
            module_name = module_name.replace("txt.", "txt_")  # fix txt
            module_name = module_name.replace("attn.", "attn_")  # fix attn
            diffusers_prefix = "diffusion_model"
            
            if "lora_down" in key:
                new_key = f"{diffusers_prefix}.{module_name}.lora_A.weight"
                dim = weight.shape[0]
            elif "lora_up" in key:
                new_key = f"{diffusers_prefix}.{module_name}.lora_B.weight"
                dim = weight.shape[1]
            else:
                logger.info(f"予期しないキー: {key} (Musubiフォーマット)")
                continue
                
            # スケーリング
            if lora_name in lora_alphas:
                scale = lora_alphas[lora_name] / dim
                scale = scale.sqrt()
                weight = weight * scale
            else:
                logger.info(f"alpha情報が見つかりません: {lora_name}")
                
            converted_lora[new_key] = weight
    
    logger.info(f"Musubi変換完了: {len(converted_lora)} パラメータ")
    return converted_lora

def filter_lora_weights_by_block_type(lora_weights: Dict[str, torch.Tensor], blocks_type: str = "all") -> Dict[str, torch.Tensor]:
    """
    ブロックタイプに基づいてLoRA重みをフィルタリング
    
    Args:
        lora_weights: フィルタリング前のLoRA重み
        blocks_type: フィルタリングするブロックタイプ
        
    Returns:
        Dict[str, torch.Tensor]: フィルタリング後のLoRA重み
    """
    # パラメータが空の場合はそのまま返す
    if not lora_weights:
        logger.warning("引数が空です。フィルタリングをスキップします。")
        return lora_weights
    
    # すべてのブロックを対象とする場合はフィルタリングなし
    if blocks_type == "all":
        return lora_weights
    
    # 特定のブロックタイプに対するフィルタリング
    filtered_weights = {}
    total_keys = len(lora_weights)
    kept_keys = 0
    
    # プリセットの取得
    base_name, base_layer = PRESET_BLOCKS.get(blocks_type, (None, None))
    
    # 重要なレイヤーのみのフィルタリング
    if blocks_type == "important":
        return filter_lora_weights_by_important_layers(lora_weights)
    
    # ブロックタイプに基づくフィルタリング
    for key, value in lora_weights.items():
        # キーを変換して処理
        parts = key.split('.')
        
        if base_name is None:
            # フィルタリングなし
            filtered_weights[key] = value
            kept_keys += 1
            continue
            
        # ブロック名検出
        if base_name in key:
            if base_layer is None:
                # 指定ブロックタイプのすべてのレイヤー
                filtered_weights[key] = value
                kept_keys += 1
            else:
                # 指定ブロックタイプの特定インデックスレイヤーのみ
                try:
                    # インデックス抽出
                    base_split = key.split('.')
                    block_index = -1
                    for i, part in enumerate(base_split):
                        if part == base_name.split('.')[-1] and i+1 < len(base_split):
                            try:
                                block_index = int(base_split[i+1])
                                break
                            except ValueError:
                                continue
                    
                    if block_index in base_layer:
                        filtered_weights[key] = value
                        kept_keys += 1
                except Exception as e:
                    logger.warning(f"レイヤーインデックス抽出エラー: {key}, {e}")
    
    # フィルタリング後にパラメータが0の場合は、空のセットを返し、LoRA適用を中止
    if not filtered_weights:
        logger.warning(f"ブロックタイプ '{blocks_type}' でのフィルタリング後にパラメータが0になりました。LoRAは適用されません。")
        return {}
        
    # パラメータの削減率を計算
    if total_keys > 0:
        filter_percent = (kept_keys / total_keys) * 100.0
        logger.info(f"ブロックタイプ '{blocks_type}' による選択的適用: {kept_keys}/{total_keys} パラメータ保持 ({filter_percent:.2f}%)")
    else:
        logger.warning("全パラメータ数が0です")
        
    return filtered_weights

def filter_lora_weights_by_important_layers(lora_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    重要な層のみにLoRA重みをフィルタリング
    
    Args:
        lora_weights: フィルタリング前のLoRA重み
        
    Returns:
        Dict[str, torch.Tensor]: フィルタリング後のLoRA重み
    """
    # パラメータが空の場合はそのまま返す
    if not lora_weights:
        logger.warning("引数が空です。フィルタリングをスキップします。")
        return lora_weights
        
    important_layers = [
        # 変換ブロック全体（より広範なカバレッジ）
        "transformer_blocks", "single_blocks", "double_blocks", "single_transformer_blocks",
        
        # 注意機構と正規化層（視覚効果に重要）
        "attn", "norm", "rgb_linear",
        
        # 出力・入力層（必須）
        "norm_out", "proj_out", "input_blocks", "output_blocks",
        
        # 一般的なLoRAターゲット層
        "conv", "to_q", "to_k", "to_v", "to_out"
    ]
    
    filtered_weights = {}
    total_keys = len(lora_weights)
    kept_keys = 0
    
    for key, value in lora_weights.items():
        # キーからモジュールパスを抽出
        parts = key.split('.')
        if len(parts) >= 2:
            module_path = f"{parts[0]}.{parts[1]}" if len(parts) > 1 else parts[0]
            
            # 重要層リストと照合
            if any(layer in module_path for layer in important_layers) or any(layer in key for layer in important_layers):
                filtered_weights[key] = value
                kept_keys += 1
    
    # フィルタリング後にパラメータが0の場合は、空のセットを返し、LoRA適用を中止
    if not filtered_weights:
        logger.warning("選択的適用後にパラメータが0になりました。LoRAは適用されません。")
        return {}
        
    # パラメータの削減率を計算
    if total_keys > 0:
        filter_percent = (kept_keys / total_keys) * 100.0
        logger.info(f"選択的適用: {kept_keys}/{total_keys} パラメータ保持 ({filter_percent:.2f}%)")
    else:
        logger.warning("全パラメータ数が0です")
        
    return filtered_weights

def prune_lora_weights(lora_weights: Dict[str, torch.Tensor], threshold: float = 0.0005) -> Dict[str, torch.Tensor]:
    """
    閾値以下の小さなLoRAパラメータを0に設定
    
    Args:
        lora_weights: プルーニング前のLoRA重み
        threshold: プルーニング閾値（絶対値）
        
    Returns:
        Dict[str, torch.Tensor]: プルーニング後のLoRA重み
    """
    # パラメータが空の場合はそのまま返す
    if not lora_weights:
        logger.warning("引数が空です。プルーニングをスキップします。")
        return lora_weights
        
    pruned_weights = {}
    total_params = 0
    pruned_params = 0
    
    for key, value in lora_weights.items():
        if isinstance(value, torch.Tensor):
            total_params += value.numel()
            # 閾値以下の値をゼロにマスク
            mask = torch.abs(value) >= threshold
            pruned_tensor = value * mask
            pruned_params += (value.numel() - mask.sum().item())
            pruned_weights[key] = pruned_tensor
        else:
            pruned_weights[key] = value
    
    # ゼロ除算を確実に回避
    if total_params > 0:  # 0除算を避ける
        prune_percent = (pruned_params / total_params) * 100.0
        logger.info(f"プルーニング: {pruned_params}/{total_params} パラメータ削減 ({prune_percent:.2f}%)")
    else:
        logger.warning("プルーニング対象のパラメータが0です。プルーニングをスキップします。")
    
    return pruned_weights

def log_memory_usage(message: str) -> None:
    """
    現在のGPUメモリ使用状況をログに出力
    
    Args:
        message: ログメッセージのプレフィックス
    """
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"{message}: GPU使用メモリ {allocated:.2f}GB (予約: {reserved:.2f}GB)")
        except Exception as e:
            logger.warning(f"メモリ使用状況の取得に失敗: {e}")
            
def detailed_pruning_log(original_weights, pruned_weights):
    """
    プルーニング前後の状態を詳細に分析してログ出力
    
    Args:
        original_weights: プルーニング前の重み
        pruned_weights: プルーニング後の重み
    """
    total_elements = sum(t.numel() for t in original_weights.values() if isinstance(t, torch.Tensor))
    non_zero_after = sum((t != 0).sum().item() for t in pruned_weights.values() if isinstance(t, torch.Tensor))
    
    memory_before = sum(t.element_size() * t.numel() for t in original_weights.values() if isinstance(t, torch.Tensor)) / 1024**2
    memory_after = sum(t.element_size() * t.numel() for t in pruned_weights.values() if isinstance(t, torch.Tensor)) / 1024**2
    
    logger.info(f"プルーニング詳細: 非ゼロ要素 {non_zero_after}/{total_elements} ({non_zero_after/total_elements*100:.2f}%)")
    logger.info(f"メモリ削減: {memory_before:.2f}MB → {memory_after:.2f}MB (節約: {memory_before-memory_after:.2f}MB)")

def apply_lora_to_model(
    model: torch.nn.Module,
    lora_state_dict: Dict[str, torch.Tensor],
    scale: float = 0.8,
    format_type: Optional[str] = None
) -> Tuple[torch.nn.Module, int]:
    """
    モデルにLoRAを適用する
    
    Args:
        model: ベースモデル
        lora_state_dict: LoRAの状態辞書
        scale: LoRAの適用強度 (0.0-1.0)
        format_type: 明示的なフォーマットタイプ ('diffusers', 'hunyuan', 'kohya', None)
        
    Returns:
        torch.nn.Module: LoRAが適用されたモデル
    """
    logger.info(f"LoRAをモデルに適用中 (スケール係数: {scale})")
    
    # フォーマット自動検出（指定されていない場合）
    if format_type is None:
        format_type = detect_lora_format(lora_state_dict)
    
    # フォーマットに応じて変換
    if format_type == 'diffusers':
        lora_state_dict = convert_diffusers_lora_to_framepack(lora_state_dict)
    elif format_type == 'musubi':
        lora_state_dict = check_for_musubi(lora_state_dict)
    
    # 適用率カウンタの追加
    total_params = sum(1 for key in lora_state_dict if ".lora_down" in key)
    applied_params = 0
    
    # モデルの現在の状態を保存
    original_state = {}
    modified_modules = []
    
    try:
        # LoRAの適用
        with torch.no_grad():
            for name, param in model.named_parameters():
                # LoRAに対応するキーを探す
                lora_down_key = f"{name}.lora_down"
                lora_up_key = f"{name}.lora_up"
                
                if lora_down_key in lora_state_dict and lora_up_key in lora_state_dict:
                    # 元の値を保存
                    original_state[name] = param.data.clone()
                    
                    # LoRAの重みを取得
                    lora_down = lora_state_dict[lora_down_key].to(param.device)
                    lora_up = lora_state_dict[lora_up_key].to(param.device)
                    
                    # LoRAの演算
                    delta = torch.matmul(lora_up, lora_down) * scale
                    
                    # 形状を確認して調整
                    if delta.shape != param.shape:
                        logger.warning(f"形状不一致: {delta.shape} != {param.shape}, スキップ: {name}")
                        continue
                    
                    # 元のパラメータに適用
                    param.data += delta
                    # LoRA適用フラグを設定（チェック用）
                    param._lora_applied = True
                    modified_modules.append(name)
                    applied_params += 1
        
        # 適用率の出力
        if total_params > 0:
            application_rate = (applied_params / total_params) * 100.0
            logger.info(f"LoRA適用状況: {applied_params}/{total_params} パラメータ ({application_rate:.2f}%)")
        else:
            logger.warning("LoRAパラメータが見つかりません。適用率は0%です。")
        
        # 適用率が0%の場合は詳細診断情報を出力
        if applied_params == 0:
            try:
                from .lora_check_helper import diagnose_lora_application_failure, log_key_mapping_attempts
                diagnosis = diagnose_lora_application_failure(model, lora_state_dict)
                mapping_report = log_key_mapping_attempts(model, lora_state_dict)
                logger.warning("LoRA適用に失敗しました。詳細診断情報:")
                logger.warning(diagnosis)
                logger.warning(mapping_report)
            except Exception as e:
                logger.error(f"診断情報出力中にエラーが発生: {e}")
            
        logger.info(f"LoRA適用完了: {len(modified_modules)} モジュールが修正されました")
        return model, applied_params
    
    except Exception as e:
        # エラー時には元の状態に戻す
        logger.error(f"LoRA適用エラー: {e}")
        logger.error(traceback.format_exc())
        
        with torch.no_grad():
            for name, original_data in original_state.items():
                param = dict(model.named_parameters())[name]
                param.data.copy_(original_data)
        
        logger.info("エラーにより元の状態に復元しました")
        raise

def load_and_apply_lora(
    model: torch.nn.Module,
    lora_path: str,
    scale: float = 0.8,
    is_diffusers: bool = False,
    selective_application: bool = True,
    pruning: bool = True,
    pruning_threshold: float = 0.0005,
    blocks_type: str = "all",
    return_applied_count: bool = False
) -> Union[torch.nn.Module, Tuple[torch.nn.Module, int]]:
    """
    LoRAをロードしてモデルに適用する便利関数
    
    Args:
        model: ベースモデル
        lora_path: LoRAファイルへのパス
        scale: LoRAの適用強度 (0.0-1.0)
        is_diffusers: Diffusersフォーマットかどうか
        selective_application: 選択的なLoRA適用を行うかどうか
        pruning: LoRAパラメータのプルーニングを行うかどうか
        pruning_threshold: プルーニングの閾値
        blocks_type: フィルタリングするブロックタイプ
        
    Returns:
        torch.nn.Module: LoRAが適用されたモデル
    """
    logger.info(f"LoRAを読み込み・適用: {lora_path}, スケール: {scale}, ブロックタイプ: {blocks_type}")
    
    # LoRAの読み込み
    lora_state_dict = load_lora_weights(lora_path)
    
    # 明示的なフォーマット指定（オプション）
    format_type = 'diffusers' if is_diffusers else None
    format_type = format_type or detect_lora_format(lora_state_dict)
    
    # フォーマットに応じて変換
    if format_type == 'diffusers':
        lora_state_dict = convert_diffusers_lora_to_framepack(lora_state_dict)
    elif format_type == 'musubi':
        lora_state_dict = check_for_musubi(lora_state_dict)
        
    logger.info(f"元のLoRA: {len(lora_state_dict)} パラメータ")
    
    # ブロックタイプに基づくフィルタリング
    if blocks_type != "all" and lora_state_dict:
        try:
            filtered_state_dict = filter_lora_weights_by_block_type(lora_state_dict, blocks_type)
            if filtered_state_dict:  # 空でない場合のみ更新
                lora_state_dict = filtered_state_dict
        except Exception as e:
            logger.warning(f"ブロックタイプによるフィルタリング中にエラーが発生しました: {e}")
            logger.warning(traceback.format_exc())
            logger.info("ブロックフィルタリングをスキップし、標準的な選択的適用を使用します")
            
            # 標準的な選択的適用にフォールバック
            if selective_application:
                try:
                    filtered_state_dict = filter_lora_weights_by_important_layers(lora_state_dict)
                    if filtered_state_dict:  # 空でない場合のみ更新
                        lora_state_dict = filtered_state_dict
                except Exception as e:
                    logger.warning(f"選択的適用中にエラーが発生しました: {e}")
                    logger.warning(traceback.format_exc())
                    logger.info("選択的適用をスキップし、全パラメータを使用します")
        
    # フィルタリングなしの場合の選択的適用
    elif selective_application and blocks_type == "all" and lora_state_dict:
        try:
            filtered_state_dict = filter_lora_weights_by_important_layers(lora_state_dict)
            if filtered_state_dict:  # 空でない場合のみ更新
                lora_state_dict = filtered_state_dict
        except Exception as e:
            logger.warning(f"選択的適用中にエラーが発生しました: {e}")
            logger.warning(traceback.format_exc())
            logger.info("選択的適用をスキップし、全パラメータを使用します")
    
    # LoRAパラメータのプルーニング
    if pruning and lora_state_dict:
        try:
            log_memory_usage("プルーニング前")
            start_time = time()
            pruned_state_dict = prune_lora_weights(lora_state_dict, pruning_threshold)
            if pruned_state_dict:  # 空でない場合のみ更新
                # 詳細ログ出力
                detailed_pruning_log(lora_state_dict, pruned_state_dict)
                lora_state_dict = pruned_state_dict
                logger.info(f"プルーニング処理時間: {time() - start_time:.2f}秒")
                log_memory_usage("プルーニング後")
        except Exception as e:
            logger.warning(f"プルーニング中にエラーが発生しました: {e}")
            logger.warning(traceback.format_exc())
            logger.info("プルーニングをスキップします")
    
    # モデルに適用
    applied_params = 0
    if lora_state_dict:
        model, applied_params = apply_lora_to_model(model, lora_state_dict, scale)
    else:
        logger.warning("LoRAパラメータが空です。モデルに適用しません。")
    
    # 適用パラメータ数を返すかどうか
    if return_applied_count:
        return model, applied_params
    else:
        return model
