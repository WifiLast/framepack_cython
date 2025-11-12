"""FramePack-eichi LoRA loader module
Provides LoRA application functionality for HunyuanVideo model
基本的なLoRAの読み込みと適用機能を提供
"""

import os
import torch
import logging
import traceback
import safetensors.torch as sf
from typing import Dict, List, Union, Optional, Any, Tuple

# ロギング設定
logger = logging.getLogger("lora_loader")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# LoRAファイルのキャッシュ
_lora_cache = {}

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

def convert_diffusers_lora_to_hunyuan(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    DiffusersフォーマットのLoRAをHunyuan互換形式に変換
    
    Args:
        state_dict: Diffusersフォーマットの状態辞書
    
    Returns:
        Dict[str, torch.Tensor]: Hunyuan互換の状態辞書
    """
    logger.info("DiffusersフォーマットからHunyuanフォーマットへ変換中...")
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
                new_key = f"{diffusers_prefix}.{module_name}.lora_down"
                dim = weight.shape[0]
            elif "lora_up" in key:
                new_key = f"{diffusers_prefix}.{module_name}.lora_up"
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

def load_and_apply_lora(
    model: torch.nn.Module,
    lora_path: str,
    scale: float = 0.8,
    is_diffusers: bool = False
) -> torch.nn.Module:
    """
    LoRAをロードしてモデルに適用する便利関数
    
    Args:
        model: ベースモデル
        lora_path: LoRAファイルへのパス
        scale: LoRAの適用強度 (0.0-1.0)
        is_diffusers: Diffusersフォーマットかどうか
        
    Returns:
        torch.nn.Module: LoRAが適用されたモデル
    """
    logger.info(f"LoRAを読み込み・適用: {lora_path}, スケール: {scale}")
    
    # LoRAの読み込み
    lora_state_dict = load_lora_weights(lora_path)
    
    # 明示的なフォーマット指定（オプション）
    format_type = 'diffusers' if is_diffusers else None
    format_type = format_type or detect_lora_format(lora_state_dict)
    
    # フォーマットに応じて変換
    if format_type == 'diffusers':
        lora_state_dict = convert_diffusers_lora_to_hunyuan(lora_state_dict)
    elif format_type == 'musubi':
        lora_state_dict = check_for_musubi(lora_state_dict)
        
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
                    modified_modules.append(name)
        
        # _lora_appliedフラグを設定（診断用）
        model._lora_applied = True
        model._lora_source = "direct_application"
        
        logger.info(f"LoRA適用完了: {len(modified_modules)} モジュールが修正されました")
        return model
    
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
