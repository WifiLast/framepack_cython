"""
FramePack-eichi DynamicSwap対応LoRAモジュール
DynamicSwapメモリ管理システムと互換性のあるLoRA適用機能を提供します。
拡張機能: ブロック選択とMusubiフォーマット対応
"""

import os
import torch
import logging
import traceback
from typing import Dict, List, Optional, Union, Tuple, Any

# LoRAローダーをインポート
from .lora_loader import (load_lora_weights, detect_lora_format, convert_diffusers_lora_to_hunyuan, 
                        check_for_musubi, filter_lora_weights_by_important_layers, 
                        filter_lora_weights_by_block_type, prune_lora_weights, PRESET_BLOCKS)

# ロギング設定
logger = logging.getLogger("dynamic_swap_lora")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class OptimizedDynamicSwapLoRA:
    """
    メモリ使用量を最適化したDynamicSwapに対応するLoRAクラス
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.original_state = {}
        self.applied_keys = set()
    
    def install(self, lora_weights: Dict[str, torch.Tensor], scale: float = 1.0) -> torch.nn.Module:
        """
        最適化されたDynamicSwapモードでLoRAをインストール
        
        Args:
            lora_weights: LoRA重み
            scale: 適用スケール
            
        Returns:
            torch.nn.Module: LoRAが適用されたモデル
        """
        # 元の状態を最小限だけ保存
        self._save_minimal_original_state(lora_weights)
        
        # LoRAを適用
        self._apply_weights(lora_weights, scale)
        
        logger.info(f"最適化済みDynamicSwapモードでLoRAをインストールしました")
        return self.model
    
    def _save_minimal_original_state(self, lora_weights: Dict[str, torch.Tensor]) -> None:
        """
        変更される部分のみ元の状態を保存
        
        Args:
            lora_weights: LoRA重み
        """
        saved_params = 0
        lora_keys = set(lora_weights.keys())
        
        for name, module in self.model.named_modules():
            if not any(name in key for key in lora_keys):
                continue
                
            # このモジュールがLoRAで変更される場合のみ保存
            if name not in self.original_state:
                # CPUメモリに保存してGPUメモリを節約
                self.original_state[name] = {}
                for param_name, param in module.named_parameters(recurse=False):
                    self.original_state[name][param_name] = param.detach().cpu().clone()
                saved_params += 1
        
        logger.info(f"元の状態を保存: {saved_params} モジュール（最適化済み）")
    
    def _apply_weights(self, lora_weights: Dict[str, torch.Tensor], scale: float) -> None:
        """
        LoRA重みをモデルに適用
        
        Args:
            lora_weights: LoRA重み
            scale: 適用スケール
        """
        with torch.no_grad():
            for key, value in lora_weights.items():
                if ".lora_down" not in key and ".lora_up" not in key:
                    continue
                    
                # モジュールパス、パラメータ名、LoRAタイプを取得
                parts = key.split('.')
                module_path = '.'.join(parts[:-2]) if ".lora_down" in key or ".lora_up" in key else '.'.join(parts[:-1])
                param_name = parts[-2] if ".lora_down" in key or ".lora_up" in key else parts[-1]
                lora_type = "lora_down" if ".lora_down" in key else "lora_up"
                
                if module_path not in self.applied_keys:
                    self.applied_keys.add(module_path)
                
                # モジュールとパラメータを取得（再帰的に探索）
                module = self.model
                for name in module_path.split('.'):
                    if not name:
                        continue
                    try:
                        module = getattr(module, name)
                    except AttributeError:
                        logger.warning(f"モジュール {name} が見つかりません: {module_path}")
                        break
                
                if not hasattr(module, param_name):
                    continue
                    
                # パラメータを取得
                param = getattr(module, param_name)
                if not isinstance(param, torch.nn.Parameter):
                    continue
                    
                # LoRAを適用
                if lora_type == "lora_down" and f"{module_path}.{param_name}.lora_up" in lora_weights:
                    # lora_downとlora_upの両方がある場合のみ適用
                    lora_down = value.to(param.device)
                    lora_up = lora_weights[f"{module_path}.{param_name}.lora_up"].to(param.device)
                    
                    # LoRAの演算
                    delta = torch.matmul(lora_up, lora_down) * scale
                    
                    # 形状を確認して調整
                    if delta.shape != param.shape:
                        logger.warning(f"形状不一致: {delta.shape} != {param.shape}, スキップ: {module_path}.{param_name}")
                        continue
                    
                    # 元のパラメータに適用
                    param.data += delta
                    logger.debug(f"LoRA適用: {module_path}.{param_name}")
    
    def uninstall(self) -> torch.nn.Module:
        """
        LoRAを削除し、元の状態に戻す
        
        Returns:
            torch.nn.Module: 元の状態に戻されたモデル
        """
        with torch.no_grad():
            for module_path, params in self.original_state.items():
                # モジュールを取得（再帰的に探索）
                module = self.model
                for name in module_path.split('.'):
                    if not name:
                        continue
                    try:
                        module = getattr(module, name)
                    except AttributeError:
                        logger.warning(f"モジュール {name} が見つかりません: {module_path}")
                        break
                
                # 元の状態に戻す
                for param_name, original_value in params.items():
                    if hasattr(module, param_name):
                        param = getattr(module, param_name)
                        if isinstance(param, torch.nn.Parameter):
                            # 保存された元の状態がCPUにある場合は、デバイスを合わせる
                            if original_value.device != param.device:
                                original_value = original_value.to(param.device)
                            
                            # 現在のパラメータを元の状態に戻す
                            param.data.copy_(original_value)
                            logger.debug(f"LoRA削除: {module_path}.{param_name}")
        
        # 状態をリセット
        self.original_state = {}
        self.applied_keys.clear()
        
        logger.info("LoRAをアンインストールしました")
        return self.model


class DynamicSwapLoRAManager:
    """
    DynamicSwapと互換性のあるLoRA管理クラス
    """
    
    def __init__(self):
        # LoRAの状態を保持
        self.lora_state_dict = None
        # 適用スケール
        self.scale = 0.0
        # 適用したレイヤーの記録
        self.applied_layers = set()
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
                lora_state_dict = convert_diffusers_lora_to_hunyuan(lora_state_dict)
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
                    # メモリ使用状況のログ出力
                    self.log_memory_usage("プルーニング前")
                    
                    # 階層的プルーニングを適用
                    pruned_weights = self.prune_lora_weights(lora_state_dict, pruning_threshold, hierarchical_pruning)
                    if pruned_weights:  # 空でない場合のみ更新
                        lora_state_dict = pruned_weights
                        
                    # メモリ使用状況のログ出力
                    self.log_memory_usage("プルーニング後")
                    
                    # テンソルキャッシュのクリア
                    self.clear_tensor_cache()
                except Exception as prune_error:
                    logger.warning(f"プルーニング中にエラーが発生しました: {prune_error}")
                    logger.warning(traceback.format_exc())
                    logger.info("プルーニングをスキップします")
            
            # 最終的な状態設定
            try:
                if lora_state_dict and len(lora_state_dict) > 0:
                    self.lora_state_dict = lora_state_dict
                    logger.info(f"LoRA読み込み完了: {len(lora_state_dict)} パラメータ")
                    
                    # 適用記録をクリア
                    self.applied_layers.clear()
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
    
    def apply_to_layer(self, layer_name: str, layer: torch.nn.Module) -> torch.nn.Module:
        """
        単一レイヤーにLoRAを適用
        
        Args:
            layer_name: レイヤー名
            layer: レイヤーモジュール
            
        Returns:
            torch.nn.Module: LoRAが適用されたレイヤー
        """
        if not self.is_active or self.lora_state_dict is None:
            return layer
        
        if layer_name in self.applied_layers:
            logger.debug(f"レイヤーは既にLoRA適用済み: {layer_name}")
            return layer
        
        try:
            with torch.no_grad():
                # レイヤーのパラメータを取得
                for param_name, param in layer.named_parameters():
                    full_param_name = f"{layer_name}.{param_name}"
                    
                    # メモリ使用状況をログ出力（最初のパラメータのみ）
                    if param_name == next(iter(layer.named_parameters()))[0]:
                        self.log_memory_usage(f"レイヤー適用前({layer_name})")
                        
                    # LoRAに対応するキーを探す
                    lora_down_key = f"{full_param_name}.lora_down"
                    lora_up_key = f"{full_param_name}.lora_up"
                    
                    if lora_down_key in self.lora_state_dict and lora_up_key in self.lora_state_dict:
                        # LoRAの重みを取得
                        lora_down = self.lora_state_dict[lora_down_key].to(param.device)
                        lora_up = self.lora_state_dict[lora_up_key].to(param.device)
                        
                        # LoRAの演算
                        delta = torch.matmul(lora_up, lora_down) * self.scale
                        
                        # 形状を確認して調整
                        if delta.shape != param.shape:
                            logger.warning(f"形状不一致: {delta.shape} != {param.shape}, スキップ: {full_param_name}")
                            continue
                        
                        # 元のパラメータに適用
                        param.data += delta
                        logger.debug(f"LoRA適用: {full_param_name}")
            
            # メモリ使用状況をログ出力（最後のパラメータのみ）
            if list(layer.named_parameters()) and param_name == list(layer.named_parameters())[-1][0]:
                self.log_memory_usage(f"レイヤー適用後({layer_name})")
                
            # 適用記録を更新
            self.applied_layers.add(layer_name)
            return layer
            
        except Exception as e:
            logger.error(f"レイヤーへのLoRA適用エラー: {layer_name}, {e}")
            logger.error(traceback.format_exc())
            return layer
    
    def remove_from_layer(self, layer_name: str, layer: torch.nn.Module, original_state: Dict[str, torch.Tensor]) -> torch.nn.Module:
        """
        レイヤーからLoRAの効果を削除（元の状態に戻す）
        
        Args:
            layer_name: レイヤー名
            layer: レイヤーモジュール
            original_state: 元のパラメータ状態
            
        Returns:
            torch.nn.Module: 元の状態に戻されたレイヤー
        """
        if layer_name not in self.applied_layers:
            return layer
        
        try:
            with torch.no_grad():
                # レイヤーのパラメータを元に戻す
                for param_name, param in layer.named_parameters():
                    full_param_name = f"{layer_name}.{param_name}"
                    
                    if full_param_name in original_state:
                        # 保存された元の状態がCPUにある場合は、デバイスを合わせる
                        state_tensor = original_state[full_param_name]
                        if state_tensor.device != param.device:
                            state_tensor = state_tensor.to(param.device)
                        
                        # 現在のパラメータを元の状態に戻す
                        param.data.copy_(state_tensor)
                        logger.debug(f"LoRA効果を削除: {full_param_name}")
            
            # 適用記録を更新
            self.applied_layers.remove(layer_name)
            return layer
            
        except Exception as e:
            logger.error(f"レイヤーからのLoRA削除エラー: {layer_name}, {e}")
            logger.error(traceback.format_exc())
            return layer

    def prune_lora_weights(self, lora_weights: Dict[str, torch.Tensor], threshold: float = 0.005, use_hierarchical: bool = True) -> Dict[str, torch.Tensor]:
        """
        閾値以下の小さなLoRAパラメータを0に設定
        
        Args:
            lora_weights: プルーニング前のLoRA重み
            threshold: プルーニング閾値（絶対値）
            use_hierarchical: 階層的プルーニングを使用するか
            
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
        
        # 階層的プルーニングのための層ごとの閾値設定
        if use_hierarchical:
            # 特に重要な層（視覚効果に大きく影響）は低い閾値を使用
            critical_layers = [
                "transformer_blocks", "single_transformer_blocks", 
                "norm_out", "proj_out", "to_q", "to_k", "to_v", "to_out"
            ]
            # 中程度の重要性の層
            important_layers = ["attn", "norm", "rgb_linear", "conv", "input_blocks", "output_blocks", "single_blocks", "double_blocks"]
            # その他の層（高い閾値＝より積極的な削減）
            
            logger.info(f"階層的プルーニングを使用: 重要度に応じて異なる閾値を適用")
        
        for key, value in lora_weights.items():
            if isinstance(value, torch.Tensor):
                total_params += value.numel()
                
                # 階層的プルーニングの場合、層ごとに異なる閾値を適用
                current_threshold = threshold
                if use_hierarchical:
                    if any(layer in key for layer in critical_layers):
                        # 最重要な層は最も低い閾値を使用（保持率高）
                        current_threshold = 0.00025    # 最重要層（固定値）
                    elif any(layer in key for layer in important_layers):
                        # 重要な層もやや低い閾値を使用
                        current_threshold = 0.0005    # 重要層（固定値）
                    else:
                        # その他の層は標準閾値
                        current_threshold = 0.001     # その他の層（固定値）
                
                # 閾値以下の値をゼロにマスク
                mask = torch.abs(value) >= current_threshold
                pruned_tensor = value * mask
                pruned_params += (value.numel() - mask.sum().item())
                pruned_weights[key] = pruned_tensor
            else:
                pruned_weights[key] = value
        
        # ゼロ除算を確実に回避
        if total_params > 0:  # 0除算を避ける
            prune_percent = (pruned_params / total_params) * 100.0
            logger.info(f"プルーニング: {pruned_params}/{total_params} パラメータ削減 ({prune_percent:.2f}%)")
            
            # メモリ削減量の計算と出力
            tensor_size = 4  # float32 = 4 bytes
            memory_saved_mb = (pruned_params * tensor_size) / (1024 * 1024)  # MB単位
            logger.info(f"推定メモリ削減量: {memory_saved_mb:.2f}MB")
        else:
            logger.warning("プルーニング対象のパラメータが0です。プルーニングをスキップします。")
        
        return pruned_weights

    def log_memory_usage(self, message: str) -> None:
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
    
    def clear_tensor_cache(self) -> None:
        """
        Pytorchのテンソルキャッシュをクリア
        """
        if torch.cuda.is_available():
            try:
                # GPUキャッシュのクリア
                torch.cuda.empty_cache()
                logger.info("GPUテンソルキャッシュをクリアしました")
            except Exception as e:
                logger.warning(f"キャッシュクリア中にエラー: {e}")
    
    def reset(self) -> None:
        """
        LoRA状態をリセット
        """
        self.lora_state_dict = None
        self.scale = 0.0
        self.applied_layers.clear()
        self.is_active = False
        # キャッシュもクリア
        self.clear_tensor_cache()
        logger.info("LoRA状態をリセットしました")

    def install_hooks(self, model: torch.nn.Module) -> None:
        """
        DynamicSwapフックをインストールして動的にLoRAを適用
        
        Args:
            model: DynamicSwapが設定されたモデル
        """
        if not self.is_active or self.lora_state_dict is None:
            logger.info("LoRAが読み込まれていないか非アクティブです。フック設定をスキップします。")
            return
        
        # モデルの元の状態を保存（最適化版）
        original_states = {}
        
        def collect_minimal_original_states(module, module_name):
            # LoRAで変更される予定のパラメータのみを保存
            lora_keys = set(self.lora_state_dict.keys())
            for param_name, param in module.named_parameters():
                full_name = f"{module_name}.{param_name}"
                # このパラメータにLoRAが適用されるかチェック
                lora_down_key = f"{full_name}.lora_down"
                lora_up_key = f"{full_name}.lora_up"
                if lora_down_key in lora_keys and lora_up_key in lora_keys:
                    # CPUメモリに保存してGPUメモリを節約
                    original_states[full_name] = param.data.detach().cpu().clone()
        
        # 元の状態を収集（最適化版）
        saved_modules = 0
        for name, module in model.named_modules():
            if len(list(module.parameters(recurse=False))) > 0:
                collect_minimal_original_states(module, name)
                saved_modules += 1
        
        logger.info(f"最適化済み状態保存: {len(original_states)} パラメータ（{saved_modules}モジュール中）")
        
        # 読み込み前フックを定義（LoRA適用）
        def pre_load_hook(module, hook_name):
            if not self.is_active:
                return
            
            logger.debug(f"読み込み前フック呼び出し: {hook_name}")
            self.apply_to_layer(hook_name, module)
        
        # アンロード後フックを定義（元の状態に戻す）
        def post_unload_hook(module, hook_name):
            if not self.is_active:
                return
            
            if hook_name in self.applied_layers:
                logger.debug(f"アンロード後フック呼び出し: {hook_name}")
                self.remove_from_layer(hook_name, module, original_states)
        
        # DynamicSwapの内部状態を取得
        # 注意: 実際の実装ではDynamicSwapの構造に応じた調整が必要
        for name, module in model.named_modules():
            if hasattr(module, '_hook_register_handle_pre_load'):
                # 既存のフックに加えて、LoRAフックを登録
                old_pre_load = module._hook_register_handle_pre_load
                
                def combined_pre_load(mod):
                    old_pre_load(mod)
                    pre_load_hook(mod, name)
                
                module._hook_register_handle_pre_load = combined_pre_load
            
            if hasattr(module, '_hook_register_handle_post_unload'):
                # 既存のフックに加えて、LoRAフックを登録
                old_post_unload = module._hook_register_handle_post_unload
                
                def combined_post_unload(mod):
                    post_unload_hook(mod, name)
                    old_post_unload(mod)
                
                module._hook_register_handle_post_unload = combined_post_unload
        
        logger.info(f"DynamicSwapフックを設定しました")
