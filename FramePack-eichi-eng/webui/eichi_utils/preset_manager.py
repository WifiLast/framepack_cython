"""
プリセット管理モジュール
endframe_ichi.pyから外出しされたプリセット関連処理を含む
"""

import os
import json
import traceback
from datetime import datetime

def get_presets_folder_path():
    """プリセットフォルダの絶対パスを取得する"""
    # eichi_utils直下からwebuiフォルダに移動
    webui_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(webui_path, 'presets')

def initialize_presets():
    """初期プリセットファイルがない場合に作成する関数"""
    presets_folder = get_presets_folder_path()
    os.makedirs(presets_folder, exist_ok=True)
    preset_file = os.path.join(presets_folder, 'prompt_presets.json')
    
    # デフォルトのプロンプト
    default_prompts = [
        'A character doing some simple body movements.',
        'A character uses expressive hand gestures and body language.',
        'A character walks leisurely with relaxed movements.',
        'A character performs dynamic movements with energy and flowing motion.',
        'A character moves in unexpected ways, with surprising transitions poses.',
    ]
    
    # デフォルト起動時プロンプト
    default_startup_prompt = "A character doing some simple body movements."
    
    # 既存ファイルがあり、正常に読み込める場合は終了
    if os.path.exists(preset_file):
        try:
            with open(preset_file, 'r', encoding='utf-8') as f:
                presets_data = json.load(f)
                
            # 起動時デフォルトがあるか確認
            startup_default_exists = any(preset.get("is_startup_default", False) for preset in presets_data.get("presets", []))
            
            # なければ追加
            if not startup_default_exists:
                presets_data.setdefault("presets", []).append({
                    "name": "起動時デフォルト",
                    "prompt": default_startup_prompt,
                    "timestamp": datetime.now().isoformat(),
                    "is_default": True,
                    "is_startup_default": True
                })
                presets_data["default_startup_prompt"] = default_startup_prompt
                
                with open(preset_file, 'w', encoding='utf-8') as f:
                    json.dump(presets_data, f, ensure_ascii=False, indent=2)
            return
        except:
            # エラーが発生した場合は新規作成
            pass
    
    # 新規作成
    presets_data = {
        "presets": [],
        "default_startup_prompt": default_startup_prompt
    }
    
    # デフォルトのプリセットを追加
    for i, prompt_text in enumerate(default_prompts):
        presets_data["presets"].append({
            "name": f"デフォルト {i+1}: {prompt_text[:20]}...",
            "prompt": prompt_text,
            "timestamp": datetime.now().isoformat(),
            "is_default": True
        })
    
    # 起動時デフォルトプリセットを追加
    presets_data["presets"].append({
        "name": "起動時デフォルト",
        "prompt": default_startup_prompt,
        "timestamp": datetime.now().isoformat(),
        "is_default": True,
        "is_startup_default": True
    })
    
    # 保存
    try:
        with open(preset_file, 'w', encoding='utf-8') as f:
            json.dump(presets_data, f, ensure_ascii=False, indent=2)
    except:
        # 保存に失敗してもエラーは出さない（次回起動時に再試行される）
        pass

def load_presets():
    """プリセットを読み込む関数"""
    presets_folder = get_presets_folder_path()
    os.makedirs(presets_folder, exist_ok=True)
    preset_file = os.path.join(presets_folder, 'prompt_presets.json')
    
    # 初期化関数を呼び出し（初回実行時のみ作成される）
    initialize_presets()
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            with open(preset_file, 'r', encoding='utf-8') as f:
                file_contents = f.read()
                if not file_contents.strip():
                    print(f"読み込み時に空ファイルが検出されました: {preset_file}")
                    # 空ファイルの場合は再初期化を試みる
                    initialize_presets()
                    retry_count += 1
                    continue
                    
                data = json.loads(file_contents)
                print(f"プリセットファイル読み込み成功: {len(data.get('presets', []))}件")
                return data
                
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            # JSONパースエラーの場合はファイルが破損している可能性がある
            print(f"プリセットファイルの形式が不正です: {e}")
            # ファイルをバックアップ
            backup_file = f"{preset_file}.bak.{int(datetime.now().timestamp())}"
            try:
                import shutil
                shutil.copy2(preset_file, backup_file)
                print(f"破損したファイルをバックアップしました: {backup_file}")
            except Exception as backup_error:
                print(f"バックアップ作成エラー: {backup_error}")
            
            # 再初期化
            initialize_presets()
            retry_count += 1
            
        except Exception as e:
            print(f"プリセット読み込みエラー: {e}")
            # エラー発生
            retry_count += 1
    
    # 再試行しても失敗した場合は空のデータを返す
    print("再試行しても読み込みに失敗しました。空のデータを返します。")
    return {"presets": []}

def get_default_startup_prompt():
    """起動時に表示するデフォルトプロンプトを取得する関数"""
    print("起動時デフォルトプロンプト読み込み開始")
    presets_data = load_presets()
    
    # プリセットからデフォルト起動時プロンプトを探す
    for preset in presets_data["presets"]:
        if preset.get("is_startup_default", False):
            startup_prompt = preset["prompt"]
            print(f"起動時デフォルトプロンプトを読み込み: '{startup_prompt[:30]}...' (長さ: {len(startup_prompt)}文字)")
            
            # 重複しているかチェック
            # 例えば「A character」が複数回出てくる場合は重複している可能性がある
            if "A character" in startup_prompt and startup_prompt.count("A character") > 1:
                print("プロンプトに重複が見つかりました。最初のセンテンスのみを使用します。")
                # 最初のセンテンスのみを使用
                sentences = startup_prompt.split(".")
                if len(sentences) > 0:
                    clean_prompt = sentences[0].strip() + "."
                    print(f"正規化されたプロンプト: '{clean_prompt}'")
                    return clean_prompt
                
            return startup_prompt
    
    # 見つからない場合はデフォルト設定を使用
    if "default_startup_prompt" in presets_data:
        default_prompt = presets_data["default_startup_prompt"]
        print(f"デフォルト設定から読み込み: '{default_prompt[:30]}...' (長さ: {len(default_prompt)}文字)")
        
        # 同様に重複チェック
        if "A character" in default_prompt and default_prompt.count("A character") > 1:
            print("デフォルトプロンプトに重複が見つかりました。最初のセンテンスのみを使用します。")
            sentences = default_prompt.split(".")
            if len(sentences) > 0:
                clean_prompt = sentences[0].strip() + "."
                print(f"正規化されたデフォルトプロンプト: '{clean_prompt}'")
                return clean_prompt
                
        return default_prompt
    
    # フォールバックとしてプログラムのデフォルト値を返す
    fallback_prompt = "A character doing some simple body movements."
    print(f"プログラムのデフォルト値を使用: '{fallback_prompt}'")
    return fallback_prompt

def save_preset(name, prompt_text):
    """プリセットを保存する関数"""
    presets_folder = get_presets_folder_path()
    os.makedirs(presets_folder, exist_ok=True)
    preset_file = os.path.join(presets_folder, 'prompt_presets.json')
    
    presets_data = load_presets()
    
    if not name:
        # 名前が空の場合は起動時デフォルトとして保存
        # 既存の起動時デフォルトを探す
        startup_default_exists = False
        for preset in presets_data["presets"]:
            if preset.get("is_startup_default", False):
                # 既存の起動時デフォルトを更新
                preset["prompt"] = prompt_text
                preset["timestamp"] = datetime.now().isoformat()
                startup_default_exists = True
                # 起動時デフォルトを更新
                break
        
        if not startup_default_exists:
            # 見つからない場合は新規作成
            presets_data["presets"].append({
                "name": "起動時デフォルト",
                "prompt": prompt_text,
                "timestamp": datetime.now().isoformat(),
                "is_default": True,
                "is_startup_default": True
            })
            print(f"起動時デフォルトを新規作成: {prompt_text[:50]}...")
        
        # デフォルト設定も更新
        presets_data["default_startup_prompt"] = prompt_text
        
        try:
            # JSON直接書き込み
            with open(preset_file, 'w', encoding='utf-8') as f:
                json.dump(presets_data, f, ensure_ascii=False, indent=2)
                
            return "プリセット '起動時デフォルト' を保存しました"
        except Exception as e:
            print(f"プリセット保存エラー: {e}")
            traceback.print_exc()
            return f"保存エラー: {e}"
    
    # 通常のプリセット保存処理
    # 同名のプリセットがあれば上書き、なければ追加
    preset_exists = False
    for preset in presets_data["presets"]:
        if preset["name"] == name:
            preset["prompt"] = prompt_text
            preset["timestamp"] = datetime.now().isoformat()
            preset_exists = True
            # 既存のプリセットを更新
            break
    
    if not preset_exists:
        presets_data["presets"].append({
            "name": name,
            "prompt": prompt_text,
            "timestamp": datetime.now().isoformat(),
            "is_default": False
        })
        # 新規プリセットを作成
    
    try:
        # JSON直接書き込み
        with open(preset_file, 'w', encoding='utf-8') as f:
            json.dump(presets_data, f, ensure_ascii=False, indent=2)
        
        # ファイル保存成功
        return f"プリセット '{name}' を保存しました"
    except Exception as e:
        print(f"プリセット保存エラー: {e}")
        # エラー発生
        return f"保存エラー: {e}"

def delete_preset(preset_name):
    """プリセットを削除する関数"""
    presets_folder = get_presets_folder_path()
    os.makedirs(presets_folder, exist_ok=True)
    preset_file = os.path.join(presets_folder, 'prompt_presets.json')
    
    if not preset_name:
        return "プリセットを選択してください"
    
    presets_data = load_presets()
    
    # 削除対象のプリセットを確認
    target_preset = None
    for preset in presets_data["presets"]:
        if preset["name"] == preset_name:
            target_preset = preset
            break
    
    if not target_preset:
        return f"プリセット '{preset_name}' が見つかりません"
    
    # デフォルトプリセットは削除できない
    if target_preset.get("is_default", False):
        return f"デフォルトプリセットは削除できません"
    
    # プリセットを削除
    presets_data["presets"] = [p for p in presets_data["presets"] if p["name"] != preset_name]
    
    try:
        with open(preset_file, 'w', encoding='utf-8') as f:
            json.dump(presets_data, f, ensure_ascii=False, indent=2)
        
        return f"プリセット '{preset_name}' を削除しました"
    except Exception as e:
        return f"削除エラー: {e}"