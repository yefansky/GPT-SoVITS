import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import gradio as gr

class WebUIConfigSaver:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance
    
    def _init(self):
        self.config_path = Path("webui_config.json")
        self.config: Dict[str, Any] = {}
        self._dirty_keys = set()
        self._save_lock = threading.Lock()
        self._load_config()
        self._start_background_saver()

    def _load_config(self):
        try:
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
        except Exception as e:
            print(f"Failed to load config: {e}")
            self.config = {}

    def _start_background_saver(self):
        def save_loop():
            while True:
                time.sleep(1)  # 每1秒检查一次保存
                self._save_if_dirty()

        saver_thread = threading.Thread(target=save_loop, daemon=True)
        saver_thread.start()

    def _save_if_dirty(self):
        with self._save_lock:
            if not self._dirty_keys:
                return
            
            # 1. 首先尝试读取当前配置，不存在则使用内存配置作为基础
            current_config = self.config.copy()  # 使用内存配置作为基础
            
            if self.config_path.exists():
                try:
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        file_config = json.load(f)
                        # 合并文件配置到当前配置(文件配置优先)
                        current_config.update(file_config)
                except Exception as e:
                    print(f"Warning: Failed to read current config: {e}")
                    # 继续使用内存配置

            # 2. 应用最新的变更(确保脏数据覆盖合并结果)
            for key in self._dirty_keys:
                current_config[key] = self.config[key]

            # 3. 写入临时文件
            try:
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(current_config, f, indent=2, ensure_ascii=False)
                
                self._dirty_keys.clear()
            except Exception as e:
                print(f"Auto-save failed: {e}")

    def get_value(self, key: str, default: Any = "") -> Any:
        return self.config.get(key, default)
    
    def set_value(self, key: str, value: Any):
        current = self.config.get(key)
        if current == value:
            return
            
        self.config[key] = value
        with self._save_lock:
            self._dirty_keys.add(key)

def grTextBox_autoSave(**kwargs):
    key = kwargs.pop("key", None)
    if key is None:
        return gr.Textbox(**kwargs)
    
    config_saver = WebUIConfigSaver()
    default_value = kwargs.get("value", "")
    saved_value = config_saver.get_value(key, default_value)
    kwargs["value"] = saved_value if saved_value.strip() else default_value
    
    textbox = gr.Textbox(**kwargs)
    
    def update_config(value):
        config_saver.set_value(key, value)
        return value
    
    textbox.change(
        fn=update_config,
        inputs=textbox,
        #outputs=textbox
    )
    
    return textbox