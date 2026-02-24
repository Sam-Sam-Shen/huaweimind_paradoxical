"""
配置管理模块

加载和管理项目配置文件
"""

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """配置管理类"""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self._config: dict[str, Any] = {}
        self._config_path = config_path
        self._load_config()

    def _load_config(self) -> None:
        """加载配置文件"""
        config_file = Path(self._config_path)
        
        if not config_file.exists():
            base_dir = Path(__file__).parent.parent
            config_file = base_dir / "config.yaml"
        
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"配置文件未找到: {config_file}")

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的键"""
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            
            if value is None:
                return default
                
        return value

    @property
    def paths(self) -> dict[str, str]:
        """获取路径配置"""
        return self._config.get("paths", {})

    @property
    def preprocessing(self) -> dict[str, Any]:
        """获取预处理配置"""
        return self._config.get("preprocessing", {})

    @property
    def topic_modeling(self) -> dict[str, Any]:
        """获取主题建模配置"""
        return self._config.get("topic_modeling", {})

    @property
    def contradiction_detection(self) -> dict[str, Any]:
        """获取矛盾检测配置"""
        return self._config.get("contradiction_detection", {})

    @property
    def word_embedding(self) -> dict[str, Any]:
        """获取词向量配置"""
        return self._config.get("word_embedding", {})

    @property
    def visualization(self) -> dict[str, Any]:
        """获取可视化配置"""
        return self._config.get("visualization", {})

    @property
    def report(self) -> dict[str, Any]:
        """获取报告配置"""
        return self._config.get("report", {})

    @property
    def logging(self) -> dict[str, str]:
        """获取日志配置"""
        return self._config.get("logging", {})

    def get_path(self, key: str) -> str:
        """获取完整路径"""
        base_dir = Path(__file__).parent.parent
        path = self.get(f"paths.{key}", "")
        return str(base_dir / path) if path else ""

    def resolve_paths(self) -> None:
        """解析所有相对路径为绝对路径"""
        base_dir = Path(__file__).parent.parent
        
        # 解析路径配置
        if "paths" in self._config:
            for key, value in self._config["paths"].items():
                if value and not Path(value).is_absolute():
                    self._config["paths"][key] = str(base_dir / value)


# 全局配置实例
_config_instance: Config | None = None


def get_config(config_path: str = "config.yaml") -> Config:
    """获取配置单例"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
        _config_instance.resolve_paths()
    return _config_instance
