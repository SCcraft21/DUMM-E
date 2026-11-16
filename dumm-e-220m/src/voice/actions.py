"""Safe local action dispatcher with synthetic Conscience clearance protocol."""

import ast
import json
import operator
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union

from src.tokenizer.special_tokens import (
    ACTION_START_TOKEN,
    ACTION_END_TOKEN,
    OBSERVATION_START_TOKEN,
    OBSERVATION_END_TOKEN,
)


class SafeCalculator:
    """Safe AST-based arithmetic evaluator preventing arbitrary code execution."""

    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    @classmethod
    def evaluate(cls, expression: str) -> Union[int, float]:
        node = ast.parse(expression.strip(), mode="eval").body
        return cls._eval_node(node)

    @classmethod
    def _eval_node(cls, node: ast.AST) -> Union[int, float]:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            op = cls.ALLOWED_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported mathematical operator: {type(node.op)}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = cls._eval_node(node.operand)
            op = cls.ALLOWED_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
            return op(operand)
        raise ValueError(f"Unsafe or unsupported expression syntax: {ast.dump(node)}")


class LocalActionDispatcher:
    """Dispatcher for executing sandboxed local system actions and tools.

    Enforces mandatory Conscience Head clearance prior to tool execution.
    """

    def __init__(
        self,
        workspace_dir: Optional[Path] = None,
        default_risk_threshold: float = 0.50,
    ) -> None:
        self.workspace_dir = Path(workspace_dir or Path.cwd()).resolve()
        self.default_risk_threshold = default_risk_threshold
        self._tools: Dict[str, Callable[[Dict[str, Any]], Any]] = {
            "calculator": self._tool_calculator,
            "system_info": self._tool_system_info,
            "datetime_query": self._tool_datetime,
            "file_read": self._tool_file_read,
            "weather_query": self._tool_weather,
        }

    def register_tool(self, name: str, handler: Callable[[Dict[str, Any]], Any]) -> None:
        """Register a custom tool handler."""
        self._tools[name] = handler

    # Safe built-in tools
    def _tool_calculator(self, params: Dict[str, Any]) -> Dict[str, Any]:
        expr = params.get("expression", "")
        result = SafeCalculator.evaluate(expr)
        return {"result": result}

    def _tool_system_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        metric = params.get("metric", "all")
        info = {
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "python_version": sys.version.split()[0],
        }
        if metric != "all" and metric in info:
            return {metric: info[metric]}
        return info

    def _tool_datetime(self, params: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now()
        return {
            "iso_time": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day": now.strftime("%A"),
        }

    def _tool_file_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        file_path = params.get("path", "")
        target = (self.workspace_dir / file_path).resolve()
        # Enforce sandbox: prohibit path traversal outside workspace
        if not str(target).startswith(str(self.workspace_dir)):
            raise PermissionError("Path traversal violation: Cannot access files outside workspace.")
        if not target.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        content = target.read_text(encoding="utf-8", errors="replace")[:1000]
        return {"content": content, "truncated": len(content) == 1000}

    def _tool_weather(self, params: Dict[str, Any]) -> Dict[str, Any]:
        location = params.get("location", "Local")
        return {"location": location, "condition": "Sunny", "temperature_c": 22, "humidity": "45%"}

    def parse_action_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse structured JSON action from within `<|action|> ... <|/action|>` tags."""
        if ACTION_START_TOKEN in text and ACTION_END_TOKEN in text:
            start = text.index(ACTION_START_TOKEN) + len(ACTION_START_TOKEN)
            end = text.index(ACTION_END_TOKEN)
            action_payload_str = text[start:end].strip()
            try:
                return json.loads(action_payload_str)
            except json.JSONDecodeError:
                return {"action": "raw", "raw": action_payload_str}
        return None

    def execute_action(
        self,
        action_payload: Union[str, Dict[str, Any]],
        conscience_risk_score: float,
        risk_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Verify conscience clearance and execute action safely.

        Args:
            action_payload: Dictionary with `{"action": "name", "params": {...}}` or raw string.
            conscience_risk_score: Penultimate layer risk prediction from Conscience Head.
            risk_threshold: Maximum permissible risk score.

        Returns:
            Observation dictionary.
        """
        threshold = risk_threshold if risk_threshold is not None else self.default_risk_threshold

        # 1. Conscience Clearance Verification
        if conscience_risk_score > threshold:
            return {
                "status": "blocked",
                "clearance": "DENIED",
                "risk_score": conscience_risk_score,
                "threshold": threshold,
                "error": "Action execution blocked: Failed synthetic conscience moral verification.",
            }

        # 2. Parse dictionary payload if necessary
        if isinstance(action_payload, str):
            try:
                action_dict = json.loads(action_payload)
            except json.JSONDecodeError:
                return {
                    "status": "error",
                    "error": f"Invalid JSON action payload: {action_payload}",
                }
        else:
            action_dict = action_payload

        action_name = action_dict.get("action")
        params = action_dict.get("params", {})

        if action_name not in self._tools:
            return {
                "status": "error",
                "error": f"Unknown action tool '{action_name}'. Registered tools: {list(self._tools.keys())}",
            }

        # 3. Safe Dispatch
        try:
            result = self._tools[action_name](params)
            return {
                "status": "success",
                "clearance": "APPROVED",
                "action": action_name,
                "result": result,
            }
        except Exception as e:
            return {
                "status": "error",
                "action": action_name,
                "error": str(e),
            }

    def format_observation(self, observation: Dict[str, Any]) -> str:
        """Format observation into model token delimiters."""
        obs_json = json.dumps(observation)
        return f"\n{OBSERVATION_START_TOKEN}\n{obs_json}\n{OBSERVATION_END_TOKEN}\n"

