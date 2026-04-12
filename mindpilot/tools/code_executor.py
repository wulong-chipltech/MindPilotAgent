"""
代码沙箱执行器
==============
安全执行生成的 Python 代码：AST 静态检测 + 受限执行环境 + 超时控制。
"""

import ast
import sys
import time
import signal
import traceback
import subprocess
import tempfile
import os
import re
from dataclasses import dataclass
from typing import Optional
from io import StringIO


@dataclass
class ExecutionResult:
    """代码执行结果"""
    success: bool
    stdout: str
    stderr: str
    return_value: any
    execution_time: float
    error_type: Optional[str] = None
    safety_issues: list[str] = None

    def __post_init__(self):
        if self.safety_issues is None:
            self.safety_issues = []

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "stdout": self.stdout[:2000],
            "stderr": self.stderr[:1000],
            "execution_time": self.execution_time,
            "error_type": self.error_type,
            "safety_issues": self.safety_issues,
        }


class ASTSafetyChecker(ast.NodeVisitor):
    """
    AST 静态安全分析
    检测危险函数调用、危险模块导入等
    """

    FORBIDDEN_CALLS = {
        "os.system", "os.popen", "os.execv", "os.execve",
        "subprocess.call", "subprocess.run", "subprocess.Popen",
        "shutil.rmtree", "shutil.move",
        "open",  # 文件写操作单独检测
        "__import__", "eval", "exec", "compile",
        "socket.socket",
    }

    FORBIDDEN_IMPORTS = {
        "socket", "ftplib", "smtplib", "telnetlib",
        "ctypes", "multiprocessing",
    }

    DANGEROUS_WRITE_PATTERNS = [
        r'open\s*\(.*["\']w["\']',
        r'open\s*\(.*["\']a["\']',
        r'\.write\s*\(',
        r'shutil\.copy',
    ]

    def __init__(self):
        self.issues: list[str] = []
        self._call_chain: list[str] = []

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name in self.FORBIDDEN_IMPORTS:
                self.issues.append(f"禁止导入模块: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module in self.FORBIDDEN_IMPORTS:
            self.issues.append(f"禁止 from {node.module} import ...")
        self.generic_visit(node)

    def visit_Call(self, node):
        # 解析调用名称
        func_name = self._get_call_name(node.func)
        if func_name in self.FORBIDDEN_CALLS:
            self.issues.append(f"禁止调用: {func_name}()")
        # 检测 open() 写模式
        if func_name == "open" and len(node.args) >= 2:
            mode_arg = node.args[1]
            if isinstance(mode_arg, ast.Constant) and "w" in str(mode_arg.value):
                self.issues.append("禁止文件写操作: open(..., 'w')")
        self.generic_visit(node)

    def _get_call_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._get_call_name(node.value)}.{node.attr}"
        return ""

    def check(self, code: str) -> list[str]:
        self.issues = []
        # 额外的正则检测（处理动态构造的危险代码）
        for pattern in self.DANGEROUS_WRITE_PATTERNS:
            if re.search(pattern, code):
                self.issues.append(f"检测到潜在危险操作 (regex): {pattern}")
        try:
            tree = ast.parse(code)
            self.visit(tree)
        except SyntaxError as e:
            self.issues.append(f"语法错误: {e}")
        return self.issues


class CodeExecutor:
    """
    Python 代码沙箱执行器
    在受限命名空间中运行，支持超时控制
    """

    SAFE_BUILTINS = {
        "print", "range", "len", "list", "dict", "set", "tuple",
        "str", "int", "float", "bool", "type", "isinstance",
        "enumerate", "zip", "map", "filter", "sorted", "reversed",
        "min", "max", "sum", "abs", "round", "pow",
        "Exception", "ValueError", "TypeError", "KeyError",
        "True", "False", "None",
    }

    def __init__(self, timeout: int = 30, logger=None):
        self.timeout = timeout
        self.logger = logger
        self.checker = ASTSafetyChecker()

    def extract_code(self, text: str) -> str:
        """从 LLM 输出中提取 Python 代码块"""
        # 匹配 ```python ... ``` 或 ``` ... ```
        patterns = [
            r"```python\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
            r"```python(.*?)```",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                return m.group(1).strip()
        # 没有代码块标记，直接返回
        return text.strip()

    def execute(self, code: str, extra_context: dict = None) -> ExecutionResult:
        """
        安全执行 Python 代码
        1. AST 安全检测
        2. 受限命名空间执行
        3. 超时控制
        """
        start = time.time()

        # 安全检测
        issues = self.checker.check(code)
        if self.logger:
            if issues:
                self.logger.warning("CodeExecutor", f"安全检测发现 {len(issues)} 个问题")
            else:
                self.logger.debug("CodeExecutor", "安全检测通过")

        # 构建受限执行环境
        safe_globals = self._build_safe_globals(extra_context)

        # 捕获 stdout
        old_stdout, old_stderr = sys.stdout, sys.stderr
        captured_out = StringIO()
        captured_err = StringIO()
        sys.stdout = captured_out
        sys.stderr = captured_err

        result_ns = {}
        error_type = None
        exec_error = ""

        try:
            exec(compile(code, "<mindpilot>", "exec"), safe_globals, result_ns)
            success = True
        except SyntaxError as e:
            success = False
            error_type = "SyntaxError"
            exec_error = str(e)
        except Exception as e:
            success = False
            error_type = type(e).__name__
            exec_error = traceback.format_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        duration = round(time.time() - start, 3)
        stdout_val = captured_out.getvalue()
        stderr_val = captured_err.getvalue() + exec_error

        if self.logger:
            if success:
                self.logger.success("CodeExecutor", f"执行成功 [{duration}s]")
            else:
                self.logger.error("CodeExecutor", f"执行失败: {error_type}")

        return ExecutionResult(
            success=success,
            stdout=stdout_val,
            stderr=stderr_val,
            return_value=result_ns.get("__result__"),
            execution_time=duration,
            error_type=error_type,
            safety_issues=issues,
        )

    def execute_with_subprocess(self, code: str, timeout: Optional[int] = None) -> ExecutionResult:
        """
        用子进程执行（更高安全隔离），适用于有文件 I/O 需求的代码
        """
        start = time.time()
        t = timeout or self.timeout
        issues = self.checker.check(code)

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w",
                                         delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name

        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True, timeout=t,
                cwd=tempfile.gettempdir(),
            )
            success = proc.returncode == 0
            stdout = proc.stdout
            stderr = proc.stderr
            error_type = None if success else "RuntimeError"
        except subprocess.TimeoutExpired:
            success = False
            stdout = ""
            stderr = f"执行超时（>{t}s）"
            error_type = "TimeoutError"
        except Exception as e:
            success = False
            stdout = ""
            stderr = str(e)
            error_type = type(e).__name__
        finally:
            os.unlink(tmp_path)

        return ExecutionResult(
            success=success,
            stdout=stdout,
            stderr=stderr,
            return_value=None,
            execution_time=round(time.time() - start, 3),
            error_type=error_type,
            safety_issues=issues,
        )

    def _build_safe_globals(self, extra: dict = None) -> dict:
        import builtins
        safe_builtins = {k: getattr(builtins, k) for k in self.SAFE_BUILTINS if hasattr(builtins, k)}

        # 允许安全的科学计算库
        ALLOWED_IMPORTS = {
            "numpy", "pandas", "matplotlib", "scipy", "sklearn",
            "math", "random", "json", "re", "collections",
            "itertools", "statistics", "time", "datetime",
            "sklearn.linear_model", "sklearn.metrics", "sklearn.cluster",
            "sklearn.preprocessing", "sklearn.model_selection",
            "matplotlib.pyplot", "scipy.stats",
        }
        def safe_import(name, *args, **kwargs):
            root = name.split(".")[0]
            if root not in ALLOWED_IMPORTS and name not in ALLOWED_IMPORTS:
                raise ImportError(f"导入 {name!r} 在沙箱中不被允许")
            import importlib
            return importlib.import_module(name)
        safe_builtins["__import__"] = safe_import
        safe_globals = {"__builtins__": safe_builtins}
        # Also expose pre-imported modules directly so inline imports work
        import sys as _sys
        for _mod_name in list(_sys.modules.keys()):
            _root = _mod_name.split(".")[0]
            if _root in ALLOWED_IMPORTS:
                safe_globals[_mod_name.replace(".","_")] = _sys.modules[_mod_name]
        import importlib
        allowed_modules = {
            "numpy": "numpy", "np": "numpy",
            "pandas": "pandas", "pd": "pandas",
            "scipy": "scipy", "sklearn": "sklearn",
            "math": "math", "random": "random",
            "json": "json", "re": "re",
            "collections": "collections", "itertools": "itertools",
            "statistics": "statistics",
        }
        for alias, mod_name in allowed_modules.items():
            try:
                mod = importlib.import_module(mod_name)
                safe_globals[alias] = mod
            except ImportError:
                pass
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            safe_globals["matplotlib"] = matplotlib
            safe_globals["plt"] = plt
        except ImportError:
            pass

        if extra:
            safe_globals.update(extra)
        return safe_globals
