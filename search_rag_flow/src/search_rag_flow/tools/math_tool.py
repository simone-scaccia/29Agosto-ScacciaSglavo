"""Tools and validators for safe evaluation of math-only Python functions.

This module exposes a validator utility and a CrewAI tool that ensure a given
piece of source code defines exactly one function limited to math operations
from an allowlist. It is intended as a guardrail for math-related tasks.
"""

from crewai.tools import tool
import ast
import math

ALLOWED_NAMES = {
    # Allowed math functions
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'sqrt': math.sqrt,
    'log': math.log,
    'exp': math.exp,
    'pi': math.pi,
    'e': math.e,
    # Built-in functions allowed
    'abs': abs,
    'round': round,
}

def is_math_function_valid(code: str) -> bool:
    """Validate that code defines a safe math-only Python function.

    The validation ensures the provided source code parses to exactly one
    top-level function definition and that all operations within the function
    body use only a restricted allowlist of names and constructs suitable for
    mathematical evaluation. Import statements and disallowed calls are
    rejected.

    Args:
        code (str): Source code string that should define a single Python
            function.

    Returns:
        bool: ``True`` if the code is a valid, restricted math function;
        otherwise ``False``.
    """
    try:
        # Parse the code into an AST
        tree = ast.parse(code)

        # There must be exactly one function defined
        if not (len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)):
            return False

        func_def = tree.body[0]

        # Walk through all nodes of the function body
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call):
                # Check that the called function is allowed
                if isinstance(node.func, ast.Name):
                    if node.func.id not in ALLOWED_NAMES:
                        return False
                else:
                    # If function call is complex (e.g., attribute), reject
                    return False
            elif isinstance(node, ast.Name):
                # Allow names only if they are function parameters or allowed names
                if node.id not in {arg.arg for arg in func_def.args.args} and node.id not in ALLOWED_NAMES:
                    return False
            elif isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
                # Disallow imports and other disallowed statements
                return False
            # Other node types like BinOp, Return, Expr, Constant, FunctionDef are acceptable

        # If all checks pass, the function is valid
        return True

    except SyntaxError:
        return False


@tool("Python math function validator")
def validate_math_expression(code: str) -> str:
    """Validate that a string defines an allowed math-only function.

    This is a safe guardrail utility for math tasks. It parses and checks the
    provided function code using a strict allowlist of names and disallows
    side-effectful constructs.

    Args:
        code (str): Source code expected to contain exactly one function
            definition.

    Returns:
        str: The string ``"Valid function"`` if validation succeeds.

    Raises:
        ValueError: If the input is empty/invalid, or the function uses
            disallowed elements.
    """
    if not isinstance(code, str) or len(code.strip()) == 0:
        raise ValueError("Empty or invalid code input.")

    if is_math_function_valid(code):
        return "Valid function"
    else:
        raise ValueError("Invalid function or contains disallowed elements.")
