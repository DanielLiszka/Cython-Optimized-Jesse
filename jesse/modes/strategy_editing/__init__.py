from jesse.services.redis import sync_publish, process_status
import json 
import importlib
import sys
import re
import ast 
import os 
import subprocess
from pydantic import BaseModel
import autopep8

def convert_hyperparameters_for_json(hyperparameters):
    json_ready_hyperparameters = []
    for param in hyperparameters:
        if 'type' in param and isinstance(param['type'], type):
            param['type'] = param['type'].__name__

        json_ready_hyperparameters.append(param)

    return json_ready_hyperparameters

def hyperparameters_editing(strategy_name, event_info: int):
    try:
        module_path = f'strategies.{strategy_name}'
        class_name = strategy_name  
        strategies_folder = f'strategies/{strategy_name}/__init__.py'  
        if strategies_folder not in sys.path:
            sys.path.append(strategies_folder)
        strategy_module = importlib.import_module(module_path)
        strategy_class = getattr(strategy_module, class_name)
        strategy_instance = strategy_class()
        hyperparameters = strategy_instance.hyperparameters()
        json_ready_hyperparameters = [convert_hyperparameters_for_json(hyperparameters), event_info]
        sync_publish('shownHyperparameters', json_ready_hyperparameters, 'backtest')
    except Exception as e: 
        print('Failed to Retrieve Strategy hyperparameters')
        print(e)
        sync_publish('shownHyperparameters', '', 'backtest')
        
def preprocess_hyperparameters(updated_hyperparameters):
    for param, value in updated_hyperparameters.items():
        if isinstance(value, (int, float)) and value in [0, 1]:
            updated_hyperparameters[param] = bool(value)

    
class HyperparameterTransformer(ast.NodeTransformer):
    def __init__(self, updated_hyperparameters):
        self.updated_hyperparameters = updated_hyperparameters

    def visit_Return(self, node):
        if isinstance(node.value, ast.List):
            for element in node.value.elts:
                if isinstance(element, ast.Dict):
                    for key, value in zip(element.keys, element.values):
                        if isinstance(key, ast.Str) and key.s == 'name' and isinstance(value, ast.Str) and value.s in self.updated_hyperparameters:
                            for k, v in zip(element.keys, element.values):
                                if isinstance(k, ast.Str) and k.s == 'default':
                                    new_value = self.updated_hyperparameters[value.s]
                                    if isinstance(new_value, int):
                                        v.n = new_value
                                    elif isinstance(new_value, float):
                                        v.n = new_value
                                    elif isinstance(new_value, str):
                                        v.s = new_value
                                    elif isinstance(new_value, bool):
                                        v.value = ast.NameConstant(new_value)
                                    elif isinstance(new_value, (int, float)) and k.get('type', None) == 'bool':
                                        v.value = ast.NameConstant(bool(new_value))

        return node
        
def code_receiving(strategy_name):
    try:
        strategy_path = f'strategies/{strategy_name}/__init__.py'  
        with open(strategy_path, 'r') as file:
            content = file.read()
        return content
    except Exception as e: 
        print(e)
        pass

def compile_code(code):
    try:
        compile(code, "<string>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, str(e)

def lint_code(code, filename='temp_code.py'):
    with open(filename, 'w') as file:
        file.write(code)
    result = subprocess.run(['flake8', filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.remove(filename)
    return result.stdout.decode('utf-8')

def format_code_with_autopep8(code):
    """Format code using autopep8."""
    return autopep8.fix_code(code, options={'aggressive': 1})

def code_saving(strategy_name, code):
    file_path = f'strategies/{strategy_name}/__init__.py'
    is_valid, error = compile_code(code)
    if not is_valid:
        return f"Syntax error: {error}"
    formatted_code = format_code_with_autopep8(code)
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            file.write(formatted_code)
        return "success"
    except Exception as e:
        return f"Error saving file: {str(e)}"   

def format_dict_list(node):
    for i, dict_ in enumerate(node.elts):
        if isinstance(dict_, ast.Dict):
            formatted_pairs = []
            for key, value in zip(dict_.keys, dict_.values):
                key_str = ast.unparse(key)
                value_str = ast.unparse(value)
                formatted_pairs.append(f"{key_str}: {value_str}")
            node.elts[i] = ast.parse("{" + ",\n".join(formatted_pairs) + "}")


def update_hyperparameters_from_json(strategy_name, updated_hyperparameters):
    preprocess_hyperparameters(updated_hyperparameters)
    for key, value in updated_hyperparameters.items():
        if isinstance(value, str) and '.' in value:
            try:
                updated_hyperparameters[key] = float(value)
            except ValueError:
                pass
    for key, value in updated_hyperparameters.items():
        if isinstance(value, str) and value.lower() in ['true', 'false']:
            updated_hyperparameters[key] = value.lower() == 'true'

    file_path = f'strategies/{strategy_name}/__init__.py'
    original_content = None
    try:
        with open(file_path, 'r') as file:
            original_content = file.read()
        tree = ast.parse(original_content)
        transformer = HyperparameterTransformer(updated_hyperparameters)
        transformer.visit(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'hyperparameters':
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.Return) and isinstance(child.value, ast.List):
                        format_dict_list(child.value)

        modified_code = ast.unparse(tree)

        with open(file_path, 'w') as file:
            file.write(modified_code)

    except Exception as e:
        print(f'Failed to update hyperparameters. Error: {e}')
        if original_content is not None:
            with open(file_path, 'w') as file:
                file.write(original_content)

