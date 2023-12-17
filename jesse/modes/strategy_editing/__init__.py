from jesse.services.redis import sync_publish, process_status
import json 
import importlib
import sys
import re
import ast 
import astor 

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
        print(json_ready_hyperparameters)
    except Exception as e: 
        print('Failed to Retrieve Strategy hyperparameters')
        print(e)
        
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
                                        # Convert numeric to boolean
                                        v.value = ast.NameConstant(bool(new_value))

        return node

def update_hyperparameters_from_json(strategy_name, updated_hyperparameters):
    preprocess_hyperparameters(updated_hyperparameters)
    for key, value in updated_hyperparameters.items():
        if isinstance(value, str) and '.' in value:
            try:
                updated_hyperparameters[key] = float(value)
            except ValueError:
                pass  # Handle or log error if necessary
    # Convert string representations of booleans to actual boolean values
    for key, value in updated_hyperparameters.items():
        if isinstance(value, str) and value.lower() in ['true', 'false']:
            updated_hyperparameters[key] = value.lower() == 'true'

    file_path = f'strategies/{strategy_name}/__init__.py'
    original_content = None
    print(updated_hyperparameters)
    try:
        with open(file_path, 'r') as file:
            original_content = file.read()
        tree = ast.parse(original_content)
        transformer = HyperparameterTransformer(updated_hyperparameters)
        transformer.visit(tree)

        # Using ast.unparse to convert AST back to source code
        modified_code = ast.unparse(tree)

        with open(file_path, 'w') as file:
            file.write(modified_code)

    except Exception as e:
        print(f'Failed to update hyperparameters. Error: {e}')
        if original_content is not None:
            with open(file_path, 'w') as file:
                file.write(original_content)

