from typing import List, Dict, Optional, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

fastapi_app = FastAPI()

origins = [
    "*",
]

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BacktestRequestJson(BaseModel):
    id: str
    routes: List[Dict[str, str]]
    extra_routes: List[Dict[str, str]]
    config: dict
    start_date: str
    finish_date: str
    debug_mode: bool
    export_csv: bool
    export_json: bool
    export_chart: bool
    export_tradingview: bool
    export_full_reports: bool
    export_backtesting_chart: bool
    export_correlation_table: bool

class ParamEvalRequestJson(BaseModel):
    id: str
    routes: List[Dict[str, str]]
    extra_routes: List[Dict[str, str]]
    config: dict
    start_date: str
    finish_date: str
    debug_mode: bool

class InitialChartingRequestJson(BaseModel):
    id: str
    config: dict
    routes: Dict[str, str]
    start_date: str
    finish_date: str
    destination: list
    indicator_info: dict
    

class OptimizationRequestJson(BaseModel):
    id: str
    routes: List[Dict[str, str]]
    extra_routes: List[Dict[str, str]]
    config: dict
    start_date: str
    finish_date: str
    optimal_total: int
    debug_mode: bool
    export_csv: bool
    export_json: bool

class OptunaRequestJson(BaseModel):
    id: str
    checked_study: bool
    continueExistingStudy: bool
    routes: List[Dict[str, str]]
    extra_routes: List[Dict[str, str]]
    config: dict
    start_date: str
    finish_date: str
    optimal_total: int
    debug_mode: bool
    optimizer: str
    fitnessMetric1: str
    fitnessMetric2: str
    isMultivariate: bool
    secondObjectiveDirection: str
    consider_prior: bool
    prior_weight: float
    n_startup_trials: int
    n_ei_candidates: int
    group: bool 
    sigma: float
    consider_pruned_trials: bool
    population_size: int
    crossover_prob: float
    swapping_prob: float
    qmc_type: str
    scramble: bool
    n_trials: int
    do_analysis: bool
    validation_interval: int
    robust_test_iteration_count: int
    robust_test_max: float
    robust_test_min: float
    max_final_number_of_validation_results: int
    optuna_visualizations: bool

class OptunaSpecialRequestJson(BaseModel):
    id: str
    string: str
    
class CodeSendingRequestJson(BaseModel):
    strategy_name: str
    code: str
    
class HyperparametersSendingRequestJson(BaseModel):
    id: str
    current_page: str
    strategy_name: str
    
HyperparameterValue = Union[int, float, str]
class HyperparametersSavingRequestJson(BaseModel):
    id: str
    current_page: str
    strategy_name: str
    hyperparameters: Dict[str, HyperparameterValue]
    
class ImportCandlesRequestJson(BaseModel):
    id: str
    exchange: str
    symbol: str
    start_date: str

class CodeFormattingRequestJson(BaseModel):
    code: str
    
class CancelRequestJson(BaseModel):
    id: str


class LiveRequestJson(BaseModel):
    id: str
    config: dict
    routes: List[Dict[str, str]]
    extra_routes: List[Dict[str, str]]
    debug_mode: bool
    paper_mode: bool


class LiveCancelRequestJson(BaseModel):
    id: str
    paper_mode: bool


class GetCandlesRequestJson(BaseModel):
    id: str
    exchange: str
    symbol: str
    timeframe: str


class GetLogsRequestJson(BaseModel):
    id: str
    session_id: str
    type: str


class GetOrdersRequestJson(BaseModel):
    id: str
    session_id: str


class ConfigRequestJson(BaseModel):
    current_config: dict


class LoginRequestJson(BaseModel):
    password: str


class LoginJesseTradeRequestJson(BaseModel):
    email: str
    password: str


class NewStrategyRequestJson(BaseModel):
    name: str


class FeedbackRequestJson(BaseModel):
    description: str
    email: Optional[str] = None


class ReportExceptionRequestJson(BaseModel):
    description: str
    traceback: str
    mode: str
    attach_logs: bool
    session_id: Optional[str] = None
    email: Optional[str] = None
