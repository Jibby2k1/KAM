from .char_text import CharacterDataset, load_text
from .dyck import BoundedDyck2Dataset
from .mackey_glass import MackeyGlassDataset, generate_mackey_glass, make_mackey_splits
from .prototype_switch import generate_prototype_switch, make_prototype_switch_splits
from .mqar import MQARDataset
from .narma import NARMADataset, generate_narma, make_narma_splits
from .stream_schedules import (
    ScheduleSegment,
    generate_switching_mackey_glass,
    generate_switching_narma,
    schedule_labels,
    schedule_segments,
)
from .synthetic import CopyLanguageDataset, RegimeGrammarDataset
from .switching import SwitchingRegressionDataset, SwitchingRegressionSplits, make_switching_mackey_splits, make_switching_narma_splits
from .variable_copy import VariableCopyLanguageDataset, variable_copy_collate

__all__ = [
    "BoundedDyck2Dataset",
    "CharacterDataset",
    "CopyLanguageDataset",
    "VariableCopyLanguageDataset",
    "variable_copy_collate",
    "SwitchingRegressionDataset",
    "SwitchingRegressionSplits",
    "make_switching_mackey_splits",
    "make_switching_narma_splits",
    "MQARDataset",
    "MackeyGlassDataset",
    "NARMADataset",
    "RegimeGrammarDataset",
    "ScheduleSegment",
    "generate_mackey_glass",
    "generate_narma",
    "generate_switching_mackey_glass",
    "generate_switching_narma",
    "load_text",
    "make_mackey_splits",
    "generate_prototype_switch",
    "make_prototype_switch_splits",
    "make_narma_splits",
    "schedule_labels",
    "schedule_segments",
]
