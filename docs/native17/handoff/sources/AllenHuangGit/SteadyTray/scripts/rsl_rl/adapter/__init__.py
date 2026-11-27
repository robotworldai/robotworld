
"""
Adapter module for parameter-efficient fine-tuning of RL policies.

This module provides two adapter approaches combined with temporal encoders:
1. FiLM adapters: Feature-wise linear modulation at intermediate layers
2. Residual action adapters: Direct action-space corrections

Components:
- AdaptedActorCritic: Actor-critic with FiLM adapters for feature modulation
- ResidualActorCritic: Actor-critic with residual action corrections
- AdaptedStudentTeacher: Student-teacher distillation (supports both adapter types)
- FiLMAdapter: Feature-wise modulation layer
- ResidualActionAdapter: Action-space residual layer
- Encoder: GRU/Transformer-based history encoder for temporal context
"""

from .actor_critic import AdaptedActorCritic, ResidualActorCritic, AdapterSequential
from .adapter import FiLMAdapter, ResidualActionAdapter
from .encoder import GRUEncoder
from .student_teacher import AdaptedStudentTeacher

__all__ = [
    "AdaptedActorCritic",
    "ResidualActorCritic",
    "AdaptedStudentTeacher",
    "AdapterSequential", 
    "FiLMAdapter",
    "ResidualActionAdapter",
    "GRUEncoder",
]
