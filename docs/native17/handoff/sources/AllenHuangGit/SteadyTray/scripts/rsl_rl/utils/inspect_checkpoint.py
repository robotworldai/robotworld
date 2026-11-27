#!/usr/bin/env python3
"""
Standalone utility to inspect checkpoint files.

Usage:
    python scripts/rsl_rl/utils/inspect_checkpoint.py path/to/checkpoint.pt
    python scripts/rsl_rl/utils/inspect_checkpoint.py path/to/checkpoint.pt --no-shapes
    python scripts/rsl_rl/utils/inspect_checkpoint.py path/to/checkpoint.pt --max-params 50
"""

import argparse
import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description='Inspect checkpoint file structure and contents')
    parser.add_argument('checkpoint_path', type=str, help='Path to checkpoint file (.pt)')
    parser.add_argument('--no-shapes', action='store_true', help='Hide parameter shapes')
    parser.add_argument('--max-params', type=int, default=0, 
                       help='Maximum parameters to show per group (0 = show all)')
    parser.add_argument('--compare', type=str, default=None,
                       help='Compare with another checkpoint')
    
    args = parser.parse_args()
    
    # Inspect primary checkpoint
    inspect_checkpoint(
        args.checkpoint_path,
        show_shapes=not args.no_shapes,
        max_params=args.max_params
    )
    
    # Compare if requested
    if args.compare:
        print("\n" + "="*70)
        print("COMPARING WITH SECOND CHECKPOINT")
        print("="*70 + "\n")
        
        inspect_checkpoint(
            args.compare,
            show_shapes=not args.no_shapes,
            max_params=args.max_params
        )


def inspect_checkpoint(checkpoint_path: str, show_shapes: bool = True, max_params: int = 5):
        """
        Inspect a checkpoint file to see its structure and contents.
        
        Args:
            checkpoint_path: Path to the checkpoint file (.pt)
            show_shapes: Whether to show parameter shapes
            max_params: Maximum number of parameters to display per group (0 = show all)
        
        Usage:
            AdaptedActorCritic.inspect_checkpoint("logs/baseline/model_5000.pt")
        """
        import os
        
        print(f"\n{'='*70}")
        print(f"Checkpoint: {checkpoint_path}")
        
        # Check file exists and size
        if not os.path.exists(checkpoint_path):
            print(f"❌ File not found")
            print(f"{'='*70}\n")
            return
        
        file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"Size: {file_size_mb:.2f} MB")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"❌ Error loading: {e}")
            print(f"{'='*70}\n")
            return
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if 'iter' in checkpoint:
                print(f"Iteration: {checkpoint['iter']}")
        else:
            state_dict = checkpoint if isinstance(checkpoint, dict) else {}
        
        # Detect checkpoint type
        has_actor = any(k.startswith('actor.') for k in state_dict.keys())
        has_actor_body = any(k.startswith('actor_body.') for k in state_dict.keys())
        has_frozen_actor = any(k.startswith('frozen_actor.') for k in state_dict.keys())
        has_film_adapters = any(('alpha' in k or 'mod.' in k) and 'actor_body' in k for k in state_dict.keys())
        has_residual_adapter = any('residual_adapter' in k for k in state_dict.keys())
        has_encoder = any('history_encoder' in k for k in state_dict.keys())
        has_action_head = any('action_head' in k for k in state_dict.keys())
        has_student_encoder = any('student_encoder' in k for k in state_dict.keys())
        has_teacher_encoder = any('teacher_encoder' in k for k in state_dict.keys())
        
        # Determine checkpoint type
        if has_student_encoder and has_teacher_encoder:
            if has_actor_body and has_film_adapters:
                checkpoint_type = "AdaptedStudentTeacher (FiLM)"
            elif has_frozen_actor and has_residual_adapter:
                checkpoint_type = "AdaptedStudentTeacher (Residual)"
            else:
                checkpoint_type = "AdaptedStudentTeacher"
            print(f"Type: {checkpoint_type} ✓ Distillation checkpoint")
        elif has_actor_body and has_film_adapters and has_encoder and has_action_head:
            checkpoint_type = "AdaptedActorCritic (FiLM)"
            print(f"Type: {checkpoint_type} ✓ FiLM adapter checkpoint")
        elif has_frozen_actor and has_residual_adapter and has_encoder:
            checkpoint_type = "ResidualActorCritic"
            print(f"Type: {checkpoint_type} ✓ Residual adapter checkpoint")
        elif has_actor and not (has_film_adapters or has_residual_adapter):
            checkpoint_type = "Base Policy"
            print(f"Type: {checkpoint_type} ✓ Load as pre-trained")
        else:
            checkpoint_type = "Unknown"
            print(f"Type: {checkpoint_type} ⚠ Manual handling needed")
        
        # Group and count parameters
        param_groups = {}
        total_params = 0
        
        for name, param in state_dict.items():
            num_params = param.numel() if hasattr(param, 'numel') else 0
            total_params += num_params
            shape_str = f" {list(param.shape)}" if show_shapes and hasattr(param, 'shape') else ""
            
            # Categorize parameter - simplified for both FiLM and Residual
            if 'student_encoder' in name:
                group = 'Student Encoder'
            elif 'teacher_encoder' in name:
                group = 'Teacher Encoder'
            elif 'history_encoder' in name:
                group = 'History Encoder'
            elif 'frozen_actor' in name:
                # ResidualActorCritic: frozen full actor
                group = 'Frozen Actor'
            elif 'residual_adapter' in name:
                # ResidualActorCritic: residual action adapter
                group = 'Residual Adapter'
            elif 'actor_body' in name:
                # AdaptedActorCritic: FiLM adapter layers (includes base + adapters)
                group = 'Actor Body (FiLM)'
            elif 'action_head' in name:
                # AdaptedActorCritic: action head (frozen)
                group = 'Action Head'
            elif name.startswith('actor.'):
                # Base policy checkpoint structure
                group = 'Actor (Base)'
            elif name.startswith('critic.'):
                group = 'Critic'
            elif 'std' in name or 'log_std' in name:
                group = 'Noise (std)'
            else:
                group = 'Other'
            
            if group not in param_groups:
                param_groups[group] = []
            param_groups[group].append((name, num_params, shape_str))
        
        # Print parameter summary
        print(f"{'-'*70}")
        print(f"Parameters: {total_params:,} total")
        
        # Define display order based on checkpoint type
        if "AdaptedStudentTeacher" in checkpoint_type:
            # Distillation checkpoints
            group_order = ['Student Encoder', 'Teacher Encoder', 
                          'Frozen Actor', 'Actor Body (FiLM)', 'Action Head',
                          'Residual Adapter', 'History Encoder',
                          'Critic', 'Noise (std)', 'Other']
        elif "ResidualActorCritic" in checkpoint_type:
            # Residual adapter checkpoints
            group_order = ['Frozen Actor', 'Residual Adapter', 'History Encoder',
                          'Critic', 'Noise (std)', 'Other']
        elif "AdaptedActorCritic" in checkpoint_type:
            # FiLM adapter checkpoints
            group_order = ['Actor Body (FiLM)', 'Action Head', 'History Encoder',
                          'Critic', 'Noise (std)', 'Other']
        else:
            # Base policy or unknown
            group_order = ['Actor (Base)', 'Critic', 'Noise (std)',
                          'Frozen Actor', 'Actor Body (FiLM)', 'Action Head',
                          'Residual Adapter', 'History Encoder',
                          'Student Encoder', 'Teacher Encoder', 'Other']
        
        for group_name in group_order:
            if group_name in param_groups:
                params = param_groups[group_name]
                group_total = sum(p[1] for p in params)
                print(f"\n{group_name}: {group_total:,} ({len(params)} tensors)")
                
                # Show sample parameters
                display_count = len(params) if max_params == 0 else min(len(params), max_params)
                for name, num_params, shape_str in params[:display_count]:
                    print(f"  • {name}{shape_str}")
                
                if len(params) > display_count:
                    print(f"  ... +{len(params) - display_count} more")
        
        print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
