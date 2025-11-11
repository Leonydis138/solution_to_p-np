#!/usr/bin/env python3
"""CLI entrypoint: run validation (placeholder)"""
import argparse
import json
from src.validation.robust_validator import RobustIndependenceValidator
from src.validation.statistical_validator import ValidationConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-values', nargs='+', type=int, default=[20,50,100])
    parser.add_argument('--t-values', nargs='+', type=int, default=[3,5,10])
    parser.add_argument('--confidence', type=float, default=0.99)
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--output', type=str, default='results/validation.json')
    args = parser.parse_args()

    config = ValidationConfig(confidence_level=args.confidence, bootstrap_samples=1000, min_trials=args.trials)
    validator = RobustIndependenceValidator(config)
    results = validator.run_validation(n_values=args.n_values, t_values=args.t_values, circuit_type='both')

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
