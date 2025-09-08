"""
detailed_logging.py - Infrastructure for logging FR-SIC process details during evaluation
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

class FRSICDetailedLogger:
    """
    Logger for capturing detailed FR-SIC process information during evaluation.
    Saves self-motivation and neighbor influence data to CSV files.
    """
    
    def __init__(self, log_dir: str = "evaluation_logs"):
        """
        Initialize the detailed logger with CSV file writers.
        
        Args:
            log_dir: Directory to save log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV files and writers
        self.self_motivation_file = None
        self.neighbor_influence_file = None
        self.self_motivation_writer = None
        self.neighbor_influence_writer = None
        
        self._init_csv_files()
    
    def _init_csv_files(self):
        """Initialize CSV files and writers."""
        # Self-motivation log file
        self.self_motivation_file = open(
            self.log_dir / "self_motivation_log.csv", 'w', newline='', encoding='utf-8'
        )
        self.self_motivation_writer = csv.writer(self.self_motivation_file)
        self.self_motivation_writer.writerow([
            'household_id', 'timestep', 'decision_type', 'decision_name',
            'self_motivation_prob', 'period'
        ])
        
        # Neighbor influence log file
        self.neighbor_influence_file = open(
            self.log_dir / "neighbor_influence_log.csv", 'w', newline='', encoding='utf-8'
        )
        self.neighbor_influence_writer = csv.writer(self.neighbor_influence_file)
        self.neighbor_influence_writer.writerow([
            'target_household', 'source_household', 'timestep', 'decision_type', 
            'decision_name', 'influence_prob', 'link_type', 'distance', 'period'
        ])
        
        print(f"Detailed logging initialized. Files saved to: {self.log_dir}")
    
    def log_detailed_breakdown(self, detailed_breakdown: List[Dict[str, Any]], period: str):
        """
        Log detailed breakdown information from FR-SIC process.
        
        Args:
            detailed_breakdown: List of detailed breakdown dictionaries from compute_detailed_activation_probability
            period: 'train' or 'test' to indicate evaluation period
        """
        for breakdown in detailed_breakdown:
            household_id = breakdown['household_id'] + 1
            decision_type = breakdown['decision_type']
            timestep = breakdown['timestep']
            self_activation_prob = breakdown['self_activation_prob']
            neighbor_influences = breakdown.get('neighbor_influences', [])
            
            decision_name = ['vacant', 'repair', 'sell'][decision_type]
            
            # Log self-motivation
            self.self_motivation_writer.writerow([
                household_id, timestep, decision_type, decision_name,
                self_activation_prob, period
            ])
            
            # Log neighbor influences
            for neighbor_info in neighbor_influences:
                neighbor_id = neighbor_info['neighbor_id'] + 1
                influence_prob = neighbor_info['influence_prob']
                link_type = neighbor_info['link_type']
                distance = neighbor_info['distance']
                
                self.neighbor_influence_writer.writerow([
                    household_id, neighbor_id, timestep, decision_type,
                    decision_name, influence_prob, link_type, distance, period
                ])
        
        # Flush to ensure data is written immediately
        self.self_motivation_file.flush()
        self.neighbor_influence_file.flush()
    
    def close(self):
        """Close CSV files and clean up resources."""
        if self.self_motivation_file:
            self.self_motivation_file.close()
        if self.neighbor_influence_file:
            self.neighbor_influence_file.close()
        
        print(f"Detailed logging completed. Files saved to: {self.log_dir}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure files are closed."""
        self.close()