"""
Evaluation metrics module for GenAI Revenue Cycle Management System.
Measures accuracy of extraction, classification, and audit recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st

class GenAIEvaluator:
    """Evaluates the performance of GenAI agents."""
    
    def __init__(self):
        self.metrics = {
            'extraction_accuracy': [],
            'drg_classification_accuracy': [],
            'audit_precision': [],
            'undercoding_detection_rate': [],
            'execution_times_ms': []
        }
    
    def evaluate_extraction(self, extracted: Dict, ground_truth: Dict) -> Dict:
        """Evaluate clinical extraction accuracy."""
        scores = {}
        
        # Compare primary diagnosis
        if 'primary_diagnosis' in extracted and 'primary_diagnosis' in ground_truth:
            scores['diagnosis_match'] = int(
                extracted['primary_diagnosis'].lower() == ground_truth['primary_diagnosis'].lower()
            )
        
        # Compare procedures
        if 'primary_procedure' in extracted and 'primary_procedure' in ground_truth:
            scores['procedure_match'] = int(
                extracted['primary_procedure'].lower() == ground_truth['primary_procedure'].lower()
            )
        
        # Complication detection
        if 'complications' in extracted and 'complications' in ground_truth:
            extracted_set = set(extracted.get('complications', []))
            truth_set = set(ground_truth.get('complications', []))
            if truth_set:
                scores['complication_recall'] = len(extracted_set & truth_set) / len(truth_set)
        
        # Calculate overall score
        if scores:
            scores['overall_accuracy'] = sum(scores.values()) / len(scores)
        
        self.metrics['extraction_accuracy'].append(scores.get('overall_accuracy', 0))
        return scores
    
    def evaluate_drg_classification(self, predicted_drg: str, actual_drg: str) -> float:
        """Evaluate DRG classification accuracy."""
        accuracy = int(predicted_drg == actual_drg)
        self.metrics['drg_classification_accuracy'].append(accuracy)
        return accuracy
    
    def evaluate_audit(self, audit_result: Dict, known_leakage: bool) -> Dict:
        """Evaluate audit detection accuracy."""
        scores = {}
        
        if 'undercoding_detected' in audit_result:
            scores['detection_correct'] = int(
                audit_result['undercoding_detected'] == known_leakage
            )
            self.metrics['undercoding_detection_rate'].append(scores['detection_correct'])
        
        return scores
    
    def add_execution_time(self, time_ms: float):
        """Track execution time."""
        self.metrics['execution_times_ms'].append(time_ms)
    
    def get_summary(self) -> Dict:
        """Get evaluation summary."""
        summary = {}
        
        if self.metrics['extraction_accuracy']:
            summary['avg_extraction_accuracy'] = np.mean(self.metrics['extraction_accuracy'])
        
        if self.metrics['drg_classification_accuracy']:
            summary['drg_classification_accuracy'] = np.mean(self.metrics['drg_classification_accuracy'])
        
        if self.metrics['undercoding_detection_rate']:
            summary['undercoding_detection_rate'] = np.mean(self.metrics['undercoding_detection_rate'])
        
        if self.metrics['execution_times_ms']:
            summary['avg_execution_time_ms'] = np.mean(self.metrics['execution_times_ms'])
            summary['total_audits'] = len(self.metrics['execution_times_ms'])
        
        return summary
    
    def display_metrics(self):
        """Display evaluation metrics in Streamlit."""
        summary = self.get_summary()
        
        if not summary:
            st.info("Run audits to collect evaluation metrics")
            return
        
        st.markdown("### 📊 System Performance Metrics")
        
        cols = st.columns(4)
        
        with cols[0]:
            st.metric(
                "Extraction Accuracy",
                f"{summary.get('avg_extraction_accuracy', 0):.1%}",
                help="Clinical information extraction accuracy"
            )
        
        with cols[1]:
            st.metric(
                "DRG Classification",
                f"{summary.get('drg_classification_accuracy', 0):.1%}",
                help="DRG assignment accuracy"
            )
        
        with cols[2]:
            st.metric(
                "Leak Detection Rate",
                f"{summary.get('undercoding_detection_rate', 0):.1%}",
                help="Revenue leakage detection accuracy"
            )
        
        with cols[3]:
            st.metric(
                "Avg Response Time",
                f"{summary.get('avg_execution_time_ms', 0):.0f}ms",
                help="Average audit execution time"
            )