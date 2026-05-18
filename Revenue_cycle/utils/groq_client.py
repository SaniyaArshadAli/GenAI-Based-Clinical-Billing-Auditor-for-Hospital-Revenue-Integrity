"""
GenAI Multi-Agent Revenue Cycle Management System
Agents: Clinical Extractor → DRG Classifier → Cost Predictor → Billing Auditor → Explanation Generator
"""

import json
import time
from typing import Dict, Any, List, Optional, Tuple
from groq import Groq
import streamlit as st
import pandas as pd

class ClinicalExtractorAgent:
    """Agent 1: Extracts structured clinical information from unstructured text."""
    
    def __init__(self, client: Groq):
        self.client = client
        self.model = "llama-3.1-8b-instant"
    
    def extract(self, clinical_text: str) -> Dict[str, Any]:
        """Extract diagnosis, procedures, complications, medications from clinical notes."""
        system_prompt = """You are a Clinical Information Extraction AI Agent deployed in a hospital revenue cycle system.

Extract structured clinical information from the discharge summary. Return ONLY valid JSON.

{
  "primary_diagnosis": "string",
  "secondary_diagnoses": ["list"],
  "primary_procedure": "string",
  "secondary_procedures": ["list"],
  "complications": ["list of complications mentioned"],
  "comorbidities": ["list of comorbid conditions"],
  "medications_administered": ["list"],
  "icu_stay_days": number or null,
  "ventilator_hours": number or null,
  "blood_transfusions": number or null,
  "severity_level": "minor/moderate/major/extreme",
  "key_clinical_indicators": ["list of important clinical findings"]
}"""

        return self._call_api(system_prompt, f"Extract clinical information from:\n\n{clinical_text}")
    
    def _call_api(self, system_prompt: str, user_prompt: str) -> Dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=800
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e), "primary_diagnosis": "Unknown"}


class DRGClassifierAgent:
    """Agent 2: Classifies DRG based on clinical evidence with reasoning."""
    
    def __init__(self, client: Groq):
        self.client = client
        self.model = "llama-3.1-8b-instant"
    
    def classify(self, clinical_extract: Dict, available_drgs: List[Dict]) -> Dict[str, Any]:
        """Classify to appropriate DRG with clinical reasoning."""
        system_prompt = """You are a DRG Coding AI Agent. Based on extracted clinical information and available DRG options, 
assign the most appropriate DRG code and explain your reasoning.

Return ONLY valid JSON:
{
  "assigned_drg": "XXX",
  "drg_description": "string",
  "confidence": 0.95,
  "reasoning": "Detailed clinical reasoning for this DRG assignment",
  "mcc_present": true/false,
  "cc_present": true/false,
  "expected_los_days": number,
  "expected_services": ["list of services typically billed for this DRG"],
  "risk_factors": ["list of factors affecting cost/LOS"]
}"""

        user_prompt = f"""
CLINICAL EXTRACT:
{json.dumps(clinical_extract, indent=2)}

AVAILABLE DRG OPTIONS:
{json.dumps(available_drgs, indent=2)}

Assign the most appropriate DRG based on clinical evidence.
"""
        return self._call_api(system_prompt, user_prompt)
    
    def _call_api(self, system_prompt: str, user_prompt: str) -> Dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=800
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e), "assigned_drg": "000"}


class BillingAuditorAgent:
    """Agent 3: Audits billing by comparing expected vs actual charges."""
    
    def __init__(self, client: Groq):
        self.client = client
        self.model = "llama-3.1-8b-instant"
    
    def audit(self, clinical_extract: Dict, assigned_drg: Dict, 
              billed_drg: Dict, actual_charges: float) -> Dict[str, Any]:
        """Compare expected billing with actual billing to detect leakage."""
        system_prompt = """You are a Medical Billing Auditor AI Agent. Compare what SHOULD be billed vs what WAS billed.

Identify:
1. Undercoding (billed DRG lower than clinically justified)
2. Missing services/charges
3. Documentation gaps
4. Revenue leakage estimate

Return ONLY valid JSON:
{
  "undercoding_detected": true/false,
  "severity": "none/minor/moderate/critical",
  "missing_services": ["list of services that should have been billed"],
  "documentation_gaps": ["list of documentation issues"],
  "estimated_revenue_loss": number,
  "loss_percentage": number,
  "recommended_actions": ["list of corrective actions"],
  "audit_summary": "comprehensive audit summary"
}"""

        user_prompt = f"""
CLINICAL EXTRACT:
{json.dumps(clinical_extract, indent=2)}

AI-ASSIGNED DRG:
{json.dumps(assigned_drg, indent=2)}

HOSPITAL BILLED DRG:
{json.dumps(billed_drg, indent=2)}

ACTUAL CHARGES: ${actual_charges:,.2f}

Analyze for billing discrepancies and revenue leakage.
"""
        return self._call_api(system_prompt, user_prompt)
    
    def _call_api(self, system_prompt: str, user_prompt: str) -> Dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=800
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e), "undercoding_detected": False}


class ExplanationGeneratorAgent:
    """Agent 4: Generates human-readable explanations for non-technical stakeholders."""
    
    def __init__(self, client: Groq):
        self.client = client
        self.model = "llama-3.1-8b-instant"
    
    def generate(self, clinical_extract: Dict, drg_result: Dict, 
                 audit_result: Dict, financial_impact: Dict) -> str:
        """Generate comprehensive explanation of findings."""
        system_prompt = """You are a Healthcare Revenue Cycle Communication AI. 
Generate a clear, professional explanation of audit findings for hospital administration.

Include:
1. Clinical summary
2. DRG classification reasoning
3. Billing discrepancies found
4. Financial impact
5. Recommended actions

Format with clear sections. Be professional but accessible.
"""
        user_prompt = f"""
CLINICAL SUMMARY:
{json.dumps(clinical_extract, indent=2)}

DRG CLASSIFICATION:
{json.dumps(drg_result, indent=2)}

AUDIT FINDINGS:
{json.dumps(audit_result, indent=2)}

FINANCIAL IMPACT:
{json.dumps(financial_impact, indent=2)}

Generate a comprehensive explanation for hospital revenue cycle team.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating explanation: {str(e)}"


class GroqRevenueAuditor:
    """Orchestrator: Coordinates all GenAI agents for complete revenue audit."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"
        
        # Initialize agents
        self.extractor = ClinicalExtractorAgent(self.client)
        self.classifier = DRGClassifierAgent(self.client)
        self.auditor = BillingAuditorAgent(self.client)
        self.explainer = ExplanationGeneratorAgent(self.client)
        
        self.max_retries = 3
        self.retry_delay = 1
        self.retry_count = 0
    
    def validate_api_key(self) -> bool:
        """Validate the Groq API key."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Validate connection for GenAI Revenue Cycle Management System"}],
                max_tokens=20
            )
            return True
        except Exception as e:
            error_str = str(e)
            if "401" in error_str:
                st.error("❌ Invalid Groq API Key")
            elif "429" in error_str:
                st.error("⚠️ Rate limit exceeded")
            else:
                st.error(f"❌ API Error: {error_str}")
            return False
    
    def find_sibling_drgs(self, drg_code: str, df_ipps: pd.DataFrame, range_limit: int = 5) -> List[Dict]:
        """Find related DRGs for classification."""
        try:
            drg_num = int(drg_code)
            
            # Find weight column
            weight_col = None
            for col in ['Weights - Before Cap', 'Weights', 'WEIGHTS']:
                if col in df_ipps.columns:
                    weight_col = col
                    break
            
            if weight_col is None:
                return []
            
            # Find title column
            title_col = None
            for col in ['MS-DRG Title', 'Title']:
                if col in df_ipps.columns:
                    title_col = col
                    break
            
            mask = (
                (df_ipps['MS-DRG'].astype(int) >= drg_num - range_limit) & 
                (df_ipps['MS-DRG'].astype(int) <= drg_num + range_limit)
            )
            siblings_df = df_ipps[mask].copy()
            siblings_df = siblings_df.sort_values(weight_col, ascending=False)
            
            siblings = []
            for _, row in siblings_df.head(15).iterrows():
                sibling = {
                    'MS-DRG': str(row['MS-DRG']),
                    'MS-DRG Title': str(row.get(title_col, 'Unknown')) if title_col else 'Unknown',
                    'Weights - Before Cap': float(row[weight_col])
                }
                siblings.append(sibling)
            
            return siblings
        except Exception as e:
            st.warning(f"Could not find sibling DRGs: {e}")
            return []
    
    def run_full_audit(self, clinical_text: str, billed_drg_info: Dict, 
                       actual_charges: float, df_ipps: pd.DataFrame) -> Dict[str, Any]:
        """Execute complete GenAI audit pipeline."""
        
        results = {
            'extraction': None,
            'classification': None,
            'audit': None,
            'explanation': None,
            'execution_time_ms': 0,
            'errors': []
        }
        
        start_time = time.time()
    
        # Step 1: Clinical Extraction
        with st.spinner("🧠 Agent 1/4: Extracting clinical information from notes..."):
            extraction = self.extractor.extract(clinical_text)
            if 'error' in extraction:
                results['errors'].append(f"Extraction error: {extraction['error']}")
            results['extraction'] = extraction
        
        # Step 2: Find sibling DRGs for classification
        sibling_drgs = self.find_sibling_drgs(billed_drg_info['DRG'], df_ipps)
        
        # Step 3: DRG Classification
        with st.spinner("🏥 Agent 2/4: Classifying DRG with clinical reasoning..."):
            classification = self.classifier.classify(extraction, sibling_drgs)
            if 'error' in classification:
                results['errors'].append(f"Classification error: {classification['error']}")
            results['classification'] = classification
        
        # Step 4: Billing Audit
        with st.spinner("💰 Agent 3/4: Auditing billing for discrepancies..."):
            audit = self.auditor.audit(extraction, classification, billed_drg_info, actual_charges)
            if 'error' in audit:
                results['errors'].append(f"Audit error: {audit['error']}")
            results['audit'] = audit

        # Step 4.5: Get expected services for DRG
        with st.spinner("📋 Agent 3.5/4: Mapping expected services for DRG..."):
            expected_services = self.get_expected_services_for_drg(
                classification.get('assigned_drg', billed_drg_info['DRG']),
                df_ipps
            )
            results['expected_services'] = expected_services
        
        # Step 5: Financial Impact Calculation
        financial_impact = self.calculate_financial_impact(
            billed_drg_info.get('Weight', 0),
            str(classification.get('assigned_drg', '000')),
            df_ipps,
            actual_charges
        )
        results['financial_impact'] = financial_impact
        
        # Step 6: Generate Explanation
        with st.spinner("📝 Agent 4/4: Generating comprehensive explanation..."):
            explanation = self.explainer.generate(
                extraction, classification, audit, financial_impact
            )
            results['explanation'] = explanation
        
        results['execution_time_ms'] = (time.time() - start_time) * 1000
        
        return results

    def get_expected_services_for_drg(self, drg_code: str, df_ipps: pd.DataFrame) -> Dict:
        """Get expected services for a DRG using Groq reasoning."""
        
        # Find the DRG description
        drg_row = df_ipps[df_ipps['MS-DRG'] == drg_code]
        drg_desc = ""
        if not drg_row.empty:
            title_col = None
            for col in ['MS-DRG Title', 'Title']:
                if col in drg_row.columns:
                    title_col = col
                    break
            if title_col:
                drg_desc = str(drg_row.iloc[0][title_col])
        
        system_prompt = """You are a Medical Coding Expert. For a given DRG code and description,
    list ALL services, procedures, medications, and resources that should typically be billed.

    Return ONLY valid JSON:
    {
      "expected_procedures": ["list of procedures"],
      "expected_medications": ["list of medications"],
      "expected_consumables": ["list of consumables/supplies"],
      "expected_imaging": ["list of imaging studies"],
      "expected_labs": ["list of lab tests"],
      "expected_room_charges": ["ICU days", "OT charges", etc],
      "total_expected_items": number
    }"""

        user_prompt = f"DRG Code: {drg_code}\nDRG Description: {drg_desc}\n\nList all expected billable items for this DRG."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=800
            )
            return json.loads(response.choices[0].message.content)
        except:
            return {"expected_procedures": [], "expected_medications": [], "expected_consumables": [],
                    "expected_imaging": [], "expected_labs": [], "expected_room_charges": [], "total_expected_items": 0}
        
    
    def calculate_financial_impact(self, billed_weight: float, assigned_drg: str, 
                                   df_ipps: pd.DataFrame, actual_charges: float) -> Dict:
        """Calculate financial impact of DRG mismatch."""
        try:
            # Find assigned DRG weight
            rec_row = df_ipps[df_ipps['MS-DRG'] == assigned_drg]
            if not rec_row.empty:
                weight_col = None
                for col in ['Weights - Before Cap', 'Weights']:
                    if col in df_ipps.columns:
                        weight_col = col
                        break
                
                if weight_col:
                    new_weight = float(rec_row.iloc[0][weight_col])
                    weight_delta = new_weight - billed_weight
                    
                    return {
                        "billed_weight": round(billed_weight, 4),
                        "recommended_weight": round(new_weight, 4),
                        "weight_delta": round(weight_delta, 4),
                        "percentage_difference": round((weight_delta / billed_weight) * 100, 1) if billed_weight > 0 else 0,
                        "estimated_recovery_per_case": round(weight_delta * 6500, 2),
                        "actual_charges": actual_charges,
                        "potential_missed_revenue": round(max(0, weight_delta * 6500), 2)
                    }
        except:
            pass
        
        return {
            "billed_weight": billed_weight,
            "recommended_weight": billed_weight,
            "weight_delta": 0,
            "percentage_difference": 0,
            "estimated_recovery_per_case": 0,
            "actual_charges": actual_charges,
            "potential_missed_revenue": 0
        }


# Backward compatibility wrapper
def run_reconciliation_agent(api_key, clinical_notes, billed_drg_info, sibling_drgs):
    """Legacy wrapper for backward compatibility."""
    auditor = GroqRevenueAuditor(api_key)
    # Simplified call for backward compatibility
    return auditor.extractor.extract(clinical_notes)