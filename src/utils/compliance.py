"""Privacy and compliance utilities for medical AI."""

import re
import logging
from typing import Dict, List, Optional, Any, Union
import hashlib
import uuid
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class DeIdentifier:
    """De-identification utilities for medical data."""
    
    def __init__(self):
        """Initialize de-identifier with common patterns."""
        # Common PHI patterns
        self.patterns = {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'mrn': r'\bMRN\s*:?\s*\d+\b',
            'patient_id': r'\bPatient\s*ID\s*:?\s*\d+\b',
            'date': r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            'name': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Basic name pattern
        }
        
        # Replacement mappings
        self.replacements = {
            'ssn': '[REDACTED-SSN]',
            'phone': '[REDACTED-PHONE]',
            'email': '[REDACTED-EMAIL]',
            'mrn': '[REDACTED-MRN]',
            'patient_id': '[REDACTED-PATIENT-ID]',
            'date': '[REDACTED-DATE]',
            'name': '[REDACTED-NAME]',
        }
    
    def deidentify_text(self, text: str, patterns: Optional[List[str]] = None) -> str:
        """De-identify text by replacing PHI patterns.
        
        Args:
            text: Input text to de-identify
            patterns: Specific patterns to apply (if None, applies all)
            
        Returns:
            De-identified text
        """
        if patterns is None:
            patterns = list(self.patterns.keys())
        
        deidentified_text = text
        
        for pattern_name in patterns:
            if pattern_name in self.patterns:
                pattern = self.patterns[pattern_name]
                replacement = self.replacements[pattern_name]
                deidentified_text = re.sub(pattern, replacement, deidentified_text, flags=re.IGNORECASE)
        
        return deidentified_text
    
    def deidentify_dict(self, data: Dict[str, Any], fields_to_redact: Optional[List[str]] = None) -> Dict[str, Any]:
        """De-identify dictionary data.
        
        Args:
            data: Input dictionary
            fields_to_redact: Specific fields to redact
            
        Returns:
            De-identified dictionary
        """
        deidentified_data = data.copy()
        
        # Default fields to redact
        if fields_to_redact is None:
            fields_to_redact = [
                'patient_id', 'patient_name', 'mrn', 'ssn', 'phone', 'email',
                'date_of_birth', 'address', 'zip_code'
            ]
        
        for field in fields_to_redact:
            if field in deidentified_data:
                deidentified_data[field] = f'[REDACTED-{field.upper()}]'
        
        return deidentified_data


class ComplianceLogger:
    """Compliance-aware logging for medical AI systems."""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize compliance logger.
        
        Args:
            log_file: Optional log file path
        """
        self.logger = logging.getLogger('compliance')
        
        if log_file:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Compliance flags
        self.phi_logging_enabled = False
        self.audit_trail_enabled = True
    
    def log_model_prediction(
        self, 
        model_name: str, 
        prediction: Any, 
        confidence: float,
        input_hash: str,
        user_id: Optional[str] = None
    ) -> None:
        """Log model prediction for audit trail.
        
        Args:
            model_name: Name of the model
            prediction: Model prediction
            confidence: Prediction confidence
            input_hash: Hash of input data (no PHI)
            user_id: Optional user identifier
        """
        if not self.audit_trail_enabled:
            return
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'model_prediction',
            'model_name': model_name,
            'prediction': str(prediction),
            'confidence': confidence,
            'input_hash': input_hash,
            'user_id': user_id or 'anonymous'
        }
        
        self.logger.info(f"MODEL_PREDICTION: {json.dumps(log_entry)}")
    
    def log_data_access(
        self, 
        data_type: str, 
        access_type: str,
        user_id: Optional[str] = None
    ) -> None:
        """Log data access for compliance.
        
        Args:
            data_type: Type of data accessed
            access_type: Type of access (read, write, delete)
            user_id: Optional user identifier
        """
        if not self.audit_trail_enabled:
            return
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'data_access',
            'data_type': data_type,
            'access_type': access_type,
            'user_id': user_id or 'anonymous'
        }
        
        self.logger.info(f"DATA_ACCESS: {json.dumps(log_entry)}")
    
    def log_error(self, error_type: str, error_message: str, context: Optional[Dict] = None) -> None:
        """Log errors for compliance monitoring.
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Optional context information
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'error',
            'error_type': error_type,
            'error_message': error_message,
            'context': context or {}
        }
        
        self.logger.error(f"ERROR: {json.dumps(log_entry)}")


class DataAnonymizer:
    """Data anonymization utilities."""
    
    def __init__(self, salt: Optional[str] = None):
        """Initialize data anonymizer.
        
        Args:
            salt: Optional salt for hashing
        """
        self.salt = salt or str(uuid.uuid4())
    
    def anonymize_id(self, identifier: str) -> str:
        """Anonymize an identifier using hashing.
        
        Args:
            identifier: Original identifier
            
        Returns:
            Anonymized identifier
        """
        # Create hash of identifier + salt
        hash_input = f"{identifier}{self.salt}"
        hash_object = hashlib.sha256(hash_input.encode())
        return hash_object.hexdigest()[:16]  # Use first 16 characters
    
    def anonymize_dataset(self, dataset: List[Dict[str, Any]], id_fields: List[str]) -> List[Dict[str, Any]]:
        """Anonymize a dataset by hashing ID fields.
        
        Args:
            dataset: List of data records
            id_fields: Fields containing identifiers to anonymize
            
        Returns:
            Anonymized dataset
        """
        anonymized_dataset = []
        
        for record in dataset:
            anonymized_record = record.copy()
            
            for field in id_fields:
                if field in anonymized_record:
                    anonymized_record[field] = self.anonymize_id(str(anonymized_record[field]))
            
            anonymized_dataset.append(anonymized_record)
        
        return anonymized_dataset


class PrivacyFilter:
    """Privacy filtering for model outputs and logs."""
    
    def __init__(self, deidentifier: Optional[DeIdentifier] = None):
        """Initialize privacy filter.
        
        Args:
            deidentifier: Optional de-identifier instance
        """
        self.deidentifier = deidentifier or DeIdentifier()
        self.phi_detected = False
    
    def filter_output(self, output: Any) -> Any:
        """Filter model output for PHI.
        
        Args:
            output: Model output to filter
            
        Returns:
            Filtered output
        """
        if isinstance(output, str):
            return self.deidentifier.deidentify_text(output)
        elif isinstance(output, dict):
            return self.deidentifier.deidentify_dict(output)
        elif isinstance(output, list):
            return [self.filter_output(item) for item in output]
        else:
            return output
    
    def check_for_phi(self, text: str) -> bool:
        """Check if text contains PHI.
        
        Args:
            text: Text to check
            
        Returns:
            True if PHI detected
        """
        phi_detected = False
        
        for pattern_name, pattern in self.deidentifier.patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                phi_detected = True
                break
        
        self.phi_detected = phi_detected
        return phi_detected


class ComplianceChecker:
    """Compliance checking utilities."""
    
    def __init__(self):
        """Initialize compliance checker."""
        self.violations = []
    
    def check_data_retention(self, data_age_days: int, max_retention_days: int = 365) -> bool:
        """Check data retention compliance.
        
        Args:
            data_age_days: Age of data in days
            max_retention_days: Maximum retention period
            
        Returns:
            True if compliant
        """
        if data_age_days > max_retention_days:
            self.violations.append(f"Data retention violation: {data_age_days} days > {max_retention_days} days")
            return False
        return True
    
    def check_model_bias(self, performance_by_group: Dict[str, float], threshold: float = 0.1) -> bool:
        """Check for model bias across groups.
        
        Args:
            performance_by_group: Performance metrics by group
            threshold: Maximum allowed difference
            
        Returns:
            True if compliant
        """
        if len(performance_by_group) < 2:
            return True
        
        values = list(performance_by_group.values())
        max_diff = max(values) - min(values)
        
        if max_diff > threshold:
            self.violations.append(f"Model bias detected: {max_diff:.3f} > {threshold}")
            return False
        return True
    
    def check_privacy_compliance(self, has_phi: bool) -> bool:
        """Check privacy compliance.
        
        Args:
            has_phi: Whether PHI is present
            
        Returns:
            True if compliant
        """
        if has_phi:
            self.violations.append("PHI detected in non-compliant context")
            return False
        return True
    
    def get_violations(self) -> List[str]:
        """Get list of compliance violations.
        
        Returns:
            List of violation descriptions
        """
        return self.violations.copy()
    
    def clear_violations(self) -> None:
        """Clear violation list."""
        self.violations.clear()


# Utility functions for easy integration
def create_compliance_suite() -> Dict[str, Any]:
    """Create a complete compliance suite.
    
    Returns:
        Dictionary containing all compliance utilities
    """
    return {
        'deidentifier': DeIdentifier(),
        'logger': ComplianceLogger(),
        'anonymizer': DataAnonymizer(),
        'filter': PrivacyFilter(),
        'checker': ComplianceChecker()
    }


def add_compliance_hooks(model, compliance_suite: Dict[str, Any]) -> None:
    """Add compliance hooks to a model.
    
    Args:
        model: Model to add hooks to
        compliance_suite: Compliance utilities
    """
    original_forward = model.forward
    
    def compliance_forward(*args, **kwargs):
        # Log model usage
        compliance_suite['logger'].log_model_prediction(
            model_name=model.__class__.__name__,
            prediction=None,  # Will be filled after prediction
            confidence=0.0,   # Will be filled after prediction
            input_hash=hash(str(args))  # Simple hash of inputs
        )
        
        # Run original forward pass
        result = original_forward(*args, **kwargs)
        
        # Filter output for PHI
        filtered_result = compliance_suite['filter'].filter_output(result)
        
        return filtered_result
    
    model.forward = compliance_forward
