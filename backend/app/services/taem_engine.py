"""
TAEM (Technical Assurance and Evaluation Model) Engine
======================================================

Rule-based evaluation engine for assessing HS2 asset readiness.
Evaluates deliverables, costs, certificates, and schedule compliance.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.models.hs2 import (
    HS2Asset,
    HS2Deliverable,
    HS2Cost,
    HS2Certificate,
    HS2Rule,
    HS2Evaluation,
)


class TAEMEngine:
    """
    TAEM Rule Evaluation Engine
    
    Evaluates assets against TAEM rules and calculates readiness scores.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.rules = []
    
    async def load_rules(self):
        """Load active TAEM rules from database."""
        query = select(HS2Rule).where(HS2Rule.is_active == True).order_by(HS2Rule.rule_code)
        result = await self.db.execute(query)
        self.rules = result.scalars().all()
        
        if not self.rules:
            logger.warning("No active TAEM rules found - creating default rules")
            await self.create_default_rules()
            self.rules = (await self.db.execute(query)).scalars().all()
        
        logger.info(f"Loaded {len(self.rules)} active TAEM rules")
    
    async def create_default_rules(self):
        """Create default TAEM rules."""
        default_rules = [
            {
                "rule_code": "TAEM-001",
                "rule_name": "Critical Deliverables Completeness",
                "rule_description": "All critical deliverables must be submitted and approved",
                "rule_category": "Deliverables",
                "severity": "Critical",
                "weight": 0.30,
                "threshold_value": 100.0,
                "rule_logic": {"check": "critical_deliverables_complete"},
            },
            {
                "rule_code": "TAEM-002",
                "rule_name": "Major Deliverables Completeness",
                "rule_description": "At least 90% of major deliverables must be submitted",
                "rule_category": "Deliverables",
                "severity": "Major",
                "weight": 0.15,
                "threshold_value": 90.0,
                "rule_logic": {"check": "major_deliverables_completion"},
            },
            {
                "rule_code": "TAEM-003",
                "rule_name": "No Overdue Critical Deliverables",
                "rule_description": "No critical deliverables should be overdue",
                "rule_category": "Deliverables",
                "severity": "Critical",
                "weight": 0.20,
                "threshold_value": 0.0,
                "rule_logic": {"check": "overdue_critical_deliverables"},
            },
            {
                "rule_code": "TAEM-004",
                "rule_name": "Cost Variance Within Tolerance",
                "rule_description": "Cost variance must be within ±10%",
                "rule_category": "Costs",
                "severity": "Major",
                "weight": 0.15,
                "threshold_value": 10.0,
                "rule_logic": {"check": "cost_variance_tolerance"},
            },
            {
                "rule_code": "TAEM-005",
                "rule_name": "All Certificates Valid",
                "rule_description": "All required certificates must be valid and not expired",
                "rule_category": "Certificates",
                "severity": "Critical",
                "weight": 0.15,
                "threshold_value": 100.0,
                "rule_logic": {"check": "certificates_valid"},
            },
            {
                "rule_code": "TAEM-006",
                "rule_name": "No Expiring Certificates",
                "rule_description": "No certificates should expire within 30 days",
                "rule_category": "Certificates",
                "severity": "Major",
                "weight": 0.05,
                "threshold_value": 0.0,
                "rule_logic": {"check": "certificates_expiring_soon"},
            },
        ]
        
        for rule_data in default_rules:
            rule = HS2Rule(**rule_data)
            self.db.add(rule)
        
        await self.db.commit()
        logger.info(f"Created {len(default_rules)} default TAEM rules")
    
    async def evaluate_asset(self, asset_id: UUID, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Evaluate a single asset against all TAEM rules.
        
        Args:
            asset_id: Asset UUID to evaluate
            force_refresh: Force re-evaluation even if recent evaluation exists
        
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Evaluating asset {asset_id}")
        
        # Load rules if not already loaded
        if not self.rules:
            await self.load_rules()
        
        # Get asset
        asset_query = select(HS2Asset).where(HS2Asset.id == asset_id)
        asset_result = await self.db.execute(asset_query)
        asset = asset_result.scalar_one_or_none()
        
        if not asset:
            raise ValueError(f"Asset with ID {asset_id} not found")
        
        # Check for recent evaluation
        if not force_refresh:
            recent_eval_query = select(HS2Evaluation).where(
                HS2Evaluation.asset_id == asset_id,
                HS2Evaluation.evaluation_date >= datetime.utcnow() - timedelta(hours=1)
            ).order_by(HS2Evaluation.evaluation_date.desc()).limit(1)
            
            recent_eval_result = await self.db.execute(recent_eval_query)
            recent_eval = recent_eval_result.scalar_one_or_none()
            
            if recent_eval:
                logger.info(f"Using recent evaluation from {recent_eval.evaluation_date}")
                return {
                    "overall_score": recent_eval.overall_score,
                    "readiness_status": recent_eval.readiness_status,
                    "rule_results": recent_eval.rule_results,
                }
        
        # Get related data
        deliverables = await self._get_asset_deliverables(asset_id)
        costs = await self._get_asset_costs(asset_id)
        certificates = await self._get_asset_certificates(asset_id)
        
        # Evaluate each rule
        rule_results = []
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for rule in self.rules:
            result = await self._evaluate_rule(rule, asset, deliverables, costs, certificates)
            rule_results.append(result)
            
            total_weighted_score += result["weighted_score"]
            total_weight += result["weight"]
        
        # Calculate overall score
        overall_score = (total_weighted_score / total_weight * 100) if total_weight > 0 else 0.0
        
        # Determine readiness status
        readiness_status = self._determine_readiness_status(overall_score, rule_results)
        
        # Count pass/fail
        rules_passed = sum(1 for r in rule_results if r["status"] == "Pass")
        rules_failed = sum(1 for r in rule_results if r["status"] == "Fail")
        
        # Save evaluation results
        evaluation = HS2Evaluation(
            asset_id=asset_id,
            evaluation_date=datetime.utcnow(),
            overall_score=overall_score,
            readiness_status=readiness_status,
            rules_evaluated=len(rule_results),
            rules_passed=rules_passed,
            rules_failed=rules_failed,
            rule_results=rule_results,
            evaluation_trigger="Manual",
            evaluator="TAEM Engine",
        )
        
        self.db.add(evaluation)
        
        # Update asset
        asset.taem_evaluation_score = overall_score
        asset.readiness_status = readiness_status
        asset.last_evaluation_date = datetime.utcnow()
        
        await self.db.commit()
        
        logger.info(f"Asset {asset.asset_id} evaluated: Score={overall_score:.2f}, Status={readiness_status}")
        
        return {
            "overall_score": overall_score,
            "readiness_status": readiness_status,
            "rule_results": rule_results,
            "rules_passed": rules_passed,
            "rules_failed": rules_failed,
        }
    
    async def _get_asset_deliverables(self, asset_id: UUID) -> List[HS2Deliverable]:
        """Get all deliverables for an asset."""
        query = select(HS2Deliverable).where(HS2Deliverable.asset_id == asset_id)
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def _get_asset_costs(self, asset_id: UUID) -> List[HS2Cost]:
        """Get all costs for an asset."""
        query = select(HS2Cost).where(HS2Cost.asset_id == asset_id)
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def _get_asset_certificates(self, asset_id: UUID) -> List[HS2Certificate]:
        """Get all certificates for an asset."""
        query = select(HS2Certificate).where(HS2Certificate.asset_id == asset_id)
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def _evaluate_rule(
        self,
        rule: HS2Rule,
        asset: HS2Asset,
        deliverables: List[HS2Deliverable],
        costs: List[HS2Cost],
        certificates: List[HS2Certificate],
    ) -> Dict[str, Any]:
        """Evaluate a single rule against asset data."""
        
        rule_logic = rule.rule_logic or {}
        check_type = rule_logic.get("check", "")
        
        # Initialize result
        result = {
            "rule_code": rule.rule_code,
            "rule_name": rule.rule_name,
            "status": "Pass",
            "score": 100.0,
            "weight": rule.weight,
            "weighted_score": 0.0,
            "message": "",
            "details": {},
        }
        
        try:
            # Evaluate based on rule type
            if check_type == "critical_deliverables_complete":
                result = self._check_critical_deliverables(deliverables, rule)
            elif check_type == "major_deliverables_completion":
                result = self._check_major_deliverables(deliverables, rule)
            elif check_type == "overdue_critical_deliverables":
                result = self._check_overdue_critical_deliverables(deliverables, rule)
            elif check_type == "cost_variance_tolerance":
                result = self._check_cost_variance(costs, rule)
            elif check_type == "certificates_valid":
                result = self._check_certificates_valid(certificates, rule)
            elif check_type == "certificates_expiring_soon":
                result = self._check_certificates_expiring(certificates, rule)
            else:
                result["status"] = "Warning"
                result["message"] = f"Unknown rule check type: {check_type}"
            
            # Calculate weighted score
            result["weighted_score"] = result["score"] / 100.0 * result["weight"]
            
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.rule_code}: {str(e)}")
            result["status"] = "Error"
            result["message"] = f"Evaluation error: {str(e)}"
        
        return result
    
    def _check_critical_deliverables(self, deliverables: List[HS2Deliverable], rule: HS2Rule) -> Dict[str, Any]:
        """Check if all critical deliverables are complete."""
        critical_delivs = [d for d in deliverables if d.priority == "Critical"]
        
        if not critical_delivs:
            return {
                "rule_code": rule.rule_code,
                "rule_name": rule.rule_name,
                "status": "Warning",
                "score": 100.0,
                "weight": rule.weight,
                "message": "No critical deliverables found",
                "details": {"total": 0, "approved": 0},
            }
        
        approved = sum(1 for d in critical_delivs if d.status == "Approved")
        completion_pct = (approved / len(critical_delivs)) * 100
        
        status = "Pass" if completion_pct >= rule.threshold_value else "Fail"
        score = completion_pct
        
        return {
            "rule_code": rule.rule_code,
            "rule_name": rule.rule_name,
            "status": status,
            "score": score,
            "weight": rule.weight,
            "message": f"{approved}/{len(critical_delivs)} critical deliverables approved ({completion_pct:.1f}%)",
            "details": {"total": len(critical_delivs), "approved": approved, "completion_pct": completion_pct},
        }
    
    def _check_major_deliverables(self, deliverables: List[HS2Deliverable], rule: HS2Rule) -> Dict[str, Any]:
        """Check major deliverables completion rate."""
        major_delivs = [d for d in deliverables if d.priority == "Major"]
        
        if not major_delivs:
            return {
                "rule_code": rule.rule_code,
                "rule_name": rule.rule_name,
                "status": "Pass",
                "score": 100.0,
                "weight": rule.weight,
                "message": "No major deliverables found",
                "details": {"total": 0, "submitted": 0},
            }
        
        submitted = sum(1 for d in major_delivs if d.status in ["Submitted", "Approved"])
        completion_pct = (submitted / len(major_delivs)) * 100
        
        status = "Pass" if completion_pct >= rule.threshold_value else "Fail"
        score = completion_pct
        
        return {
            "rule_code": rule.rule_code,
            "rule_name": rule.rule_name,
            "status": status,
            "score": score,
            "weight": rule.weight,
            "message": f"{submitted}/{len(major_delivs)} major deliverables submitted ({completion_pct:.1f}%)",
            "details": {"total": len(major_delivs), "submitted": submitted, "completion_pct": completion_pct},
        }
    
    def _check_overdue_critical_deliverables(self, deliverables: List[HS2Deliverable], rule: HS2Rule) -> Dict[str, Any]:
        """Check for overdue critical deliverables."""
        critical_delivs = [d for d in deliverables if d.priority == "Critical"]
        overdue = [d for d in critical_delivs if d.status == "Overdue"]
        
        overdue_count = len(overdue)
        status = "Pass" if overdue_count == 0 else "Fail"
        score = 100.0 if overdue_count == 0 else max(0, 100 - (overdue_count * 20))
        
        return {
            "rule_code": rule.rule_code,
            "rule_name": rule.rule_name,
            "status": status,
            "score": score,
            "weight": rule.weight,
            "message": f"{overdue_count} critical deliverables overdue" if overdue_count > 0 else "No overdue critical deliverables",
            "details": {"total_critical": len(critical_delivs), "overdue": overdue_count},
        }
    
    def _check_cost_variance(self, costs: List[HS2Cost], rule: HS2Rule) -> Dict[str, Any]:
        """Check cost variance within tolerance."""
        if not costs:
            return {
                "rule_code": rule.rule_code,
                "rule_name": rule.rule_name,
                "status": "Warning",
                "score": 100.0,
                "weight": rule.weight,
                "message": "No cost data found",
                "details": {},
            }
        
        # Use the first cost record (one per asset in our model)
        cost = costs[0]
        variance_pct = abs(cost.variance_pct)
        
        status = "Pass" if variance_pct <= rule.threshold_value else "Fail"
        score = max(0, 100 - ((variance_pct - rule.threshold_value) * 5)) if variance_pct > rule.threshold_value else 100.0
        
        return {
            "rule_code": rule.rule_code,
            "rule_name": rule.rule_name,
            "status": status,
            "score": score,
            "weight": rule.weight,
            "message": f"Cost variance: {cost.variance_pct:+.1f}% ({'within' if status == 'Pass' else 'exceeds'} ±{rule.threshold_value:.0f}% tolerance)",
            "details": {"variance_pct": float(cost.variance_pct), "threshold": rule.threshold_value},
        }
    
    def _check_certificates_valid(self, certificates: List[HS2Certificate], rule: HS2Rule) -> Dict[str, Any]:
        """Check all certificates are valid."""
        if not certificates:
            return {
                "rule_code": rule.rule_code,
                "rule_name": rule.rule_name,
                "status": "Warning",
                "score": 100.0,
                "weight": rule.weight,
                "message": "No certificates found",
                "details": {"total": 0, "valid": 0},
            }
        
        valid = sum(1 for c in certificates if c.status == "Valid")
        valid_pct = (valid / len(certificates)) * 100
        
        status = "Pass" if valid_pct >= rule.threshold_value else "Fail"
        score = valid_pct
        
        return {
            "rule_code": rule.rule_code,
            "rule_name": rule.rule_name,
            "status": status,
            "score": score,
            "weight": rule.weight,
            "message": f"{valid}/{len(certificates)} certificates valid ({valid_pct:.1f}%)",
            "details": {"total": len(certificates), "valid": valid, "valid_pct": valid_pct},
        }
    
    def _check_certificates_expiring(self, certificates: List[HS2Certificate], rule: HS2Rule) -> Dict[str, Any]:
        """Check for certificates expiring soon."""
        expiring = [c for c in certificates if c.status == "Expiring Soon"]
        
        expiring_count = len(expiring)
        status = "Pass" if expiring_count == 0 else "Fail"
        score = 100.0 if expiring_count == 0 else max(0, 100 - (expiring_count * 15))
        
        return {
            "rule_code": rule.rule_code,
            "rule_name": rule.rule_name,
            "status": status,
            "score": score,
            "weight": rule.weight,
            "message": f"{expiring_count} certificates expiring within 30 days" if expiring_count > 0 else "No certificates expiring soon",
            "details": {"total": len(certificates), "expiring_soon": expiring_count},
        }
    
    def _determine_readiness_status(self, overall_score: float, rule_results: List[Dict[str, Any]]) -> str:
        """Determine overall readiness status based on score and rule results."""
        
        # Check for any failed critical rules
        critical_failures = [
            r for r in rule_results 
            if r["status"] == "Fail" and any(
                rule.rule_code == r["rule_code"] and rule.severity == "Critical"
                for rule in self.rules
            )
        ]
        
        if critical_failures:
            return "Not Ready"
        
        # Score-based classification
        if overall_score >= 85:
            return "Ready"
        elif overall_score >= 65:
            return "At Risk"
        else:
            return "Not Ready"
