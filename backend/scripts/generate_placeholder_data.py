"""
HS2 Placeholder Data Generator
==============================

Generates realistic synthetic data for 50 HS2 assets with coherent relationships
between assets, deliverables, costs, and certificates.

Distribution:
- 20% Ready (10 assets)
- 50% Not Ready (25 assets)
- 30% At Risk (15 assets)

Asset Types:
- 30 Viaducts (VA-001 to VA-030)
- 15 Bridges (BR-001 to BR-015)
- 3 Tunnels (TN-001 to TN-003)
- 2 OLE Masts (OLE-001 to OLE-002)
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from decimal import Decimal

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ==================== Configuration ====================

# Asset distribution
ASSET_CONFIG = {
    "Viaduct": {"count": 30, "prefix": "VA", "budget_range": (3_000_000, 8_000_000)},
    "Bridge": {"count": 15, "prefix": "BR", "budget_range": (2_000_000, 6_000_000)},
    "Tunnel": {"count": 3, "prefix": "TN", "budget_range": (15_000_000, 25_000_000)},
    "OLE Mast": {"count": 2, "prefix": "OLE", "budget_range": (500_000, 1_000_000)},
}

# Status distribution (must sum to 50)
STATUS_DISTRIBUTION = {
    "Ready": 10,
    "Not Ready": 25,
    "At Risk": 15,
}

# Route sections
ROUTE_SECTIONS = [
    "London-Euston",
    "Old Oak Common",
    "Acton",
    "Northolt",
    "Denham",
]

# Contractors
CONTRACTORS = [
    "JV-Alpha",
    "JV-Bravo",
    "JV-Charlie",
]

# Deliverable types with priorities
DELIVERABLE_TYPES = [
    {"type": "Design Certificate", "priority": "Critical", "duration_days": 120},
    {"type": "Assurance Sign-off", "priority": "Critical", "duration_days": 90},
    {"type": "Test Report - Concrete", "priority": "Critical", "duration_days": 60},
    {"type": "Test Report - Welding", "priority": "Critical", "duration_days": 45},
    {"type": "QA Inspection Report", "priority": "Major", "duration_days": 30},
    {"type": "Method Statement", "priority": "Minor", "duration_days": 15},
    {"type": "Risk Assessment", "priority": "Major", "duration_days": 30},
    {"type": "Environmental Impact", "priority": "Minor", "duration_days": 45},
]

# Certificate types
CERTIFICATE_TYPES = [
    "Design Certificate",
    "Welding Qualification",
    "Concrete Test Certificate",
    "NDT Inspection Certificate",
    "Quality Assurance Certificate",
]

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "placeholder_data"

print(f"Placeholder Data Generator for HS2 Assurance Intelligence Demonstrator")
print(f"=" * 80)
print(f"Random Seed: {RANDOM_SEED}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"\nAsset Distribution:")
for asset_type, config in ASSET_CONFIG.items():
    print(f"  {asset_type}: {config['count']} assets")
print(f"\nStatus Distribution:")
for status, count in STATUS_DISTRIBUTION.items():
    print(f"  {status}: {count} assets ({count * 2}%)")
print(f"=" * 80)


# ==================== Helper Functions ====================

def generate_location_text(asset_type: str, route_section: str, sequence: int) -> str:
    """Generate realistic location description."""
    locations = {
        "London-Euston": [
            "Between Euston Station and Camden Junction",
            "Adjacent to Regent's Canal crossing",
            "Near Camden Road overbridge",
            "Parallel to West Coast Main Line",
        ],
        "Old Oak Common": [
            "Adjacent to Old Oak Common depot site",
            "Near Grand Union Canal",
            "Between Hythe Road and Atlas Road",
            "Parallel to Great Western Main Line",
        ],
        "Acton": [
            "Between Acton Main Line and North Acton",
            "Adjacent to Central Line interchange",
            "Near Victoria Road overbridge",
            "Crossing Uxbridge Road",
        ],
        "Northolt": [
            "Between Northolt Junction and Greenford",
            "Adjacent to A40 Western Avenue",
            "Near RAF Northolt boundary",
            "Crossing Grand Union Canal branch",
        ],
        "Denham": [
            "Between Denham and Gerrards Cross",
            "Adjacent to Chiltern Line",
            "Near M25 interchange",
            "Crossing River Misbourne",
        ],
    }
    
    base_locations = locations.get(route_section, ["Generic location"])
    location = base_locations[sequence % len(base_locations)]
    
    if asset_type == "Tunnel":
        return f"Tunnel portal {location}"
    elif asset_type == "OLE Mast":
        return f"OLE support structure {location}"
    else:
        return location


def generate_asset_metadata(asset_type: str) -> Dict[str, Any]:
    """Generate asset-specific metadata."""
    base_metadata = {
        "construction_start": (datetime.now() - timedelta(days=random.randint(180, 540))).strftime("%Y-%m-%d"),
        "planned_completion": (datetime.now() + timedelta(days=random.randint(-60, 180))).strftime("%Y-%m-%d"),
    }
    
    if asset_type == "Viaduct":
        base_metadata.update({
            "height_m": round(random.uniform(10, 25), 1),
            "span_m": round(random.uniform(30, 60), 1),
            "number_of_spans": random.randint(3, 8),
        })
    elif asset_type == "Bridge":
        base_metadata.update({
            "height_m": round(random.uniform(5, 15), 1),
            "span_m": round(random.uniform(20, 45), 1),
            "bridge_type": random.choice(["Steel composite", "Concrete box girder", "Steel through truss"]),
        })
    elif asset_type == "Tunnel":
        base_metadata.update({
            "length_m": round(random.uniform(500, 2000), 0),
            "diameter_m": round(random.uniform(8, 12), 1),
            "tunnel_type": random.choice(["Bored", "Cut and cover"]),
        })
    elif asset_type == "OLE Mast":
        base_metadata.update({
            "height_m": round(random.uniform(6, 10), 1),
            "mast_type": random.choice(["Cantilever", "Portal", "Twin mast"]),
            "foundation_type": random.choice(["Pile", "Raft"]),
        })
    
    return base_metadata


def calculate_taem_score(status: str) -> float:
    """Calculate realistic TAEM score based on status."""
    if status == "Ready":
        return round(random.uniform(85.0, 100.0), 2)
    elif status == "At Risk":
        return round(random.uniform(60.0, 75.0), 2)
    else:  # Not Ready
        return round(random.uniform(30.0, 60.0), 2)


# ==================== Asset Generation ====================

def generate_assets() -> List[Dict[str, Any]]:
    """Generate all asset records."""
    print("\n[1/4] Generating Assets...")
    
    assets = []
    asset_sequence = []
    
    # Create asset sequence with status distribution
    for status, count in STATUS_DISTRIBUTION.items():
        asset_sequence.extend([status] * count)
    
    # Shuffle to randomize status distribution
    random.shuffle(asset_sequence)
    
    # Generate assets by type
    global_index = 0
    for asset_type, config in ASSET_CONFIG.items():
        prefix = config["prefix"]
        count = config["count"]
        
        for i in range(1, count + 1):
            asset_id = f"{prefix}-{i:03d}"
            status = asset_sequence[global_index]
            route_section = random.choice(ROUTE_SECTIONS)
            contractor = random.choice(CONTRACTORS)
            
            # Generate planned completion date based on status
            if status == "Ready":
                # Ready assets completed or nearly complete
                completion_date = datetime.now() + timedelta(days=random.randint(-30, 15))
            elif status == "At Risk":
                # At risk assets have tight deadlines
                completion_date = datetime.now() + timedelta(days=random.randint(30, 90))
            else:  # Not Ready
                # Not ready assets may be overdue
                completion_date = datetime.now() + timedelta(days=random.randint(-60, 120))
            
            asset = {
                "asset_id": asset_id,
                "asset_name": f"{asset_type} {asset_id} - {route_section}",
                "asset_type": asset_type,
                "route_section": route_section,
                "contractor": contractor,
                "location_text": generate_location_text(asset_type, route_section, i),
                "design_status": "Approved" if status == "Ready" else random.choice(["Approved", "In Review", "Pending"]),
                "construction_status": "Complete" if status == "Ready" else random.choice(["In Progress", "On Hold", "Delayed"]),
                "readiness_status": status,
                "planned_completion_date": completion_date.isoformat(),
                "taem_evaluation_score": calculate_taem_score(status),
                "last_evaluation_date": datetime.now().isoformat(),
                "asset_metadata": generate_asset_metadata(asset_type),
            }
            
            assets.append(asset)
            global_index += 1
    
    print(f"  ✓ Generated {len(assets)} assets")
    return assets


# ==================== Deliverable Generation ====================

def generate_deliverables(assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate deliverables for all assets with status-appropriate data."""
    print("\n[2/4] Generating Deliverables...")
    
    deliverables = []
    
    for asset in assets:
        asset_id = asset["asset_id"]
        status = asset["readiness_status"]
        num_deliverables = random.randint(8, 15)
        
        # Select deliverable types (ensure critical ones are included)
        selected_types = []
        
        # Always include critical deliverables
        critical_types = [d for d in DELIVERABLE_TYPES if d["priority"] == "Critical"]
        selected_types.extend(critical_types)
        
        # Add random major and minor deliverables
        other_types = [d for d in DELIVERABLE_TYPES if d["priority"] != "Critical"]
        additional_count = num_deliverables - len(critical_types)
        selected_types.extend(random.choices(other_types, k=additional_count))
        
        for seq, deliv_type_config in enumerate(selected_types[:num_deliverables], start=1):
            deliverable_id = f"DEL-{asset_id}-{seq:02d}"
            deliv_type = deliv_type_config["type"]
            priority = deliv_type_config["priority"]
            duration = deliv_type_config["duration_days"]
            
            # Calculate due date based on asset completion date
            completion_date = datetime.fromisoformat(asset["planned_completion_date"])
            due_date = completion_date - timedelta(days=random.randint(30, 90))
            
            # Determine deliverable status based on asset status
            if status == "Ready":
                # Ready assets: all critical deliverables complete
                if priority == "Critical":
                    deliv_status = "Approved"
                    submission_date = due_date - timedelta(days=random.randint(1, 15))
                    approval_date = submission_date + timedelta(days=random.randint(1, 7))
                else:
                    deliv_status = random.choice(["Approved", "Submitted", "Pending"])
                    if deliv_status in ["Approved", "Submitted"]:
                        submission_date = due_date - timedelta(days=random.randint(1, 10))
                        approval_date = submission_date + timedelta(days=random.randint(1, 5)) if deliv_status == "Approved" else None
                    else:
                        submission_date = None
                        approval_date = None
                        
            elif status == "At Risk":
                # At risk assets: critical deliverables on track but delayed
                if priority == "Critical":
                    deliv_status = random.choice(["Submitted", "Pending", "Approved"])
                    if deliv_status != "Pending":
                        submission_date = due_date + timedelta(days=random.randint(-5, 10))
                        approval_date = submission_date + timedelta(days=random.randint(1, 7)) if deliv_status == "Approved" else None
                    else:
                        submission_date = None
                        approval_date = None
                else:
                    deliv_status = random.choice(["Submitted", "Pending", "Overdue", "Not Started"])
                    if deliv_status == "Submitted":
                        submission_date = due_date + timedelta(days=random.randint(-3, 7))
                        approval_date = None
                    elif deliv_status == "Overdue":
                        submission_date = None
                        approval_date = None
                    else:
                        submission_date = None
                        approval_date = None
                        
            else:  # Not Ready
                # Not ready assets: 1-2 critical deliverables missing or overdue
                if priority == "Critical" and random.random() < 0.4:  # 40% chance critical is missing
                    deliv_status = random.choice(["Overdue", "Not Started"])
                    submission_date = None
                    approval_date = None
                elif priority == "Critical":
                    deliv_status = random.choice(["Submitted", "Pending"])
                    if deliv_status == "Submitted":
                        submission_date = due_date + timedelta(days=random.randint(5, 30))
                        approval_date = None
                    else:
                        submission_date = None
                        approval_date = None
                else:
                    deliv_status = random.choice(["Submitted", "Pending", "Overdue", "Not Started"])
                    if deliv_status == "Submitted":
                        submission_date = due_date + timedelta(days=random.randint(-5, 20))
                        approval_date = None
                    else:
                        submission_date = None
                        approval_date = None
            
            # Calculate days overdue
            today = datetime.now()
            if deliv_status in ["Overdue", "Not Started"]:
                days_overdue = (today - due_date).days
            elif submission_date and submission_date > due_date:
                days_overdue = (submission_date - due_date).days
            else:
                days_overdue = None
            
            # Map approval status
            approval_status_map = {
                "Approved": "Approved",
                "Submitted": "Under Review",
                "Pending": None,
                "Overdue": None,
                "Not Started": None,
            }
            
            deliverable = {
                "deliverable_id": deliverable_id,
                "asset_id": asset_id,
                "deliverable_name": f"{deliv_type} - {asset['asset_name']}",
                "deliverable_type": deliv_type,
                "status": deliv_status,
                "approval_status": approval_status_map.get(deliv_status),
                "due_date": due_date.isoformat(),
                "submission_date": submission_date.isoformat() if submission_date else None,
                "approval_date": approval_date.isoformat() if approval_date else None,
                "responsible_party": asset["contractor"],
                "document_reference": f"DOC-{random.randint(100000, 999999)}",
                "days_overdue": days_overdue,
                "priority": priority,
                "notes": None,
            }
            
            deliverables.append(deliverable)
    
    print(f"  ✓ Generated {len(deliverables)} deliverables")
    return deliverables


# ==================== Cost Generation ====================

def generate_costs(assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate cost tracking data for all assets."""
    print("\n[3/4] Generating Cost Data...")
    
    costs = []
    
    for asset in assets:
        asset_id = asset["asset_id"]
        asset_type = asset["asset_type"]
        status = asset["readiness_status"]
        
        # Get budget range for asset type
        budget_min, budget_max = ASSET_CONFIG[asset_type]["budget_range"]
        budget_amount = random.randint(budget_min, budget_max)
        
        # Calculate actual amount based on status
        if status == "Ready":
            # Ready assets: under budget or slightly over
            variance_pct = random.uniform(-10.0, 5.0)
        elif status == "At Risk":
            # At risk assets: trending toward overrun
            variance_pct = random.uniform(-5.0, 15.0)
        else:  # Not Ready
            # Not ready assets: significant cost overruns
            variance_pct = random.uniform(10.0, 30.0)
        
        actual_amount = int(budget_amount * (1 + variance_pct / 100))
        variance_amount = actual_amount - budget_amount
        forecast_amount = int(actual_amount * random.uniform(1.0, 1.05))
        
        # Determine status
        if variance_pct < -5:
            cost_status = "Under Budget"
        elif variance_pct > 10:
            cost_status = "Over Budget"
        elif variance_pct > 5:
            cost_status = "At Risk"
        else:
            cost_status = "On Budget"
        
        cost = {
            "cost_line_id": f"COST-{asset_id}",
            "asset_id": asset_id,
            "budget_amount": float(budget_amount),
            "actual_amount": float(actual_amount),
            "forecast_amount": float(forecast_amount),
            "variance_amount": float(variance_amount),
            "variance_pct": round(variance_pct, 2),
            "cost_category": random.choice(["Construction", "Materials", "Labour", "Equipment"]),
            "reporting_period": "2024-Q4",
            "status": cost_status,
            "notes": None,
        }
        
        costs.append(cost)
    
    print(f"  ✓ Generated {len(costs)} cost records")
    return costs


# ==================== Certificate Generation ====================

def generate_certificates(assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate certificates for all assets."""
    print("\n[4/4] Generating Certificates...")
    
    certificates = []
    issuing_authorities = [
        "BSI Group",
        "TWI Certification",
        "Lloyd's Register",
        "Bureau Veritas",
        "SGS United Kingdom",
    ]
    
    for asset in assets:
        asset_id = asset["asset_id"]
        status = asset["readiness_status"]
        num_certificates = random.randint(3, 6)
        
        # Select certificate types
        selected_cert_types = random.sample(CERTIFICATE_TYPES, min(num_certificates, len(CERTIFICATE_TYPES)))
        
        for seq, cert_type in enumerate(selected_cert_types, start=1):
            certificate_id = f"CERT-{asset_id}-{seq:02d}"
            
            # Generate issue and expiry dates
            issue_date = datetime.now() - timedelta(days=random.randint(90, 365))
            expiry_date = issue_date + timedelta(days=random.randint(730, 1095))  # 2-3 years validity
            
            # Determine certificate status based on asset status
            if status == "Ready":
                # Ready assets: all certificates valid, none expiring soon
                cert_status = "Valid"
                # Ensure expiry is well in the future
                if (expiry_date - datetime.now()).days < 90:
                    expiry_date = datetime.now() + timedelta(days=random.randint(180, 365))
                    
            elif status == "At Risk":
                # At risk assets: valid certificates but some expiring soon
                if random.random() < 0.3:  # 30% chance expiring soon
                    cert_status = "Expiring Soon"
                    expiry_date = datetime.now() + timedelta(days=random.randint(15, 30))
                else:
                    cert_status = "Valid"
                    
            else:  # Not Ready
                # Not ready assets: some expired certificates
                if random.random() < 0.4:  # 40% chance expired
                    cert_status = "Expired"
                    expiry_date = datetime.now() - timedelta(days=random.randint(1, 30))
                else:
                    cert_status = random.choice(["Valid", "Expiring Soon"])
                    if cert_status == "Expiring Soon":
                        expiry_date = datetime.now() + timedelta(days=random.randint(10, 25))
            
            # Calculate days until expiry
            days_until_expiry = (expiry_date - datetime.now()).days
            
            certificate = {
                "certificate_id": certificate_id,
                "asset_id": asset_id,
                "certificate_name": f"{cert_type} - {asset['asset_name']}",
                "certificate_type": cert_type,
                "issuing_authority": random.choice(issuing_authorities),
                "issue_date": issue_date.isoformat(),
                "expiry_date": expiry_date.isoformat(),
                "status": cert_status,
                "days_until_expiry": days_until_expiry,
                "document_reference": f"CERT-{random.randint(100000, 999999)}",
                "notes": None,
            }
            
            certificates.append(certificate)
    
    print(f"  ✓ Generated {len(certificates)} certificates")
    return certificates


# ==================== Main Execution ====================

def main():
    """Main execution function."""
    print("\nStarting data generation...\n")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate all data
    assets = generate_assets()
    deliverables = generate_deliverables(assets)
    costs = generate_costs(assets)
    certificates = generate_certificates(assets)
    
    # Write JSON files
    print("\n" + "=" * 80)
    print("Writing JSON files...")
    
    files_written = []
    
    assets_file = OUTPUT_DIR / "assets.json"
    with open(assets_file, "w") as f:
        json.dump(assets, f, indent=2)
    files_written.append(f"  ✓ {assets_file} ({len(assets)} assets)")
    
    deliverables_file = OUTPUT_DIR / "deliverables.json"
    with open(deliverables_file, "w") as f:
        json.dump(deliverables, f, indent=2)
    files_written.append(f"  ✓ {deliverables_file} ({len(deliverables)} deliverables)")
    
    costs_file = OUTPUT_DIR / "costs.json"
    with open(costs_file, "w") as f:
        json.dump(costs, f, indent=2)
    files_written.append(f"  ✓ {costs_file} ({len(costs)} cost records)")
    
    certificates_file = OUTPUT_DIR / "certificates.json"
    with open(certificates_file, "w") as f:
        json.dump(certificates, f, indent=2)
    files_written.append(f"  ✓ {certificates_file} ({len(certificates)} certificates)")
    
    for file_info in files_written:
        print(file_info)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Data Generation Summary")
    print("=" * 80)
    
    status_counts = {}
    for asset in assets:
        status = asset["readiness_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("\nAsset Status Distribution:")
    for status, count in sorted(status_counts.items()):
        pct = (count / len(assets)) * 100
        print(f"  {status:12s}: {count:2d} assets ({pct:5.1f}%)")
    
    print(f"\nTotal Data Points:")
    print(f"  Assets:        {len(assets):4d}")
    print(f"  Deliverables:  {len(deliverables):4d}")
    print(f"  Cost Records:  {len(costs):4d}")
    print(f"  Certificates:  {len(certificates):4d}")
    print(f"  {'─' * 20}")
    print(f"  Total:         {len(assets) + len(deliverables) + len(costs) + len(certificates):4d}")
    
    print("\n" + "=" * 80)
    print("✅ Data generation complete!")
    print(f"📁 Output location: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
