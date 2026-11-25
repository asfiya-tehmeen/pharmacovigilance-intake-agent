"""Structured safety case, loosely inspired by ICH E2B(R3) sections (simplified; not a conformant E2B message).

E2B(R3) mapping (approximate):
  patient      -> D (patient characteristics)
  reporter     -> C.2 (primary source)
  products     -> G (drug/biologic information)
  events       -> E (reaction/event), incl. seriousness criteria E.i.3.2 and outcome E.i.7
  narrative    -> H.1 (case narrative)
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SeriousCriterion = Literal["death", "life_threatening", "hospitalisation", "disability", "birth_defect",
                           "other_medically_important"]
Outcome = Literal["recovered", "recovering", "not_recovered", "recovered_with_sequelae", "fatal", "unknown"]


class Evidence(BaseModel):
    quote: str = ""                 # exact text copied from the narrative
    verified: bool | None = None    # set by the deterministic span verifier


class Patient(BaseModel):
    age_years: float | None = None
    sex: Literal["F", "M", "U"] = "U"
    evidence: Evidence = Field(default_factory=Evidence)


class Reporter(BaseModel):
    type: Literal["patient", "healthcare_professional", "caregiver", "other", "unknown"] = "unknown"
    evidence: Evidence = Field(default_factory=Evidence)


class Product(BaseModel):
    name: str
    vax_type: str | None = None     # VAERS VAX_TYPE code, e.g. COVID19, FLU4
    dose_number: str | None = None
    lot: str | None = None
    evidence: Evidence = Field(default_factory=Evidence)


class Event(BaseModel):
    verbatim: str                    # reporter's own words (must be in the narrative)
    coded_term: str | None = None    # chosen from the retrieved candidate list (or None)
    onset_days: float | None = None  # days from vaccination
    evidence: Evidence = Field(default_factory=Evidence)
    expectedness: Literal["listed", "unlisted", "no_label", "not_checked"] = "not_checked"
    label_match: str | None = None


class Seriousness(BaseModel):
    serious: bool = False
    criteria: list[SeriousCriterion] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    rule_serious: bool | None = None        # deterministic rule layer verdict
    rule_criteria: list[str] = Field(default_factory=list)
    lr_serious: bool | None = None          # TF-IDF logistic-regression layer verdict (threshold set for 99% recall on dev)
    lr_score: float | None = None


class Validity(BaseModel):
    """ICH 'four minimum elements' for a valid ICSR."""
    identifiable_patient: bool = False
    identifiable_reporter: bool = False
    suspect_product: bool = False
    adverse_event: bool = False

    @property
    def valid(self) -> bool:
        return all([self.identifiable_patient, self.identifiable_reporter, self.suspect_product, self.adverse_event])


class Provenance(BaseModel):
    model: str = ""
    prompt_version: str = ""
    mode: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    cached: bool = False
    escalated: bool = False


class SafetyCase(BaseModel):
    case_id: int
    narrative: str
    patient: Patient = Field(default_factory=Patient)
    reporter: Reporter = Field(default_factory=Reporter)
    products: list[Product] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    seriousness: Seriousness = Field(default_factory=Seriousness)
    outcome: Outcome = "unknown"
    validity: Validity = Field(default_factory=Validity)
    confidence: float = 0.0
    confidence_parts: dict[str, float] = Field(default_factory=dict)
    route: Literal["auto", "human_review"] = "human_review"
    route_reasons: list[str] = Field(default_factory=list)
    rejected_fields: list[str] = Field(default_factory=list)   # dropped by the span verifier
    provenance: Provenance = Field(default_factory=Provenance)
    error: str | None = None
