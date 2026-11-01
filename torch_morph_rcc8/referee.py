from dataclasses import dataclass, asdict
from enum import Enum
import json


class DecisionState(Enum):
    CONFIRMED = "CONFIRMED"
    OVERRIDDEN = "OVERRIDDEN"
    UNKNOWN_OCCLUSION = "ABSTAIN_UNKNOWN_OCCLUSION"


@dataclass
class TopologicalProofReport:
    object_id: str
    domain: str
    dl_raw_pred: str
    dl_confidence: float
    rcc8_relation: str
    topological_metric_name: str
    topological_metric_value: float
    rule_fired: str
    final_state: str
    final_label: str

    def to_pretty_log(self) -> str:
        """Formats the explainable topological proof chain for console and paper logging."""
        return (
            f"[XAI Topological Audit Log | ID: {self.object_id} ({self.domain})]\n"
            f"  1. Initial DL Prediction : {self.dl_raw_pred} (Confidence: {self.dl_confidence:.3f})\n"
            f"  2. RCC8 Spatial Relation : {self.rcc8_relation}\n"
            f"  3. Topological Invariant : {self.topological_metric_name} = {self.topological_metric_value:.3f}\n"
            f"  4. Fired Logical Rule    : {self.rule_fired}\n"
            f"  5. Final Decision        : [{self.final_state}] -> {self.final_label}"
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class AgriTopoReferee:
    """Neuro-Symbolic Referee Engine based on RCC-8 and Higra Max-Tree LCA merge altitudes."""
    def __init__(
        self,
        lca_safe_threshold: float = 0.710,
        lca_reject_threshold: float = 0.560,
    ):
        self.lca_safe_thr = lca_safe_threshold
        self.lca_reject_thr = lca_reject_threshold

    def judge_crop_weed_blob(
        self,
        blob_id: str,
        dl_pred: str,
        dl_conf: float,
        rcc8_rel_with_crop: str,
        lca_merge_altitude: float,
        growth_stage: str = "late_06-05",
    ) -> TopologicalProofReport:
        if dl_pred == "weed" and rcc8_rel_with_crop == "NTPP":
            return TopologicalProofReport(
                object_id=blob_id,
                domain="PhenoBench_Segmentation",
                dl_raw_pred=dl_pred,
                dl_confidence=dl_conf,
                rcc8_relation="NTPP(Blob, Crop)",
                topological_metric_name="MaxTree_LCA_Merge_Altitude",
                topological_metric_value=lca_merge_altitude,
                rule_fired="NTPP(x, Crop) -> NOT_Weed(x) [Interior false-positive recovery rule]",
                final_state=DecisionState.OVERRIDDEN.value,
                final_label="crop",
            )

        if dl_pred == "weed" and rcc8_rel_with_crop in ("EC", "PO") and lca_merge_altitude >= self.lca_safe_thr:
            return TopologicalProofReport(
                object_id=blob_id,
                domain="PhenoBench_Segmentation",
                dl_raw_pred=dl_pred,
                dl_confidence=dl_conf,
                rcc8_relation=f"{rcc8_rel_with_crop}(Blob, Crop)",
                topological_metric_name="MaxTree_LCA_Merge_Altitude",
                topological_metric_value=lca_merge_altitude,
                rule_fired=f"Contact(x, Crop) AND (LCA_Alt >= {self.lca_safe_thr}) -> NOT_Weed(x)",
                final_state=DecisionState.OVERRIDDEN.value,
                final_label="crop",
            )

        if (
            dl_pred == "weed"
            and rcc8_rel_with_crop in ("EC", "PO")
            and self.lca_reject_thr <= lca_merge_altitude < self.lca_safe_thr
            and "late" in growth_stage
        ):
            return TopologicalProofReport(
                object_id=blob_id,
                domain="PhenoBench_Segmentation",
                dl_raw_pred=dl_pred,
                dl_confidence=dl_conf,
                rcc8_relation=f"{rcc8_rel_with_crop}(Blob, Crop)",
                topological_metric_name="MaxTree_LCA_Merge_Altitude",
                topological_metric_value=lca_merge_altitude,
                rule_fired="LateStage AND EC(x, Crop) AND Ambiguous_LCA -> ABSTAIN(UNKNOWN_OCCLUSION)",
                final_state=DecisionState.UNKNOWN_OCCLUSION.value,
                final_label="UNKNOWN_OCCLUSION",
            )

        return TopologicalProofReport(
            object_id=blob_id,
            domain="PhenoBench_Segmentation",
            dl_raw_pred=dl_pred,
            dl_confidence=dl_conf,
            rcc8_relation=f"{rcc8_rel_with_crop}(Blob, Crop)",
            topological_metric_name="MaxTree_LCA_Merge_Altitude",
            topological_metric_value=lca_merge_altitude,
            rule_fired="No_Override_Condition_Met -> Keep_DL_Prediction",
            final_state=DecisionState.CONFIRMED.value,
            final_label=dl_pred,
        )
