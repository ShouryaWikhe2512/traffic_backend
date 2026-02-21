from typing import Dict, List, Optional

class KalmanFilter:
    def __init__(self, process_variance: float = 1e-5, measurement_variance: float = 1e-1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def update(self, measurement: float) -> float:
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1.0 - blending_factor) * priori_error_estimate
        return float(self.posteri_estimate)

class SignalOptimizer:
    def __init__(self):
        self.base_cycle = 90
        self.min_green = 15
        self.yellow = 3

    def calculate_plan(self, 
                       queues: Dict[str, float], 
                       emergency_flag: bool, 
                       emergency_dir: Optional[str] = None,
                       drl_deltas: Optional[Dict[str, float]] = None) -> Dict:
        
        # 1. PRESSURE CALCULATION (Upstream - 0)
        pressures = {k: max(0.0, float(v)) for k, v in queues.items()}
        
        # 2. DYNAMIC CYCLE BREAKDOWN
        total_load = sum(pressures.values())
        load_adj = min(30, int(total_load * 1.5))
        emergency_adj = 10 if emergency_flag else 0
        final_cycle = self.base_cycle + load_adj + emergency_adj
        
        # 3. PHASE SELECTION & SEQUENCING
        sorted_dirs = sorted(pressures.items(), key=lambda x: x[1], reverse=True)
        maxp_selected = sorted_dirs[0][0]
        phase_sequence = [d[0].upper() for d in sorted_dirs]
        
        # 4. HYBRID SPLIT ALLOCATION (MaxP + DRL)
        total_p = sum(pressures.values()) or 1.0
        available_green = float(final_cycle - (4 * self.yellow))
        
        splits = {}
        for k, v in pressures.items():
            base_share = (v / total_p) * available_green
            drl_offset = drl_deltas.get(f"delta_{k}", 0.0) if drl_deltas else 0.0
            # Ensure safety floor
            splits[k] = max(float(self.min_green), float(base_share + drl_offset))

        # Re-normalize to fit available green
        actual_total = sum(splits.values())
        norm_factor = available_green / actual_total
        final_splits = {k: float(round(v * norm_factor)) for k, v in splits.items()}
        
        # Adjust for rounding
        diff = available_green - sum(final_splits.values())
        final_splits[maxp_selected] += diff

        # 5. EMERGENCY OVERRIDE
        if emergency_flag and emergency_dir:
            # Force priority split
            final_splits = {k: float(self.min_green) for k in pressures}
            final_splits[emergency_dir] = 60.0
            # Re-sum for emergency cycle
            final_cycle = int(sum(final_splits.values()) + (4 * self.yellow))

        # 6. AI REASONING TRACE
        e_dir_str = emergency_dir.upper() if emergency_dir else "NONE"
        trace = [
            f"Highest pressure approach: {maxp_selected.upper()} ({pressures[maxp_selected]:.1f})",
            f"Network load factor: +{load_adj}s (Total Load: {total_load:.1f})"
        ]
        if drl_deltas:
            trace.append(f"DRL injected {sum(float(v) for v in drl_deltas.values()):.1f}s predictive buffer")
        if emergency_flag:
            trace.append(f"!!! EMERGENCY OVERRIDE IN {e_dir_str} !!! (+{emergency_adj}s safety lock)")
        
        reasoning = " | ".join(trace)

        return {
            "raw_counts": {k: int(v) for k, v in queues.items()},
            "smoothed_queues": {k: float(round(float(v), 2)) for k, v in queues.items()},
            "pressures": {k: float(round(float(v), 2)) for k, v in pressures.items()},
            "emergency_flags": {k: (k == emergency_dir if emergency_flag else False) for k in queues},
            "splits": {k: int(v) for k, v in final_splits.items()},
            "cycle_breakdown": {
                "base": self.base_cycle,
                "load_adj": load_adj,
                "emergency_adj": emergency_adj,
                "final": final_cycle
            },
            "maxp_selected": maxp_selected.upper(),
            "drl_adjustment": drl_deltas or {},
            "safety_override": emergency_flag,
            "phase_sequence": phase_sequence,
            "selected_phase": f"{maxp_selected.upper()} APPROACH",
            "reasoning": reasoning,
            "mode": "HYBRID (MAX-P + DRL)" if not emergency_flag else "EMERGENCY PREEMPTION"
        }
