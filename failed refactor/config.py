from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    min_freq: float = 3.0
    max_freq: float = 10.0
    start_time: float = -1990.0
    end_time: float = 15000.0
    win_pre: float = 0.5
    win_post: float = 0.5
    r_window_min: float = 0.7
    move_limit_sec: float = 0.05

    all_channels: bool = True
    component: str = "R"
    align_phase: str = "S"
    verbose: bool = False

    path_prefix: str = "/Users/vidale/Documents/Research/Mingze_SJF/"
    sps_rate: str = "down100"
    event: str = "CI_40353544"

    show_individual_seismograms: bool = False
    show_record_section_plot: bool = False

    @property
    def info_root(self) -> Path:
        return Path(self.path_prefix) / "20220930_events_cut" / "event_sta_info"

    @property
    def data_path(self) -> Path:
        return Path(self.path_prefix) / "20220930_events_cut" / f"20220930_{self.sps_rate}"

    @property
    def events(self) -> list[str]:
        return [self.event]