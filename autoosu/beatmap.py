"""Beatmap data model and .osu (file format v14) serialisation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

PLAYFIELD_W = 512
PLAYFIELD_H = 384

TYPE_CIRCLE = 1
TYPE_SLIDER = 2
TYPE_NEW_COMBO = 4
TYPE_SPINNER = 8

HS_NORMAL = 0
HS_WHISTLE = 2
HS_FINISH = 4
HS_CLAP = 8


@dataclass
class HitObject:
    x: int
    y: int
    time: int
    new_combo: bool = False
    hitsound: int = 0

    @property
    def end_time(self) -> int:
        return self.time

    @property
    def end_pos(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def _type(self, base: int) -> int:
        return base | (TYPE_NEW_COMBO if self.new_combo else 0)


@dataclass
class Circle(HitObject):
    def to_line(self) -> str:
        return f"{self.x},{self.y},{self.time},{self._type(TYPE_CIRCLE)},{self.hitsound},0:0:0:0:"


@dataclass
class Slider(HitObject):
    curve_type: str = "B"                      # L = linear, B = bezier, P = perfect circle
    points: List[Tuple[int, int]] = field(default_factory=list)  # anchors after the head
    repeats: int = 1                           # osu "slides": 1 = no repeat
    length: float = 0.0                        # pixel length of one slide
    duration: int = 0                          # ms for the whole slider (all slides)
    end_x: int = 0
    end_y: int = 0

    @property
    def end_time(self) -> int:
        return self.time + self.duration

    @property
    def end_pos(self) -> Tuple[int, int]:
        return (self.end_x, self.end_y)

    def to_line(self) -> str:
        pts = "|".join(f"{px}:{py}" for px, py in self.points)
        edge_sounds = "|".join(["0"] * (self.repeats + 1))
        edge_sets = "|".join(["0:0"] * (self.repeats + 1))
        return (
            f"{self.x},{self.y},{self.time},{self._type(TYPE_SLIDER)},{self.hitsound},"
            f"{self.curve_type}|{pts},{self.repeats},{self.length:.4f},{edge_sounds},{edge_sets},0:0:0:0:"
        )


@dataclass
class Spinner(HitObject):
    end: int = 0

    @property
    def end_time(self) -> int:
        return self.end

    @property
    def end_pos(self) -> Tuple[int, int]:
        return (PLAYFIELD_W // 2, PLAYFIELD_H // 2)

    def to_line(self) -> str:
        t = TYPE_SPINNER | TYPE_NEW_COMBO
        return f"{PLAYFIELD_W // 2},{PLAYFIELD_H // 2},{self.time},{t},{self.hitsound},{self.end},0:0:0:0:"


@dataclass
class TimingPoint:
    time: int
    beat_length: float          # ms per beat (uninherited) or negative SV percent (inherited)
    meter: int = 4
    sample_set: int = 2         # 1 normal, 2 soft, 3 drum
    sample_index: int = 0
    volume: int = 70
    uninherited: bool = True
    kiai: bool = False

    def to_line(self) -> str:
        return (
            f"{self.time},{self.beat_length:.6f},{self.meter},{self.sample_set},{self.sample_index},"
            f"{self.volume},{1 if self.uninherited else 0},{1 if self.kiai else 0}"
        )


@dataclass
class Break:
    start: int
    end: int


@dataclass
class Beatmap:
    audio_filename: str
    title: str
    artist: str
    version: str
    creator: str = "AUTO-OSU"
    source: str = ""
    tags: str = "autoosu kanzei"
    hp: float = 5
    cs: float = 4
    od: float = 6
    ar: float = 8
    slider_multiplier: float = 1.4
    slider_tick_rate: float = 1
    distance_spacing: float = 1.0
    preview_time: int = -1
    stack_leniency: float = 0.7
    background: str = ""                 # image file name inside the .osz, "" = none
    timing_points: List[TimingPoint] = field(default_factory=list)
    breaks: List[Break] = field(default_factory=list)
    hit_objects: List[HitObject] = field(default_factory=list)

    def osu_filename(self) -> str:
        return f"{self.artist} - {self.title} ({self.creator}) [{self.version}].osu"

    def to_osu(self) -> str:
        lines: List[str] = []
        a = lines.append
        a("osu file format v14")
        a("")
        a("[General]")
        a(f"AudioFilename: {self.audio_filename}")
        a("AudioLeadIn: 0")
        a(f"PreviewTime: {self.preview_time}")
        a("Countdown: 0")
        a("SampleSet: Soft")
        a(f"StackLeniency: {self.stack_leniency}")
        a("Mode: 0")
        a("LetterboxInBreaks: 0")
        a("WidescreenStoryboard: 0")
        a("")
        a("[Editor]")
        a(f"DistanceSpacing: {self.distance_spacing}")
        a("BeatDivisor: 4")
        a("GridSize: 4")
        a("TimelineZoom: 1")
        a("")
        a("[Metadata]")
        a(f"Title:{self.title}")
        a(f"TitleUnicode:{self.title}")
        a(f"Artist:{self.artist}")
        a(f"ArtistUnicode:{self.artist}")
        a(f"Creator:{self.creator}")
        a(f"Version:{self.version}")
        a(f"Source:{self.source}")
        a(f"Tags:{self.tags}")
        a("BeatmapID:0")
        a("BeatmapSetID:-1")
        a("")
        a("[Difficulty]")
        a(f"HPDrainRate:{self.hp:g}")
        a(f"CircleSize:{self.cs:g}")
        a(f"OverallDifficulty:{self.od:g}")
        a(f"ApproachRate:{self.ar:g}")
        a(f"SliderMultiplier:{self.slider_multiplier:g}")
        a(f"SliderTickRate:{self.slider_tick_rate:g}")
        a("")
        a("[Events]")
        a("//Background and Video events")
        if self.background:
            a(f'0,0,"{self.background}",0,0')
        a("//Break Periods")
        for b in self.breaks:
            a(f"2,{b.start},{b.end}")
        a("//Storyboard Layer 0 (Background)")
        a("//Storyboard Layer 1 (Fail)")
        a("//Storyboard Layer 2 (Pass)")
        a("//Storyboard Layer 3 (Foreground)")
        a("//Storyboard Layer 4 (Overlay)")
        a("//Storyboard Sound Samples")
        a("")
        a("[TimingPoints]")
        for tp in sorted(self.timing_points, key=lambda t: (t.time, not t.uninherited)):
            a(tp.to_line())
        a("")
        a("[Colours]")
        a("Combo1 : 235,90,90")
        a("Combo2 : 90,170,235")
        a("Combo3 : 120,210,120")
        a("Combo4 : 240,190,80")
        a("")
        a("[HitObjects]")
        for h in sorted(self.hit_objects, key=lambda h: h.time):
            a(h.to_line())
        a("")
        return "\r\n".join(lines)
