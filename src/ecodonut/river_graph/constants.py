from dataclasses import dataclass


@dataclass(frozen=True)
class PollutantShare:
    name: str  # имя вещества
    share: float  # доля (0..1) в общей массе профиля


@dataclass
class PollutionProfile:
    level: int
    typical_annual_load_tons: float
    pollutants: list[PollutantShare]

    def normalize_shares(self) -> None:
        s = sum(p.share for p in self.pollutants)
        if s <= 0:
            return
        self.pollutants = [PollutantShare(name=p.name, share=p.share / s) for p in self.pollutants]


PROFILES: dict[int, PollutionProfile] = {
    1: PollutionProfile(
        level=1,
        typical_annual_load_tons=3707.376629,
        pollutants=[
            PollutantShare(name="Сухой остаток", share=0.435008),
            PollutantShare(name="Хлорид-анион", share=0.286259),
            PollutantShare(name="Сульфат-анион", share=0.173676),
            PollutantShare(name="Натрий", share=0.058037),
            PollutantShare(name="Нитрат-анион", share=0.047021),
        ],
    ),
    2: PollutionProfile(
        level=2,
        typical_annual_load_tons=119.157221,
        pollutants=[
            PollutantShare(name="Хлорид-анион", share=0.578153),
            PollutantShare(name="Сульфат-анион", share=0.279304),
            PollutantShare(name="Нитрат-анион", share=0.132816),
            PollutantShare(name="2-Бутилфенол (о-Бутилфенол)", share=0.008106),
            PollutantShare(name="2-Бром-1-гидроксибензол", share=0.001621),
        ],
    ),
    3: PollutionProfile(
        level=3,
        typical_annual_load_tons=20.084530,
        pollutants=[
            PollutantShare(name="Хлорид-анион", share=0.585100),
            PollutantShare(name="Сульфат-анион", share=0.211733),
            PollutantShare(name="Сухой остаток", share=0.152992),
            PollutantShare(name="Диэтиловый эфир", share=0.026370),
            PollutantShare(name="Хлоргидринстирола метиловый эфир", share=0.023805),
        ],
    ),
    4: PollutionProfile(
        level=4,
        typical_annual_load_tons=3.928993,
        pollutants=[
            PollutantShare(name="Нитрат-анион", share=0.669611),
            PollutantShare(name="Железо", share=0.181740),
            PollutantShare(name="Сухой остаток", share=0.065985),
            PollutantShare(name="Вольфрам триоксид", share=0.043988),
            PollutantShare(name="Аммиак", share=0.038675),
        ],
    ),
}
