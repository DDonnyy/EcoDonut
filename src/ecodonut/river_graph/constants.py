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
        typical_annual_load_tons=418.178818,
        pollutants=[
            PollutantShare(name="Сухой остаток", share=0.392392),
            PollutantShare(name="Хлорид-анион", share=0.258215),
            PollutantShare(name="Сульфат-анион", share=0.156662),
            PollutantShare(name="Натрий", share=0.052351),
            PollutantShare(name="Нитрат-анион", share=0.042415),
            PollutantShare(name="Кальций", share=0.041219),
            PollutantShare(name="Вольфрам триоксид", share=0.017162),
            PollutantShare(name="ХПК", share=0.016194),
            PollutantShare(name="Магний", share=0.011975),
            PollutantShare(name="Калий", share=0.011414),
        ],
    ),
    2: PollutionProfile(
        level=2,
        typical_annual_load_tons=13.972545,
        pollutants=[
            PollutantShare(name="Хлорид-анион", share=0.577117),
            PollutantShare(name="Сульфат-анион", share=0.278804),
            PollutantShare(name="Нитрат-анион", share=0.132578),
            PollutantShare(name="2-Бутилфенол (о-Бутилфенол)", share=0.008091),
            PollutantShare(name="2-Бром-1-гидроксибензол", share=0.001618),
            PollutantShare(name="4-Бутилфенол (п-Бутилфенол)", share=0.001618),
            PollutantShare(name="Вольфрам триоксид", share=0.000167),
            PollutantShare(name="Сухой остаток", share=0.000006),
            PollutantShare(name="ХПК", share=0.000001),
            PollutantShare(name="Натрий", share=0.000000),
        ],
    ),
    3: PollutionProfile(
        level=3,
        typical_annual_load_tons=0.806299,
        pollutants=[
            PollutantShare(name="Хлорид-анион", share=0.552738),
            PollutantShare(name="Сульфат-анион", share=0.200021),
            PollutantShare(name="Сухой остаток", share=0.144530),
            PollutantShare(name="Диэтиловый эфир", share=0.024912),
            PollutantShare(name="Хлоргидринстирола метиловый эфир", share=0.022488),
            PollutantShare(name="Вольфрам триоксид", share=0.019761),
            PollutantShare(name="Нитрат-анион", share=0.017052),
            PollutantShare(name="Фторид-анион", share=0.007316),
            PollutantShare(name="Натрий", share=0.006592),
            PollutantShare(name="Аммоний-ион", share=0.004590),
        ],
    ),
    4: PollutionProfile(
        level=4,
        typical_annual_load_tons=0.00322,
        pollutants=[
            PollutantShare(name="Нитрат-анион", share=0.638503),
            PollutantShare(name="Железо", share=0.173297),
            PollutantShare(name="Сухой остаток", share=0.062920),
            PollutantShare(name="Вольфрам триоксид", share=0.041945),
            PollutantShare(name="Аммиак", share=0.036879),
            PollutantShare(name="Сульфат-анион", share=0.017005),
            PollutantShare(name="Хлорид-анион", share=0.016616),
            PollutantShare(
                name="Взвешенные вещества инертная природная минеральная взвесь",
                share=0.009092,
            ),
            PollutantShare(name="Нитрит-анион", share=0.003546),
            PollutantShare(name="Микроорганизмы", share=0.000198),
        ],
    ),
}
