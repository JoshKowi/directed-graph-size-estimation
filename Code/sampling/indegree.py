"""Woher NMMC den Eingangsgrad d-(j) nimmt.

NMMC (sampling.nmmc) braucht zu jedem Knoten j die Groesse |S_j| aus
Theorem 3.1 des Papers: die Zahl der Knoten, aus denen die Vorschlagskette in
j hineinlaufen kann. Beim Simple Random Walk als Vorschlag ist das genau der
Eingangsgrad d-(j). Er steht in der Annahmewahrscheinlichkeit

    b_ij = (pi(j)/pi(i)) * d+(i)/d-(j)

und ist damit die einzige Groesse des Verfahrens, die nicht aus der
Nachbarabfrage selbst faellt.

Diese Achse steht bei NMMC an der Stelle, an der beim Random Walk die
Sackgassen-Strategie (sampling.dead_ends) und bei DURW die Sprungart
(sampling.jumps) steht: sie entscheidet, welches Oracle gebraucht wird, und
damit -- wie im Repo ueblich erst in estimators/__init__.py -- ueber die
Kategorie.

    online       -- aus selbst beobachteten Kanten geschaetzt (Paper, 6.3).
                    Braucht nur das CrawlOracle: real umsetzbar.
    exact        -- der wahre Eingangsgrad (graphs.graph.in_degrees) ueber
                    oracles.local_access.InDegreeCrawlOracle: nur Vergleich.
    cross-one    -- der Eingangsgrad derselben Entitaet im *Partnergraphen*
                    (config.CROSS_GRAPHS), fehlt sie dort: 1.
    cross-online -- derselbe Index, fehlt die Entitaet: die Online-Schaetzung.

Die beiden cross-Varianten sind real umsetzbar: der Partner ist externes Wissen
ueber Entitaeten, keine Kenntnis der Knotenmenge des geschaetzten Graphen
(Begruendung im Docstring von oracles.local_access.CrossInDegreeCrawlOracle).
Sie sind damit die realistische Fassung von `exact` -- und die eigentliche
Frage dieser Achse: traegt ein *fremder*, unvollstaendiger Prior das Verfahren
dort, wo die reine Online-Schaetzung es nicht tut?

Der Unterschied der beiden Fallbacks ist genau der Umgang mit dem, was der
Partner nicht kennt. `cross-one` setzt 1 und ueberschaetzt b_ij dort maximal --
der Walk nimmt Zuege in unbekanntes Gebiet also besonders bereitwillig an.
`cross-online` fuellt die Luecke mit dem, was der Walk selbst gesehen hat, ist
also nie schlechter informiert als `online`, aber im Kern des Partners deutlich
besser. Welcher der beiden gewinnt, ist eine empirische Frage.

Warum `exact` trotzdem gebaut wird, obwohl es nicht real umsetzbar ist: die
Online-Schaetzung verschiebt b_ij massiv. Auf Slashdot0811 gerichtet ist der
Median von b_ij ueber alle Kanten mit wahrem d- gerade 1,09 -- mit d- = 1 aber
58, bei unveraendertem Maximum von 2507. Der Grenzfall ist Korollar 3.3 des
Papers: waere d^- ueberall 1, so bliebe b_ij = d+(i) und die Kette waere
P_ij = A_ij/c, deren QSD die *Eigenvektorzentralitaet* ist und nicht die
Zielverteilung.

Gemessen wird dieser Grenzfall nicht ganz erreicht, aber die Richtung stimmt:
nach einem Lauf mit Budget 5 % hat der Walk 29 359 Knoten als Kantenziel
gesehen, 51,6 % davon stehen weiter auf dem Boden 1, und der Median von
d^-/d- liegt bei 0,20 -- eine Unterschaetzung um Faktor 5. Ueber die
tatsaechlichen *Vorschlaege* gewichtet sind es nur 16 % mit d^- = 1
(nmmc_trace.py, `frac_dinhat_1`): ein Walk, der sich in einer Region verfaengt,
beobachtet deren eingehende Kanten immer wieder und schaetzt dort brauchbar.
Die Schaetzung ist also *lokal* -- gut im schon besuchten Kern, gleich 1 an
seinem Rand. Genau dort entstehen aber die Zuege, die die Abdeckung
vergroessern wuerden.

Ohne die `exact`-Gegenprobe waere aus einem schlechten Ergebnis nicht ablesbar,
ob das Verfahren oder die Schaetzung des Eingangsgrades versagt hat.

**Zustandslos.** Anders als es der Vergleich mit sampling.jumps nahelegt, haelt
eine Instanz *keinen* Zustand: `start()` legt ihn je Lauf neu an, der Sampler
reicht ihn durch. Sonst ueberlebte der geschaetzte Eingangsgrad zwischen zwei
Laeufen desselben Estimator-Objekts -- check_nested.py ruft genau so auf
(estimate_nested(), dann estimate() auf demselben Objekt), und der zweite Lauf
liefe mit vorgewaermter Schaetzung.

Schnittstelle:
    class InDegreeSource
        .name
        .start(oracle) -> state
        .get(state, oracle, v) -> int    (immer >= 1)
        .observe(state, out) -> None     (Ausgangskanten eines Erstbesuchs)
    OnlineInDegree, ExactInDegree
    IN_DEGREES: dict[str, type[InDegreeSource]]
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class InDegreeSource(ABC):
    name: str = "indeg"

    def start(self, oracle):
        """Frischer Zustand je Lauf -- s. Modul-Docstring."""
        return None

    @abstractmethod
    def get(self, state, oracle, v) -> int:
        """Eingangsgrad von v; nie 0, sonst teilte b_ij durch null."""

    def observe(self, state, out) -> None:
        """Die Ausgangskanten eines gerade *erstmals* expandierten Knotens."""


class OnlineInDegree(InDegreeSource):
    """Aus den selbst beobachteten Kanten geschaetzt (Paper, 6.3).

    Jede beim Crawlen gesehene Kante i -> j ist ein Beleg fuer eine eingehende
    Kante von j. Der Schaetzer zaehlt sie mit -- und zwar nur beim *Erstbesuch*
    von i, sonst zaehlte jeder Wiederbesuch dieselbe Kante erneut. Die Information
    ist gratis: die Nachbarliste von i war ohnehin zu bezahlen.

    Ein dict, kein Array der Laenge |V|: das Verfahren soll nichts wissen, was
    ein Crawler nicht wissen kann -- so wie Oracle._fetched, Oracle.visits und
    das adj des DurwSampler nur ueber tatsaechlich Gesehenes buchfuehren.

    d- ist damit systematisch *unterschaetzt* (jede noch nicht gesehene Kante
    fehlt), b_ij also ueberschaetzt und die Annahme zu grosszuegig -- und zwar
    am staerksten bei genau den Knoten mit vielen eingehenden Kanten. Das
    korrigiert sich erst mit der Abdeckung; wie stark es bei kleinen Budgets
    durchschlaegt, misst nmmc_trace.py als `frac_dinhat_1`.

    Der Boden bei 1 ist eine Nulldivisionsschranke, keine Vorabmasse: ein noch
    nie als Ziel gesehener Knoten bekommt d- = 1, ein einmal gesehener
    ebenfalls. Woertlich gelesen setzt das Paper d- := 1 *und* erhoeht dann je
    Kante, gibt einem einmal gesehenen Knoten also 2. Bei der Abdeckung hier
    liegt fast die ganze Stichprobe in diesen beiden Faellen, der Unterschied
    waere also ein Faktor 2 in b ueber weite Strecken -- deshalb steht er hier
    als bewusste Wahl und nicht als Beilaeufigkeit.
    """

    name = "online"

    def start(self, oracle):
        return {}

    def get(self, state, oracle, v) -> int:
        d = state.get(v, 0)
        return d if d else 1

    def observe(self, state, out) -> None:
        # .tolist() gibt echte Python-ints: als dict-Schluessel schneller und
        # mit den int(...) des Samplers identisch. Dass dabei keine Kante
        # doppelt gezaehlt wird, haengt an graphs.graph._simplify() -- es
        # entfernt Mehrfachkanten schon beim Laden.
        for v in out.tolist():
            state[v] = state.get(v, 0) + 1


class ExactInDegree(InDegreeSource):
    """Der wahre Eingangsgrad -- Vergleichsvariante.

    Braucht oracles.local_access.InDegreeCrawlOracle und damit eine global
    berechnete Groesse. Der Zugriff kostet dort nichts (Begruendung im
    Docstring des Oracles).
    """

    name = "exact"

    def get(self, state, oracle, v) -> int:
        d = oracle.in_degree(v)
        # Nur fuer Knoten ohne jede eingehende Kante -- die kann der Walk zwar
        # nicht ueber eine Kante erreichen, wohl aber als Einstiegsknoten.
        return d if d else 1


class _CrossInDegree(InDegreeSource):
    """Gemeinsamer Teil der beiden cross-Varianten.

    Der Index ist vorab gebaut (build_indeg_index.py) und liegt am Oracle: zur
    Laufzeit wird kein Name angefasst, die Abfrage ist ein Array-Zugriff. -1
    heisst "im Partner nicht vorhanden" -- nur dann greift der Fallback.

    Ein Grad 0 im Partner ist dagegen *kein* Fehlschlag, sondern die Auskunft
    "dort kennt niemand diese Entitaet": auf 1 gehoben wie ueberall, weil b_ij
    sonst durch null teilte.
    """

    def get(self, state, oracle, v) -> int:
        d = oracle.cross_in_degree(v)
        if d >= 0:
            return d if d else 1
        return self.fallback(state, v)

    def fallback(self, state, v) -> int:
        raise NotImplementedError


class CrossOneInDegree(_CrossInDegree):
    """Partnergraph, fehlende Entitaeten bekommen 1.

    Die schlichte Variante: was der Partner nicht kennt, gilt als kaum
    verlinkt. Das ueberschaetzt b_ij fuer genau die Knoten, ueber die am
    wenigsten bekannt ist -- der Walk nimmt Zuege dorthin also besonders
    bereitwillig an. Ob das hilft (mehr Diffusion) oder schadet (falsche QSD),
    ist der Vergleich zu cross-online.
    """

    name = "cross-one"

    def fallback(self, state, v) -> int:
        return 1


class CrossOnlineInDegree(_CrossInDegree):
    """Partnergraph, fehlende Entitaeten aus den selbst beobachteten Kanten.

    Nie schlechter informiert als `online` und im Kern des Partners deutlich
    besser. Der Preis ist, dass zwei verschiedene Groessen in dieselbe Formel
    gehen: ein Grad aus dem Partner und ein selbst gezaehlter, die
    systematisch verschieden skalieren (der gezaehlte ist untererfasst, siehe
    OnlineInDegree). Wie stark das stoert, misst nmmc_trace.py.
    """

    name = "cross-online"

    def start(self, oracle):
        return {}

    def observe(self, state, out) -> None:
        for v in out.tolist():
            state[v] = state.get(v, 0) + 1

    def fallback(self, state, v) -> int:
        d = state.get(v, 0)
        return d if d else 1


IN_DEGREES: dict[str, type[InDegreeSource]] = {
    "online": OnlineInDegree,
    "exact": ExactInDegree,
    "cross-one": CrossOneInDegree,
    "cross-online": CrossOnlineInDegree,
}
