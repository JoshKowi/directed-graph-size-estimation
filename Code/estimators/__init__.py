"""Registry aller Estimators -- hier wird ein neues Verfahren eingetragen.

Ein Eintrag besteht aus einer Factory (ohne Argumente bzw. mit Defaults) und
der Kategorie. Die Kategorie haengt *nicht* am Estimator-Modul, sondern wird
erst hier vergeben: ob ein Verfahren real umsetzbar ist, entscheidet das
Oracle, und dasselbe Verfahren kann mit anderem Oracle in die andere Kategorie
fallen. `build()` setzt das Label nach der Konstruktion auf die Instanz.

Die Random-Walk-Varianten werden als Kreuzprodukt erzeugt:
    Sackgassen-Strategie (restart | backtrack | history)
  x Umgang mit Abhaengigkeit (none | simple | shifted | margin)

Dieselbe Form haben die beiden Verfahren mit bekannter Verteilung auf der
gerichteten Sicht: bei DURW steht die Sprungart im ersten Slot, bei NMMC die
Herkunft des Eingangsgrades (online | exact). Beide entscheiden ueber das
Oracle und damit ueber die Kategorie -- deshalb stehen sie dort, wo beim
Random Walk die Sackgassen-Strategie steht.

`margin` steht im selben Namensslot wie das Thinning, ist aber keines: es
verwirft keine Samples, sondern laesst bei der Kollisionszaehlung Paare aus,
die im Walk weniger als m+1 Schritte auseinanderliegen (estimators.formulas).
Die Groesse kommt aus config.SAFETY_MARGIN und laesst sich im Namen
ueberschreiben: `rw-plain__restart__margin20` benutzt m = 20. Dasselbe gilt
fuer die Zahl der Faenge bei Schnabel (`capture-recapture__restart__schnabel8`).
Solche Namen stehen nicht in der REGISTRY, `build()` loest sie zur Laufzeit
auf -- `names()` und `build_all()` listen deshalb nur die Default-Variante.

Schnittstelle:
    REGISTRY: dict[str, Entry]
    register(name, factory, category)
    build(name) -> Estimator            (setzt .name und .category)
    build_all(selected=None, category=None) -> list[Estimator]
    names(category=None) -> list[str]
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import config
import namelists
from estimators.base import Category, Estimator
from estimators.formulas import FORMULAS
from estimators.methods import (capture_recapture, deg_weighted_independent,
                                dufs, durw, name_list_collision, nmmc,
                                random_walk_collision, short_walk_independent,
                                uniform_collision)
from sampling.dead_ends import DEAD_ENDS
from sampling.indegree import IN_DEGREES
from sampling.jumps import JUMPS
from sampling.thinning import THINNINGS


@dataclass(frozen=True)
class Entry:
    factory: Callable[[], Estimator]
    category: Category


REGISTRY: dict[str, Entry] = {
    # -- Vergleich: gleichverteiltes Ziehen aus V ------------------------
    "uniform-collision": Entry(
        partial(uniform_collision.build, formula="uis-collision"), Category.COMPARISON),
    "uniform-collision__weighted": Entry(
        partial(uniform_collision.build, formula="wis-col-katzir"), Category.COMPARISON),
    # Capture-Recapture ohne Grad-/Walk-Verzerrung: dieselben Formeln, aber mit
    # UniformSampler statt RandomWalkSampler (s. capture_recapture.build).
    # dead_end ist dabei wirkungslos, also nur einmal, nicht je Dead-End.
    "capture-recapture__uniform": Entry(
        partial(capture_recapture.build, sampler="uniform"), Category.COMPARISON),
    "capture-recapture__uniform__chapman": Entry(
        partial(capture_recapture.build, sampler="uniform", formula="chapman"),
        Category.COMPARISON),
    "capture-recapture__uniform__schnabel": Entry(
        partial(capture_recapture.build, sampler="uniform", formula="schnabel",
                n_captures=config.DEFAULT_CAPTURES),
        Category.COMPARISON),
}
for _cf in ("cross", "cross-wis"):
    REGISTRY[f"capture-recapture__uniform__{_cf}"] = Entry(
        partial(capture_recapture.build, sampler="uniform", formula=_cf),
        Category.COMPARISON)
del _cf

# -- Real umsetzbar: Random Walk x Sackgassen-Strategie x Thinning -------
for _dead_end in DEAD_ENDS:
    for _thinning in THINNINGS:
        REGISTRY[f"rw-plain__{_dead_end}__{_thinning}"] = Entry(
            partial(random_walk_collision.build,
                    dead_end=_dead_end, thinning=_thinning, formula="uis-collision"),
            Category.REALIZABLE,
        )
    # Safety Margin: vierter Wert im Thinning-Slot, aber kein Thinning (s.o.).
    # Immer mit thinning="none" -- beide zusammen waeren doppelt gemoppelt.
    REGISTRY[f"rw-plain__{_dead_end}__margin"] = Entry(
        partial(random_walk_collision.build, dead_end=_dead_end, thinning="none",
                margin=config.SAFETY_MARGIN, formula="uis-collision"),
        Category.REALIZABLE,
    )
    # Capture-Recapture: dieselben Faenge, drei Formeln darueber. Der Name ohne
    # Zusatz bleibt Lincoln-Petersen, damit vorhandene Aufrufe weiter gelten.
    REGISTRY[f"capture-recapture__{_dead_end}"] = Entry(
        partial(capture_recapture.build, dead_end=_dead_end), Category.REALIZABLE)
    REGISTRY[f"capture-recapture__{_dead_end}__chapman"] = Entry(
        partial(capture_recapture.build, dead_end=_dead_end, formula="chapman"),
        Category.REALIZABLE)
    REGISTRY[f"capture-recapture__{_dead_end}__schnabel"] = Entry(
        partial(capture_recapture.build, dead_end=_dead_end, formula="schnabel",
                n_captures=config.DEFAULT_CAPTURES),
        Category.REALIZABLE)
    # Kollisionen zwischen den Faengen -- die einzige Capture-Recapture-Form,
    # die sich gradkorrigieren laesst. Beide Gewichtungen auf denselben Faengen.
    for _cf in ("cross", "cross-wis"):
        REGISTRY[f"capture-recapture__{_dead_end}__{_cf}"] = Entry(
            partial(capture_recapture.build, dead_end=_dead_end, formula=_cf),
            Category.REALIZABLE)

# -- WIS-Vergleichsreihe -------------------------------------------------
# Dieselbe Formel (Katzir) auf zwei Sampling-Verfahren mit
# derselben Verteilung pi(v) ~ deg(v):
#   *__indep : unabhaengige Ziehungen  -> nur Gradverzerrung
#   *__rw-*  : echter Random Walk      -> Gradverzerrung + Autokorrelation
# Die Differenz ist der Preis der Abhaengigkeit. Referenz fuer beide ist
# "uniform-collision" (gleichverteilt, ohne Gewicht).
REGISTRY["wis-katzir__indep"] = Entry(
    partial(deg_weighted_independent.build, formula="wis-col-katzir"),
    Category.COMPARISON)
for _de in DEAD_ENDS:
    REGISTRY[f"wis-katzir__rw-{_de}"] = Entry(
        partial(random_walk_collision.build,
                dead_end=_de, thinning="none", formula="wis-col-katzir"),
        Category.REALIZABLE,
    )
    # dieselbe Formel auf den echten Thinnings -- Gegenstueck zu
    # rw-plain__<dead_end>__<thinning>, nur gradgewichtet (wis-col-katzir)
    # statt uis-collision.
    for _th in THINNINGS:
        if _th == "none":
            continue
        REGISTRY[f"wis-katzir__rw-{_de}__{_th}"] = Entry(
            partial(random_walk_collision.build,
                    dead_end=_de, thinning=_th, formula="wis-col-katzir"),
            Category.REALIZABLE,
        )
    # dieselbe Formel mit Safety Margin -- auf `undirected` die interessantere
    # Reihe, weil dort pi ~ deg stimmt und nur die Abhaengigkeit stoert
    REGISTRY[f"wis-katzir__rw-{_de}__margin"] = Entry(
        partial(random_walk_collision.build, dead_end=_de, thinning="none",
                margin=config.SAFETY_MARGIN, formula="wis-col-katzir"),
        Category.REALIZABLE,
    )
del _de, _th

# -- Kurze unabhaengige Walks -------------------------------------------
# Endknoten eines 5-Schritt-Walks je Sample: Walk-Verzerrung ohne
# Autokorrelation. Beide Formeln auf denselben Samples, damit ablesbar ist, was
# die Gradgewichtung bewirkt -- auf `undirected` passt sie, auf `directed`
# nicht. Ein Sample kostet eine Query, die Schritte sind gratis.
for _f, _tag in (("wis-col-katzir", "wis-katzir"), ("uis-collision", "uis")):
    REGISTRY[f"{_tag}__walk5"] = Entry(
        partial(short_walk_independent.build, formula=_f, steps=5),
        Category.COMPARISON)
del _f, _tag

# Gradkorrigierte Referenz -- zeigt, was die Verzerrung des Walks ausmacht.
REGISTRY["rw-weighted__restart__none"] = Entry(
    partial(random_walk_collision.build,
            dead_end="restart", thinning="none", formula="wis-col-katzir"),
    Category.REALIZABLE,
)

del _dead_end, _thinning, _cf

# -- DURW: Random Walk mit aufgebautem G_u und gradproportionalem Sprung ---
# Dieselben Formeln, Thinnings und Capture-Recapture-Varianten wie beim
# einfachen Random Walk, nur mit einem Sampler, dessen Stationaerverteilung
# auch auf den gerichteten Views bekannt ist (sampling.durw). An der Stelle
# von `dead_end` steht die Sprungart -- eine Sackgassen-Strategie braucht DURW
# nicht, weil eine Sackgasse dort nie absorbierend wird (s. sampling.durw).
#
# Die Kategorie haengt daran, wie der Sprung beschafft wird, nicht am Sampler:
# `uniform` zieht gleichverteilt aus V und setzt damit dieselbe Kenntnis der
# Knotenmenge voraus, die auch `uniform-collision` zur COMPARISON macht. Die
# Listenquellen (namelists.SOURCES) brauchen sie nicht -- sie ziehen aus
# externem Wissen und sind deshalb REALIZABLE. Dass sie den Graphen nur zu 8
# bis 38 % abdecken, ist eine Frage der Guete, nicht der Umsetzbarkeit; was
# das fuer die Stationaerverteilung bedeutet, steht in oracles/name_list.py.
_JUMP_CATEGORY = {"uniform": Category.COMPARISON}
# Die In-Grad-Listen stammen aus dem Graphen selbst und sind deshalb
# COMPARISON, nicht REALIZABLE -- namelists.NameList.realizable sagt es.
_JUMP_CATEGORY.update({
    _s: (Category.REALIZABLE if _l.realizable else Category.COMPARISON)
    for _s, _l in namelists.SOURCES.items()})

for _jump in ("uniform",):
    _cat = _JUMP_CATEGORY[_jump]
    for _thinning in THINNINGS:
        REGISTRY[f"durw-plain__{_jump}__{_thinning}"] = Entry(
            partial(durw.build, jump=_jump, thinning=_thinning,
                    formula="uis-collision"), _cat)
        # gradkorrigiert; ohne Thinning heisst der Eintrag nur "wis-durw__<jump>",
        # analog zu wis-katzir__rw-<dead_end>
        _name = (f"wis-durw__{_jump}" if _thinning == "none"
                 else f"wis-durw__{_jump}__{_thinning}")
        REGISTRY[_name] = Entry(
            partial(durw.build, jump=_jump, thinning=_thinning,
                    formula="wis-col-katzir"), _cat)
    # Safety Margin: wie oben immer mit thinning="none".
    for _tag, _f in (("durw-plain", "uis-collision"), ("wis-durw", "wis-col-katzir")):
        REGISTRY[f"{_tag}__{_jump}__margin"] = Entry(
            partial(durw.build, jump=_jump, thinning="none",
                    margin=config.SAFETY_MARGIN, formula=_f), _cat)
    # w-Sweep: dieselbe Variante wie wis-durw__<jump>__margin, nur mit
    # explizitem Sprunggewicht. Bewusst auf diese eine Variante beschraenkt --
    # gefragt ist die Wirkung von w, nicht die von w x Thinning x Formel.
    # w=1 ist mit dabei, obwohl es wis-durw__<jump>__margin entspricht: die
    # Plot-Legende liest sich dadurch einheitlich, und weil beide denselben
    # walk_key haben, kostet der Doppeleintrag mit --share-walks nichts.
    # Das w steht *vor* dem margin-Slot, damit "...__margin<N>" weiter greift.
    for _w in config.DURW_JUMP_WEIGHTS:
        REGISTRY[f"wis-durw__{_jump}__w{_w:g}__margin"] = Entry(
            partial(durw.build, jump=_jump, thinning="none",
                    margin=config.SAFETY_MARGIN, formula="wis-col-katzir",
                    jump_weight=_w), _cat)
    # Capture-Recapture auf DURW-Faengen -- jeder Fang baut sein eigenes G_u.
    REGISTRY[f"capture-recapture__durw-{_jump}"] = Entry(
        partial(capture_recapture.build, sampler="durw", jump=_jump), _cat)
    REGISTRY[f"capture-recapture__durw-{_jump}__chapman"] = Entry(
        partial(capture_recapture.build, sampler="durw", jump=_jump,
                formula="chapman"), _cat)
    REGISTRY[f"capture-recapture__durw-{_jump}__schnabel"] = Entry(
        partial(capture_recapture.build, sampler="durw", jump=_jump,
                formula="schnabel", n_captures=config.DEFAULT_CAPTURES), _cat)
    for _cf in ("cross", "cross-wis"):
        REGISTRY[f"capture-recapture__durw-{_jump}__{_cf}"] = Entry(
            partial(capture_recapture.build, sampler="durw", jump=_jump,
                    formula=_cf), _cat)

# DURW ohne Spruenge (w -> 0, sampling.durw no_jumps): reiner Random Walk auf
# G_u. Braucht keine Kenntnis von V und keine Sprungquelle -- REALIZABLE ohne
# Einschraenkung, staerker noch als "uniform" (das dafuer COMPARISON ist).
for _tag, _f in (("durw-plain", "uis-collision"), ("wis-durw", "wis-col-katzir")):
    REGISTRY[f"{_tag}__nojump"] = Entry(
        partial(durw.build, thinning="none", formula=_f, no_jumps=True),
        Category.REALIZABLE)
REGISTRY["wis-durw__nojump__margin"] = Entry(
    partial(durw.build, thinning="none", formula="wis-col-katzir",
            margin=config.SAFETY_MARGIN, no_jumps=True),
    Category.REALIZABLE)

del _jump, _cat, _thinning, _name, _tag, _f, _cf, _w

# -- Ziehung aus externen Namenslisten ------------------------------------
# Zwei Verwendungen derselben Ziehung (oracles.name_list.NameListOracle):
#
#   namelist-<quelle>__b<n>   unabhaengige Stichprobe, Gegenstueck zu
#                             "uniform-collision" -- misst die Ziehung selbst
#   durw-<quelle>__b<n>       dieselbe Ziehung als DURW-Sprung
#
# `b<n>` ist der Burn-in *nach jedem Treffer* (config.DEFAULT_DRAW_BURN_IN),
# nicht der des Walks. Er soll die Verzerrung der Liste abbauen -- getroffene
# Knoten haben deutlich hoeheren Grad als verfehlte -- und fuehrt dafuer die
# Verzerrung eines Random-Walk-Schritts ein. Deshalb die Reihe statt eines
# festen Werts.
#
# Bewusst kein Kreuzprodukt ueber Thinnings und Formeln: gefragt ist die
# Wirkung der Ziehung, nicht die von Ziehung x Thinning x Formel. Wer mehr
# braucht, ruft die build()-Funktionen direkt auf.
# `n<N>` ist die Listenlaenge: nur die ersten N Eintraege der (nach Relevanz
# sortierten) Quelle. Sie steht *vor* dem margin-Slot, damit "...__margin<N>"
# weiter aufgeloest wird -- dieselbe Stellung wie beim w-Sweep. Ohne `n<N>`
# meint der Eintrag die volle Liste.
#
# Laengere Liste heisst mehr erreichbare Knoten *und* mehr Fehlschlaege; wo der
# Handel kippt, zeigen die Kurven aus plotting/name_coverage.py. Genau diese
# Achse faehrt der Sweep ab.
for _src in sorted(namelists.SOURCES):
    _cat = _JUMP_CATEGORY[_src]
    for _b in config.DRAW_BURN_INS:
        for _n in (None,) + tuple(config.DRAW_LIMITS):
            _tag = f"__n{_n}" if _n else ""
            REGISTRY[f"namelist-{_src}{_tag}__b{_b}"] = Entry(
                partial(name_list_collision.build, source=_src, draw_burn_in=_b,
                        draw_limit=_n, formula="uis-collision"), _cat)
            REGISTRY[f"durw-{_src}{_tag}__b{_b}__margin"] = Entry(
                partial(durw.build, jump=_src, thinning="none", draw_burn_in=_b,
                        draw_limit=_n, margin=config.SAFETY_MARGIN,
                        formula="wis-col-katzir"), _cat)
            # Korrigierte Gewichtung -- **Abweichung vom Original**. Der Sprung
            # aus einer Liste erreicht nur die Trefferteilmenge S, sigma ist
            # also nicht mit ganz V verbunden und pi(v) ~ deg_Gu(v) + w*1[v in S]
            # statt pi(v) ~ w + deg_Gu(v). Nur fuer b = 0: mit Burn-in liefert
            # der Sprung Knoten ausserhalb von S und die Verteilung ist wieder
            # unbekannt (s. weighting.DurwJumpSetWeighting).
            #
            # Eigener Name statt Umwidmung, damit die vorhandenen Ergebnisse
            # unter durw-<quelle>__* ihre Bedeutung behalten -- der Vergleich
            # naiv gegen korrigiert ist selbst ein Ergebnis. Beide teilen sich
            # bei --share-walks eine Trajektorie, der Vergleich ist also gepaart.
            # Sprung auf S u H -- die theoretisch saubere Fassung. **Ebenfalls
            # eine Abweichung vom Original**: sigma ist nicht mit ganz V
            # verbunden, sondern mit Liste und Historie. Sprungregel *und*
            # Gewicht folgen daraus, die Kette bleibt reversibel und
            # pi(u) ~ deg_Gu + w*(m(u) + 1). Siehe weighting.DurwSigmaWeighting.
            # durwset-* bleibt fuer die vorhandenen Ergebnisse stehen, ist aber
            # widerlegt: dort springt jeder Knoten, sigma landet nur auf S, und
            # die Kette hat keine geschlossene Stationaerverteilung.
            if _b == 0:
                REGISTRY[f"durwhist-{_src}{_tag}__b0__margin"] = Entry(
                    partial(durw.build, jump=_src, thinning="none",
                            draw_burn_in=0, draw_limit=_n,
                            margin=config.SAFETY_MARGIN,
                            formula="wis-col-katzir",
                            history_jumps=True), _cat)
                # Sprung gleichverteilt auf S u H, jeder Knoten nur einmal in
                # der Liste -- Absprungregel und Gewicht des Originals. Die
                # kleinste Abweichung vom Paper: sigma ist mit S u H statt mit
                # V verbunden (sampling.durw, `union_jumps`).
                REGISTRY[f"durwunion-{_src}{_tag}__b0__margin"] = Entry(
                    partial(durw.build, jump=_src, thinning="none",
                            draw_burn_in=0, draw_limit=_n,
                            margin=config.SAFETY_MARGIN,
                            formula="wis-col-katzir",
                            union_jumps=True), _cat)
                # Dieselbe Idee, aber H sind alle *gesehenen* Knoten statt nur
                # der besuchten (sampling.durw, `union_seen`) -- wächst
                # schneller, weil auch unbesuchte Nachbarn dazuzählen.
                REGISTRY[f"durwunionE-{_src}{_tag}__b0__margin"] = Entry(
                    partial(durw.build, jump=_src, thinning="none",
                            draw_burn_in=0, draw_limit=_n,
                            margin=config.SAFETY_MARGIN,
                            formula="wis-col-katzir",
                            union_seen=True), _cat)
                REGISTRY[f"durwset-{_src}{_tag}__b0__margin"] = Entry(
                    partial(durw.build, jump=_src, thinning="none",
                            draw_burn_in=0, draw_limit=_n,
                            margin=config.SAFETY_MARGIN,
                            formula="wis-col-katzir",
                            jump_set_weighting=True), _cat)
del _src, _cat, _b, _n, _tag

# -- IE2: Kreuzkollisionen statt Knoten-Kollisionen -----------------------
# Alle Eintraege oben zaehlen Wiederholungen (s_i == s_j). IE2 zaehlt Treffer
# gegen A = Vereinigung aller beobachteten Nachbarschaften: ein Nachbar wird mit
# ~<k>/N getroffen, ein Knoten selbst nur mit ~1/N, also gibt es aus demselben
# Budget rund <k>-mal mehr verwertbare Ereignisse (estimators.formulas.
# IE2SetEstimator). A kostet nichts -- die Nachbarlisten sind beim Erstbesuch
# ohnehin bezahlt, der Sampler warf sie bisher nur weg (sampling.observed).
#
# Zwei Achsen, beide aus der Vorlage und beide offen:
#
#   ie2-   A als Set (Duplikate verworfen) -- die vom Paper *empfohlene* Form.
#   ie2m-  A als Multiset -- die fuer Random Walks *hingeschriebene* Form
#          (Gl. 24). Dort gilt n^xcol == n^IE.
#   __gu   A aus der eingefrorenen G_u-Nachbarschaft statt aus den rohen
#          Ausgangsnachbarn. Konsistent mit dem Gewicht, aber kleineres A.
#
# Die Kategorie ist dieselbe wie bei der jeweiligen Kollisions-Variante: IE2
# aendert nur, *was* gezaehlt wird, nicht welches Wissen der Lauf braucht.
#
# Immer mit thinning="none" und Margin -- ohne Margin ist IE2 auf einer
# RW-Stichprobe wertlos, nicht nur ungenau (s. IE2SetEstimator, Abschnitt
# "Warum der Margin hier nicht optional ist"). Der m-Sweep laeuft ueber die
# generische Aufloesung "...__margin<N>" weiter unten und braucht keine
# eigenen Eintraege.
_IE2_FORMS = (("ie2", "ie2-xcol"), ("ie2m", "ie2m-xcol"))
_IE2_A = (("", "raw"), ("__gu", "gu"))

for _form, _f in _IE2_FORMS:
    for _atag, _a in _IE2_A:
        # Ohne Sprung -- die eigentliche Frage: verbessert IE2 den einzigen
        # DURW, der ohne jede Kenntnis von V laeuft?
        REGISTRY[f"{_form}-durw__nojump{_atag}__margin"] = Entry(
            partial(durw.build, thinning="none", formula=_f, a_source=_a,
                    margin=config.SAFETY_MARGIN, no_jumps=True),
            Category.REALIZABLE)
        # Gleichverteilter Sprung -- COMPARISON, als obere Schranke daneben.
        REGISTRY[f"{_form}-durw__uniform{_atag}__margin"] = Entry(
            partial(durw.build, jump="uniform", thinning="none", formula=_f,
                    a_source=_a, margin=config.SAFETY_MARGIN),
            _JUMP_CATEGORY["uniform"])
        # Listenquellen, volle Liste und b = 0: die realisierbaren Varianten.
        # Bewusst ohne den b- und n-Sweep der Kollisions-Eintraege -- gefragt
        # ist die Wirkung von IE2, nicht die von IE2 x Burn-in x Listenlaenge.
        # IE2 korrigiert die pi-Abweichung der Listenspruenge nicht (s.
        # weighting.DurwJumpSetWeighting); die Gewichtung bleibt dieselbe wie
        # bei durw-<quelle>__b0__margin bzw. durwunion-<quelle>__b0__margin.
        for _src in sorted(namelists.SOURCES):
            REGISTRY[f"{_form}-durw-{_src}{_atag}__b0__margin"] = Entry(
                partial(durw.build, jump=_src, thinning="none", draw_burn_in=0,
                        margin=config.SAFETY_MARGIN, formula=_f, a_source=_a),
                _JUMP_CATEGORY[_src])
            REGISTRY[f"{_form}-durwunion-{_src}{_atag}__b0__margin"] = Entry(
                partial(durw.build, jump=_src, thinning="none", draw_burn_in=0,
                        margin=config.SAFETY_MARGIN, formula=_f, a_source=_a,
                        union_jumps=True),
                _JUMP_CATEGORY[_src])
            REGISTRY[f"{_form}-durwunionE-{_src}{_atag}__b0__margin"] = Entry(
                partial(durw.build, jump=_src, thinning="none", draw_burn_in=0,
                        margin=config.SAFETY_MARGIN, formula=_f, a_source=_a,
                        union_seen=True),
                _JUMP_CATEGORY[_src])

del _form, _f, _atag, _a, _src

# -- Sprung auf eine Zufallsteilmenge -------------------------------------
# Dieselben drei Sprungvarianten wie bei den Namenslisten, nur trifft der
# Sprung eine gleichverteilt gezogene Teilmenge S mit P % der Knoten
# (oracles.random_subset) statt der Treffer einer Liste:
#
#   durw-rand<P>__b0__margin       naive Gewichtung 1/(w + deg_Gu)
#   durwset-rand<P>__b0__margin    Sprung von ueberall, Landung nur auf S
#                                  (widerlegt, s. oben -- zum Vergleich)
#   durwhist-rand<P>__b0__margin   Sprung auf S u H, reversibel, S n H doppelt
#   durwunion-rand<P>__b0__margin  Sprung gleichverteilt auf S u H, Gewicht
#                                  des Originals; bei P = 100 das Original
#   durwunionE-rand<P>__b0__margin dasselbe, aber H sind alle *gesehenen*
#                                  Knoten statt nur der besuchten
#
# Damit laufen die Varianten auf jedem Graphen, auch ohne Namensliste, und die
# Frage "was kostet es, dass der Sprung nur einen Teil von V erreicht?" laesst
# sich getrennt von der Gradverzerrung einer echten Liste stellen. S steht je
# Seed fest (alle Laeufe eines Experiments teilen es), ein neuer Seed zieht ein
# neues. "b0" steht nur der Einheitlichkeit halber im Namen -- einen Burn-in
# nach dem Sprung gibt es hier nicht.
#
# Kategorie Vergleich: S aus V zu ziehen setzt voraus, V zu kennen.
# Beliebige Anteile ("durwhist-rand2.5__b0__margin") loest build() auf.
# Dieselben Varianten mit IE2 statt Knoten-Kollisionen ("ie2-durw-rand10__
# b0__margin", "ie2m-durwunion-rand2.5__b0__margin", je auch mit "__gu"). Nur
# fuer durw, durwunion und durwunionE -- durwset ist widerlegt und durwhist
# wird nicht mehr gebraucht (s. oben), eine IE2-Fassung davon waere toter
# Ballast.
_RAND_VARIANTS = {"durw": {}, "durwset": {"jump_set_weighting": True},
                  "durwhist": {"history_jumps": True},
                  "durwunion": {"union_jumps": True},
                  "durwunionE": {"union_seen": True}}
_IE2_RAND_VARIANTS = {"durw": {}, "durwunion": {"union_jumps": True},
                      "durwunionE": {"union_seen": True}}


def _rand_entry(variant: str, percent: float, formula: str = "wis-col-katzir",
                a_source: str = "raw") -> Entry:
    extra = ({"a_source": a_source}
             if FORMULAS[formula].needs_neighbors else {})
    return Entry(partial(durw.build, jump=f"rand{percent:g}", thinning="none",
                         margin=config.SAFETY_MARGIN, formula=formula,
                         **extra, **_RAND_VARIANTS[variant]),
                 Category.COMPARISON)


for _p in config.JUMP_SUBSET_PERCENTS:
    for _v in _RAND_VARIANTS:
        REGISTRY[f"{_v}-rand{_p:g}__b0__margin"] = _rand_entry(_v, _p)
del _p, _v

# w-Sweep auf den In-Grad-Kreuzlisten -- das Gegenstueck zu
# wis-durw__uniform__w<W>__margin, nur mit simuliertem statt gleichverteiltem
# Sprung. Bewusst schmal: nur die aus Phase 1 gewaehlte Laenge n = 100000 und
# Burn-in 0, gefragt ist die Wirkung von w, nicht die von w x Laenge x Burn-in.
# Das w steht vor dem margin-Slot, damit "...__margin<N>" weiter greift.
for _src in ("indeg-gpt4_io", "indeg-gpt4o_io"):
    _cat = _JUMP_CATEGORY[_src]
    for _w in config.DURW_JUMP_WEIGHTS:
        REGISTRY[f"durw-{_src}__n100000__w{_w:g}__b0__margin"] = Entry(
            partial(durw.build, jump=_src, thinning="none", draw_burn_in=0,
                    draw_limit=100000, jump_weight=_w,
                    margin=config.SAFETY_MARGIN, formula="wis-col-katzir"),
            _cat)
del _src, _cat, _w

# -- DUFS: dasselbe mit k koordinierten Walkern ---------------------------
# DUFS ist DURW mit k Walkern auf einem gemeinsamen G_u (sampling.dufs).
# Stationaerverteilung, Sprungregel und Gewichtung sind identisch -- verschieden
# ist nur die Autokorrelation der Sample-Folge, und genau die entscheidet, was
# der Kollisionsschaetzer sieht. k = 1 *ist* DURW und gehoert als Gegenprobe in
# jeden Plot (Code/check_dufs.py prueft die Gleichheit bitgenau).
#
# Das k steht als `k<K>` im Namen -- `b<N>`, `n<N>` und `w<N>` sind im Repo
# schon fuer draw_burn_in, Listenlaenge und Sprunggewicht belegt. Es steht
# jeweils *vor* dem margin-Slot, damit "...__margin<N>" weiter aufgeloest wird,
# dieselbe Stellung wie beim w-Sweep.
#
# Bewusst *nicht* uebernommen: `durwhist-*` (nicht mehr gebraucht) und
# `durwset-*` (widerlegt, s. weighting.DurwJumpSetWeighting -- es steht nur
# noch da, damit die vorhandenen Ergebnisse ihre Bedeutung behalten; fuer DUFS
# gibt es keine zu erhalten).


def _dufs_entry(category: Category, **kwargs) -> Entry:
    """Die Standardachse: kein Thinning, Safety Margin, gradkorrigiert."""
    return Entry(partial(dufs.build, thinning="none",
                         margin=config.SAFETY_MARGIN,
                         formula="wis-col-katzir", **kwargs), category)


for _k in config.DUFS_WALKER_COUNTS:
    _cat = _JUMP_CATEGORY["uniform"]
    for _thinning in THINNINGS:
        REGISTRY[f"dufs-plain__uniform__k{_k}__{_thinning}"] = Entry(
            partial(dufs.build, thinning=_thinning, n_walkers=_k,
                    formula="uis-collision"), _cat)
        _name = (f"wis-dufs__uniform__k{_k}" if _thinning == "none"
                 else f"wis-dufs__uniform__k{_k}__{_thinning}")
        REGISTRY[_name] = Entry(
            partial(dufs.build, thinning=_thinning, n_walkers=_k,
                    formula="wis-col-katzir"), _cat)
    for _tag, _f in (("dufs-plain", "uis-collision"), ("wis-dufs", "wis-col-katzir")):
        REGISTRY[f"{_tag}__uniform__k{_k}__margin"] = Entry(
            partial(dufs.build, thinning="none", n_walkers=_k,
                    margin=config.SAFETY_MARGIN, formula=_f), _cat)
    # Ohne Sprung, Startknoten gleichverteilt aus V. Anders als
    # durw-plain__nojump ist das COMPARISON: die k Startknoten kommen hier aus
    # V, nicht aus externem Wissen (s. estimators/methods/dufs.py).
    REGISTRY[f"wis-dufs__uniform__k{_k}__nojump__margin"] = _dufs_entry(
        _cat, n_walkers=_k, no_jumps=True)
# w-Sweep bewusst nur beim Default-k: gefragt ist die Wirkung von w, nicht die
# von w x k. Das k-Sweep-Gegenstueck laeuft bei w = DURW_JUMP_WEIGHT.
for _w in config.DURW_JUMP_WEIGHTS:
    REGISTRY[f"wis-dufs__uniform__k{config.DUFS_WALKERS}__w{_w:g}__margin"] = \
        _dufs_entry(_JUMP_CATEGORY["uniform"], n_walkers=config.DUFS_WALKERS,
                    jump_weight=_w)

# Namenslisten. Sie liefern bei DUFS *zwei* Dinge: das Sprungziel und die k
# Startknoten -- letzteres ist der Grund, warum die nojump-Variante hier
# ueberhaupt Sinn ergibt und REALIZABLE bleibt.
for _src in sorted(namelists.SOURCES):
    _cat = _JUMP_CATEGORY[_src]
    for _n in (None,) + tuple(config.DRAW_LIMITS):
        _tag = f"__n{_n}" if _n else ""
        # k-Sweep ueber alle Listenlaengen, aber nur bei b = 0.
        for _k in config.DUFS_WALKER_COUNTS:
            REGISTRY[f"dufs-{_src}{_tag}__k{_k}__b0__margin"] = _dufs_entry(
                _cat, jump=_src, draw_burn_in=0, draw_limit=_n, n_walkers=_k)
        # Burn-in nur beim Default-k -- wie beim w-Sweep: eine Achse je Sweep.
        for _b in config.DRAW_BURN_INS:
            if _b:
                REGISTRY[f"dufs-{_src}{_tag}__k{config.DUFS_WALKERS}__b{_b}__margin"] = \
                    _dufs_entry(_cat, jump=_src, draw_burn_in=_b, draw_limit=_n,
                                n_walkers=config.DUFS_WALKERS)
    # Sprung auf S u H und die sprunglose Variante nur fuer die volle Liste:
    # gefragt ist die Wirkung von k, nicht die von k x Listenlaenge.
    for _k in config.DUFS_WALKER_COUNTS:
        REGISTRY[f"dufsunion-{_src}__k{_k}__b0__margin"] = _dufs_entry(
            _cat, jump=_src, draw_burn_in=0, n_walkers=_k, union_jumps=True)
        # k Startknoten aus der Liste, danach reiner Random Walk auf G_u. Die
        # Zusammenhangs-Einschraenkung von durw-plain__nojump faellt hier viel
        # milder aus, weil k ueber die Liste verteilte Startpunkte viele
        # Komponenten treffen (s. sampling.dufs).
        REGISTRY[f"dufs-{_src}__k{_k}__b0__nojump__margin"] = _dufs_entry(
            _cat, jump=_src, draw_burn_in=0, n_walkers=_k, no_jumps=True)
del _src, _cat, _b, _n, _tag, _k, _w, _thinning, _name, _f

# Sprung bzw. Startknoten aus einer Zufallsteilmenge -- dieselbe Achse wie
# durw-rand<P>, damit DUFS auch auf Graphen ohne Namensliste laeuft.
# Beliebige Anteile und beliebige k loest build() ueber _DUFS_RAND_RE auf.
_DUFS_RAND_VARIANTS = {"dufs": {}, "dufsunion": {"union_jumps": True}}


def _dufs_rand_entry(variant: str, percent: float, n_walkers: int,
                     no_jumps: bool = False) -> Entry:
    return _dufs_entry(Category.COMPARISON, jump=f"rand{percent:g}",
                       n_walkers=n_walkers, no_jumps=no_jumps,
                       **_DUFS_RAND_VARIANTS[variant])


for _p in config.JUMP_SUBSET_PERCENTS:
    for _k in config.DUFS_WALKER_COUNTS:
        for _v in _DUFS_RAND_VARIANTS:
            REGISTRY[f"{_v}-rand{_p:g}__k{_k}__b0__margin"] = _dufs_rand_entry(
                _v, _p, _k)
        REGISTRY[f"dufs-rand{_p:g}__k{_k}__b0__nojump__margin"] = _dufs_rand_entry(
            "dufs", _p, _k, no_jumps=True)
del _p, _k, _v

# -- NMMC: Non-Markovian Monte Carlo (Lee/Kang/Eun 2019) -----------------
# Rejection auf dem Simple Random Walk: ein abgelehnter Zug absorbiert die
# Kette, die daraufhin auf ihre eigene gewichtete Historie umverteilt wird
# (sampling.nmmc). An der Stelle von `dead_end` bzw. `jump` steht hier die
# Herkunft des Eingangsgrades -- sie entscheidet ueber das Oracle und damit
# ueber die Kategorie.
#
# Der Grund, warum das Verfahren ueberhaupt hier steht: "online" schaetzt d-
# aus selbst beobachteten Kanten und kommt mit dem reinen CrawlOracle aus.
# NMMC ist damit das erste Verfahren im Repo mit bekannter Zielverteilung auf
# der gerichteten Sicht, das *keine* gleichverteilte Ziehung aus V braucht --
# genau die macht DURW zur Vergleichsvariante. "exact" liest den wahren
# Eingangsgrad und ist die Gegenprobe dazu: nur mit ihr ist ablesbar, ob ein
# schlechtes Ergebnis am Verfahren oder an der d--Schaetzung liegt.
# "cross-*" holt den Eingangsgrad aus dem Partnergraphen (config.CROSS_GRAPHS).
# Das ist externes Wissen ueber Entitaeten und keine Kenntnis der Knotenmenge
# des geschaetzten Graphen -- dieselbe Begruendung, die die Namenslisten real
# umsetzbar macht. Damit sind sie die *realistische* Fassung von "exact", und
# die eigentliche Frage der Achse: traegt ein fremder, unvollstaendiger Prior
# das Verfahren dort, wo die reine Online-Schaetzung es nicht tut?
_INDEG_CATEGORY = {"online": Category.REALIZABLE, "exact": Category.COMPARISON,
                   "cross-one": Category.REALIZABLE,
                   "cross-online": Category.REALIZABLE}

# Ziel der QSD, zugehoerige Formel und Namenspraefix:
#   nmmc-uni    -- pi = u, ungewichtet (die Samples sind schon gleichverteilt)
#   wis-nmmc    -- pi ~ d-, gradkorrigiert (weighting.InDegreeWeighting)
#   nmmc-indeg  -- dieselben Faenge wie wis-nmmc, aber ohne Gewicht: der Preis
#                  der Verzerrung, genau wie durw-plain neben wis-durw
_NMMC_UNI = ("nmmc-uni", "uniform", "uis-collision")
_NMMC_WIS = ("wis-nmmc", "indeg", "wis-col-katzir")
_NMMC_RAW = ("nmmc-indeg", "indeg", "uis-collision")

for _indeg in IN_DEGREES:
    _cat = _INDEG_CATEGORY[_indeg]
    # Volles Kreuzprodukt mit dem Thinning nur fuer die real umsetzbare
    # Variante -- "exact" ist die Gegenprobe zur d--Schaetzung, nicht zum
    # Umgang mit Autokorrelation, und jeder Eintrag multipliziert jeden Lauf.
    if _indeg == "online":
        for _thinning in THINNINGS:
            REGISTRY[f"nmmc-uni__{_indeg}__{_thinning}"] = Entry(
                partial(nmmc.build, target="uniform", indeg=_indeg,
                        thinning=_thinning, formula="uis-collision"), _cat)
            # ohne Thinning heisst der Eintrag nur "wis-nmmc__<indeg>",
            # analog zu wis-durw__<jump> und wis-katzir__rw-<dead_end>
            _name = (f"wis-nmmc__{_indeg}" if _thinning == "none"
                     else f"wis-nmmc__{_indeg}__{_thinning}")
            REGISTRY[_name] = Entry(
                partial(nmmc.build, target="indeg", indeg=_indeg,
                        thinning=_thinning, formula="wis-col-katzir"), _cat)
    # Safety Margin: vierter Wert im Thinning-Slot, wie oben immer mit
    # thinning="none".
    for _tag, _target, _f in (_NMMC_UNI, _NMMC_WIS, _NMMC_RAW):
        REGISTRY[f"{_tag}__{_indeg}__margin"] = Entry(
            partial(nmmc.build, target=_target, indeg=_indeg, thinning="none",
                    margin=config.SAFETY_MARGIN, formula=_f), _cat)
    # alpha-Sweep wie der w-Sweep bei DURW: bewusst nur auf der
    # margin-Variante, gefragt ist die Wirkung von alpha, nicht die von
    # alpha x Thinning x Formel. alpha = NMMC_ALPHA ist mit dabei, obwohl es
    # dem Eintrag ohne alpha entspricht -- die Plot-Legende liest sich dadurch
    # einheitlich, und weil beide denselben walk_key haben, kostet der
    # Doppeleintrag mit --share-walks nichts. Das alpha steht *vor* dem
    # margin-Slot, damit "...__margin<N>" weiter greift.
    # alpha-Sweep fuer online, exact und cross-online: alpha steuert die
    # Umverteilung, nicht die d--Quelle -- online und exact spannen die Frage
    # auf, cross-online prueft sie auf dem realistischen fremden Prior.
    for _a in (config.NMMC_ALPHAS
               if _indeg in ("online", "exact", "cross-online") else ()):
        for _tag, _target, _f in (_NMMC_UNI, _NMMC_WIS):
            REGISTRY[f"{_tag}__{_indeg}__a{_a:g}__margin"] = Entry(
                partial(nmmc.build, target=_target, indeg=_indeg,
                        thinning="none", margin=config.SAFETY_MARGIN,
                        formula=_f, alpha=_a), _cat)

# _thinning/_name binden nur im "online"-Zweig -- IN_DEGREES fuehrt "online"
# deshalb zuerst (dict-Reihenfolge ist zugesichert).
del _indeg, _cat, _thinning, _name, _tag, _target, _f, _a
del _NMMC_UNI, _NMMC_WIS, _NMMC_RAW

# -- NMMC: Agentenzahl ---------------------------------------------------
# Das Paper faehrt in jeder Simulation 100 bis 10^4 Agenten; ein einzelner
# kommt dort nicht vor. Sie teilen sich hier Budget und Cache, dazu die
# Online-Schaetzung des Eingangsgrades -- eigen bleibt je Agent die Historie
# (Belegstellen im Docstring von sampling.nmmc).
#
# Nur auf dem Ziel pi ~ d-: beim Uniform-Ziel bleibt die Schaetzung gemessen
# bei 0,000-0,001, egal wie viele Agenten laufen. Ueber
# methods.nmmc.build(target="uniform", n_agents=...) ist es trotzdem direkt
# aufrufbar.
#
# Achtung Rechenzeit: mit K > 1 liest der Sampler das Budget, supports_nested
# faellt also (s. methods/nmmc.py) und --checkpoint-budgets nimmt diese
# Eintraege nicht mit -- je Budget ein eigener Lauf.
for _indeg in ("exact", "online", "cross-online"):
    _cat = _INDEG_CATEGORY[_indeg]
    for _k in config.NMMC_AGENTS:
        # k1 hat denselben walk_key wie wis-nmmc__<indeg>__margin -- mit
        # --share-walks kostet der Doppeleintrag nichts, und die Plot-Legende
        # liest sich einheitlich (wie beim w-Sweep von DURW).
        REGISTRY[f"wis-nmmc__{_indeg}__k{_k}__margin"] = Entry(
            partial(nmmc.build, target="indeg", indeg=_indeg, thinning="none",
                    margin=config.SAFETY_MARGIN, formula="wis-col-katzir",
                    n_agents=_k), _cat)
        # Gegenprobe: c_t ueber alle Agenten stehen lassen statt je Agent
        # zurueckzusetzen. Bei einem Agenten waere das identisch, deshalb erst
        # ab k > 1.
        if _k > 1:
            REGISTRY[f"wis-nmmc__{_indeg}__k{_k}__sharedc__margin"] = Entry(
                partial(nmmc.build, target="indeg", indeg=_indeg,
                        thinning="none", margin=config.SAFETY_MARGIN,
                        formula="wis-col-katzir", n_agents=_k, shared_c=True),
                _cat)

del _indeg, _cat, _k


def register(name: str, factory: Callable[[], Estimator], category: Category) -> None:
    REGISTRY[name] = Entry(factory, category)


# Namen mit angehaengter Zahl werden zur Laufzeit aufgeloest:
#   "...__margin20"    -> Eintrag "...__margin"   mit margin=20
#   "...__schnabel8"   -> Eintrag "...__schnabel" mit n_captures=8
#   "...__shifted16"   -> Eintrag "...__shifted"  mit step=16
#   "...__simple8"     -> Eintrag "...__simple"   mit step=8
_NUMBERED = {"margin": "margin", "schnabel": "n_captures",
             "shifted": "step", "simple": "step"}
_NUMBERED_RE = re.compile(r"^(?P<base>.+__(?P<kind>" + "|".join(_NUMBERED)
                          + r"))(?P<n>\d+)$")


# "<variante>-rand<P>__b0__margin" mit beliebigem Anteil P (s. oben)
_RAND_RE = re.compile(r"^(?P<v>" + "|".join(_RAND_VARIANTS)
                      + r")-rand(?P<p>\d+(?:\.\d+)?)__b0__margin$")

# Dasselbe mit IE2: "ie2-durw-rand10__b0__margin",
# "ie2m-durwunion-rand2.5__gu__b0__margin"
_IE2_RAND_RE = re.compile(r"^(?P<f>ie2m?)-(?P<v>"
                          + "|".join(_IE2_RAND_VARIANTS)
                          + r")-rand(?P<p>\d+(?:\.\d+)?)(?P<a>__gu)?"
                          + r"__b0__margin$")

# Dieselben beiden freien Achsen bei DUFS -- Anteil P *und* Walker-Zahl k:
#   "dufs-rand2.5__k250__b0__margin", "dufsunion-rand10__k7__b0__margin",
#   "dufs-rand10__k250__b0__nojump__margin"
_DUFS_RAND_RE = re.compile(r"^(?P<v>" + "|".join(_DUFS_RAND_VARIANTS)
                           + r")-rand(?P<p>\d+(?:\.\d+)?)__k(?P<k>\d+)"
                           + r"__b0(?P<nj>__nojump)?__margin$")
# Beliebiges k beim gleichverteilten Sprung:
#   "wis-dufs__uniform__k250__margin", "wis-dufs__uniform__k250__nojump__margin"
_DUFS_UNIFORM_RE = re.compile(
    r"^wis-dufs__uniform__k(?P<k>\d+)(?P<nj>__nojump)?__margin$")


def _lookup(name: str):
    entry = REGISTRY.get(name)
    if entry is not None:
        return entry
    m = _RAND_RE.match(name)
    if m and 0 < float(m.group("p")) <= 100:
        return _rand_entry(m.group("v"), float(m.group("p")))
    m = _IE2_RAND_RE.match(name)
    if m and 0 < float(m.group("p")) <= 100:
        return _rand_entry(m.group("v"), float(m.group("p")),
                           formula=f"{m.group('f')}-xcol",
                           a_source="gu" if m.group("a") else "raw")
    m = _DUFS_RAND_RE.match(name)
    # dufsunion ohne Sprung gibt es nicht -- dufs.build() wuerde es ablehnen,
    # aber ein KeyError sagt hier deutlicher, dass der Name nicht existiert.
    if (m and 0 < float(m.group("p")) <= 100 and int(m.group("k")) >= 1
            and not (m.group("v") == "dufsunion" and m.group("nj"))):
        return _dufs_rand_entry(m.group("v"), float(m.group("p")),
                                int(m.group("k")), no_jumps=bool(m.group("nj")))
    m = _DUFS_UNIFORM_RE.match(name)
    if m and int(m.group("k")) >= 1:
        return _dufs_entry(_JUMP_CATEGORY["uniform"], n_walkers=int(m.group("k")),
                           no_jumps=bool(m.group("nj")))
    return None


def build(name: str) -> Estimator:
    entry, kwargs = _lookup(name), {}
    if entry is None:
        m = _NUMBERED_RE.match(name)
        if m:
            entry = _lookup(m.group("base"))
            kwargs = {_NUMBERED[m.group("kind")]: int(m.group("n"))}
    if entry is None:
        raise KeyError(
            f"{name!r} ist kein bekannter Estimator. Bekannt sind: "
            f"{', '.join(sorted(REGISTRY))} (dazu '...__margin<N>' fuer einen "
            "abweichenden Safety Margin bzw. '...__shifted<N>'/'...__simple<N>' "
            "fuer eine abweichende Thinning-Schrittweite, "
            "'<durw|durwset|durwhist|durwunion|durwunionE>-rand<P>__b0__margin' "
            "fuer einen Sprung auf eine Zufallsteilmenge mit P % der Knoten, "
            "0 < P <= 100, "
            "'<ie2|ie2m>-<durw|durwunion|durwunionE>-rand<P>[__gu]__b0__margin' fuer "
            "dieselbe Teilmenge mit IE2, sowie "
            "'<dufs|dufsunion>-rand<P>__k<K>__b0[__nojump]__margin' und "
            "'wis-dufs__uniform__k<K>[__nojump]__margin' fuer eine abweichende "
            "Walker-Zahl K)."
        )
    # partial-Keywords werden von Aufruf-Keywords ueberschrieben
    est = entry.factory(**kwargs)
    est.name = name
    est.category = entry.category  # Label erst hier, nach der Konstruktion
    return est


def applicable(ests: list, graph_name: str) -> tuple[list, list[str]]:
    """Estimators aufteilen in "laeuft auf diesem Graphen" und "nicht".

    Manche Oracles setzen etwas am Graphen voraus, das es nicht ueberall gibt
    -- CrossInDegreeCrawlOracle braucht einen Partnergraphen
    (config.CROSS_GRAPHS). Ein Lauf ueber die *ganze* Registry soll daran nicht
    scheitern, sondern die betroffenen Verfahren abwaehlen und das sagen. Wer
    sie namentlich anfordert, bekommt weiterhin den lauten Fehler beim Bauen
    des Oracles.

    Rueckgabe: (anwendbare Estimators, Namen der uebersprungenen).
    """
    ok, skipped = [], []
    for e in ests:
        cls = getattr(e, "oracle_cls", None)
        cls = getattr(cls, "func", cls)          # partial aufloesen, s. pipeline
        if cls is not None and not cls.applicable(graph_name):
            skipped.append(e.name)
        else:
            ok.append(e)
    return ok, skipped


def names(category: Category | None = None) -> list[str]:
    return [n for n, e in REGISTRY.items() if category is None or e.category == category]


def build_all(selected: list[str] | None = None, category: Category | None = None) -> list[Estimator]:
    # REGISTRY.get statt [], weil dynamische Margin-Namen dort nicht stehen
    return [build(n) for n in (selected or names(category))
            if category is None or getattr(REGISTRY.get(n), "category", None) == category]
