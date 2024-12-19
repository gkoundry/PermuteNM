""" Implementation of Geometric Nelder-Mead Algorithm for Permutations """

from collections import Counter
from dataclasses import dataclass
from random import choice, random
from typing import Any, Callable, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


@dataclass
class FinalResult:
    best_score: float
    best_member: Sequence[Any]


class GNMPermutationOptimizer:
    def __init__(
        self,
        objective: Callable[[Sequence[Any]], float],
        alpha: float = 1,
        phi: float = 0.5,
        gamma: float = 2,
        sigma: float = 0.5,
        verbose: bool = False
    ):
        self.objective = objective
        self.num_elements = 0
        self.alpha = alpha
        self.phi = phi
        self.gamma = gamma
        self.sigma = sigma
        self.verbose = verbose

    def _center_of_mass(
        self, population_member: Sequence[Tuple[float, Sequence[T]]]
    ) -> List[T]:
        cm_pop = [list(p[1]) for p in population_member]

        most_frequent = []
        for i in range(self.num_elements):
            el, cnt = Counter([p[i] for p in cm_pop]).most_common(1)[0]
            most_frequent.append((cnt, el, i))

        consp = [False] * self.num_elements
        while not all(consp):
            cnt, el_max, pos_max = max(most_frequent)
            for p in cm_pop:
                ix = p.index(el_max)
                tmp = p[ix]
                p[ix] = p[pos_max]
                p[pos_max] = tmp
            most_frequent = []
            consp[pos_max] = True
            for i in range(self.num_elements):
                if consp[i]:
                    continue
                el, cnt = Counter([p[i] for p in cm_pop]).most_common(1)[0]
                most_frequent.append((cnt, el, i))
        return cm_pop[0]

    def _swap_distance(self, pa: Sequence[Any], pb: Sequence[Any]) -> int:
        pb = list(pb)
        dist = 0
        for i in range(self.num_elements):
            if pa[i] != pb[i]:
                j = pb.index(pa[i])
                tmp = pb[i]
                pb[i] = pb[j]
                pb[j] = tmp
                dist += 1
        return dist

    def _extension_ray(
        self, pa: Sequence[Any], pb: Sequence[Any], wab: float, wbc: float
    ) -> Sequence[Any]:
        sdab = self._swap_distance(pa, pb)
        sdbc = sdab * wab / wbc
        p = sdbc / (self.num_elements - sdab)
        pc = list(pb)
        for i in range(self.num_elements):
            if pc[i] == pa[i] and random() < p:
                j = choice([x for x in range(self.num_elements) if x != i])
                tmp = pc[j]
                pc[j] = pc[i]
                pc[i] = tmp
        return pc

    def _convex_combination(
        self, pa: Sequence[Any], pb: Sequence[Any], wa: float, wb: float
    ) -> Sequence[Any]:
        pa = list(pa)
        pb = list(pb)
        mask = ["a" if random() < wa else "b" for _ in range(self.num_elements)]
        for i in range(self.num_elements):
            if pa[i] != pb[i]:
                if mask[i] == "a":
                    j = pb.index(pa[i])
                    tmp = pb[i]
                    pb[i] = pb[j]
                    pb[j] = tmp
                else:
                    j = pa.index(pb[i])
                    tmp = pa[i]
                    pa[i] = pa[j]
                    pa[j] = tmp
        return pa

    def minimize(self, initial_population: Sequence[Sequence[Any]]) -> FinalResult:
        assert (
            len(set(len(p) for p in initial_population)) == 1
        ), "Population members must all have the same size."
        self.num_elements = len(initial_population[0])
        assert (
            len(initial_population) == self.num_elements + 1
        ), "Number of population members should be equal to member size plus one."
        pop = []
        for p in initial_population:
            sc = self.objective(p)
            pop.append((sc, p))

        while True:
            pop = sorted(pop)
            if pop[0][0] == pop[-1][0]:
                return FinalResult(best_score=pop[0][0], best_member=pop[0][1])
            if self.verbose:
                for p in pop:
                    print(f"POP {p[0]:.1f} {' '.join(p[1])}")
            m = self._center_of_mass(pop[:-1])
            r = self._extension_ray(
                pop[-1][1], m, self.alpha / (1 + self.alpha), 1 / (1 + self.alpha)
            )
            scr = self.objective(r)
            if self.verbose:
                print(f"RAY {scr:.1f} {' '.join(r)}")
            if pop[0][0] < scr < pop[-1][0]:
                pop[-1] = (scr, r)
            else:
                if scr <= pop[0][0]:
                    e = self._extension_ray(
                        m, r, 1 / self.gamma, (self.gamma - 1) / self.gamma
                    )
                    sce = self.objective(e)
                    if self.verbose:
                        print(f"RAY2 {sce:.1f} {' '.join(e)}")
                    if sce < scr:
                        pop[-1] = (sce, e)
                    else:
                        pop[-1] = (scr, r)
                else:
                    if self.verbose:
                        print("CX")
                    b = True
                    if scr >= pop[-2][0]:
                        c = self._convex_combination(r, m, self.phi, 1 - self.phi)
                        scc = self.objective(c)
                        if scc < scr:
                            pop[-1] = (scc, c)
                            b = False
                    if b:
                        for i in range(self.num_elements, 0, -1):
                            c0 = self._convex_combination(
                                pop[0][1], pop[i][1], 1 - self.sigma, self.sigma
                            )
                            scc0 = self.objective(c0)
                            pop[i] = (scc0, c0)
