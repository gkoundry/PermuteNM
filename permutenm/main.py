""" Implementation of Geometric Nelder-Mead Algorithm for Permutations """

import logging
from collections import Counter
from dataclasses import dataclass
from random import choice, random
from typing import Any, Callable, List, Sequence, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class FinalResult:
    best_score: float
    best_member: Sequence[Any]


class GNMPermutationOptimizer:
    """Geometric Nelder-Mead algorithm for optimizing permutations,
    Parameters
    ==========
    objective: callable
        the objective function to be minimized.  Should accept a permutation
        and return the score for that permutation.
    alpha: float (0,inf)
        reflection length of ray from worst point through center of mass
    gamma: float (0,inf)
        length of extension ray through new best point
    rho: float (0,1)
        amount to contract simplex if new point is not better than best point
    sigma: float (0,1)
        amount to contract simplex if new point is worse than worst point

    References
    ==========
    A. Moraglio and J. Togelius,
    "Geometric Nelder-Mead Algorithm for the permutation representation,"
    IEEE Congress on Evolutionary Computation, Barcelona, Spain, 2010,
    pp. 1-8, doi: 10.1109/CEC.2010.5586321.
    """

    def __init__(
        self,
        objective: Callable[[Sequence[Any]], float],
        alpha: float = 1,
        gamma: float = 2,
        rho: float = 0.5,
        sigma: float = 0.5,
        verbose: bool = False,
    ):
        self.objective = objective
        self.num_elements = 0
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
        self.sigma = sigma
        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

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
        p = sdbc / max(1, self.num_elements - 1 - sdab)
        pc = list(pb)
        for i in range(self.num_elements):
            if pc[i] == pa[i] and random() < p:
                j = choice([x for x in range(self.num_elements) if x != i])
                tmp = pc[j]
                pc[j] = pc[i]
                pc[i] = tmp
        return pc

    def _convex_combination(
        self,
        pa: Sequence[Any],
        pb: Sequence[Any],
        wa: float,
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

    def _shrink_population(self, pop: Sequence[Sequence[Any]]) -> None:
        logger.debug("Shrink")
        for i in range(self.num_elements, 0, -1):
            c0 = self._convex_combination(pop[0][1], pop[i][1], 1 - self.sigma)
            scc0 = self.objective(c0)
            pop[i] = (scc0, c0)

    def _contract_population(
        self,
        pop: Sequence[Sequence[T]],
        ray: Sequence[T],
        center_of_mass: Sequence[T],
        ray_score: float,
    ) -> bool:
        logger.debug("Contract")
        do_shrink = True
        if ray_score >= pop[-2][0]:
            c = self._convex_combination(ray, center_of_mass, self.rho)
            scc = self.objective(c)
            if scc < ray_score:
                pop[-1] = (scc, c)
                do_shrink = False
        return do_shrink

    def minimize(
        self, initial_population: Sequence[Sequence[Any]], max_iter: int = 1000
    ) -> FinalResult:
        """Find minimum value of object function staring from an initial
        population of permutations.

        Patameters
        ==========
        initial_population: Sequence[Sequence[Any]]
            array-like containing N+1 permutations, where N size the size of each permutation
        max_iter: int
            maximum number of iterations

        Returns
        =======
        final_result: FinalResult
            object with attributes `best_score` containing the lowest score found an
            `best_member` containing the permutation with the lowest score
        """

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

        for it in range(max_iter):
            pop = sorted(pop)
            if pop[0][0] == pop[-1][0]:
                return FinalResult(best_score=pop[0][0], best_member=pop[0][1])
            logger.debug("Iteration %d. Current population:", it)
            for p in pop:
                logger.debug("  %f %s", p[0], p[1])
            center_of_mass = self._center_of_mass(pop[:-1])
            ray = self._extension_ray(
                pop[-1][1],
                center_of_mass,
                self.alpha / (1 + self.alpha),
                1 / (1 + self.alpha),
            )
            ray_score = self.objective(ray)
            logger.debug("Ray: %f %s", ray_score, ray)
            if pop[0][0] < ray_score < pop[-1][0]:
                pop[-1] = (ray_score, ray)
            else:
                if ray_score <= pop[0][0]:
                    extension_ray = self._extension_ray(
                        center_of_mass,
                        ray,
                        1 / self.gamma,
                        (self.gamma - 1) / self.gamma,
                    )
                    extension_score = self.objective(extension_ray)
                    logger.debug("Extension ray %f %s", extension_score, extension_ray)
                    if extension_score < ray_score:
                        pop[-1] = (extension_score, extension_ray)
                    else:
                        pop[-1] = (ray_score, ray)
                else:
                    do_shrink = self._contract_population(
                        pop, ray, center_of_mass, ray_score
                    )
                    if do_shrink:
                        self._shrink_population(pop)
