from abc import ABC, abstractmethod
from typing import List, Set, Dict, Tuple
import copy


# ============================================================================
# БАЗОВІ КЛАСИ З ПОПЕРЕДНІХ ЛАБОРАТОРНИХ РОБІТ
# ============================================================================

class Vidnoshennya(ABC):
    """Базовий абстрактний клас для бінарних відношень"""

    def __init__(self, n: int = 0):
        self.n = n

    @abstractmethod
    def is_reflexive(self):
        pass

    @abstractmethod
    def is_symmetric(self):
        pass

    @abstractmethod
    def is_antisymmetric(self):
        pass

    @abstractmethod
    def is_transitive(self):
        pass


class VidnoshennyaMatr(Vidnoshennya):
    """Клас для роботи з бінарними відношеннями в матричному представленні"""

    def __init__(self, matrix: List[List]):
        super().__init__(len(matrix))
        self.B = copy.deepcopy(matrix)

    def is_reflexive(self):
        """Перевірка рефлексивності"""
        for i in range(self.n):
            if self.B[i][i] == 0:
                return False
        return True

    def is_symmetric(self):
        """Перевірка симетричності"""
        for i in range(self.n):
            for j in range(self.n):
                if self.B[i][j] != self.B[j][i]:
                    return False
        return True

    def is_antisymmetric(self):
        """Перевірка антисиметричності"""
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.B[i][j] != 0 and self.B[j][i] != 0:
                    return False
        return True

    def is_transitive(self):
        """Перевірка транзитивності"""
        for i in range(self.n):
            for j in range(self.n):
                if self.B[i][j]:
                    for k in range(self.n):
                        if self.B[j][k] and not self.B[i][k]:
                            return False
        return True

    # ------------------------------------------------------------------------
    # ФУНКЦІЇ ДЛЯ ПОШУКУ ОПТИМАЛЬНИХ ЕЛЕМЕНТІВ
    # ------------------------------------------------------------------------

    def find_maximums(self) -> Set[int]:
        """
        Знайти множину максимальних елементів
        Елемент є максимальним, якщо немає елемента кращого за нього
        """
        maximums = set()
        for i in range(self.n):
            is_maximum = True
            for j in range(self.n):
                if i != j and self.B[j][i] == 1:
                    is_maximum = False
                    break
            if is_maximum:
                maximums.add(i)
        return maximums

    def find_minimums(self) -> Set[int]:
        """
        Знайти множину мінімальних елементів
        Елемент є мінімальним, якщо він не переважає жодного іншого
        """
        minimums = set()
        for i in range(self.n):
            is_minimum = True
            for j in range(self.n):
                if i != j and self.B[i][j] == 1:
                    is_minimum = False
                    break
            if is_minimum:
                minimums.add(i)
        return minimums

    def find_majorants(self) -> Set[int]:
        """
        Знайти множину мажорант (найбільших елементів)
        Елемент є мажорантою, якщо він переважає всі інші
        """
        majorants = set()
        for i in range(self.n):
            is_majorant = True
            for j in range(self.n):
                if i != j and self.B[i][j] == 0:
                    is_majorant = False
                    break
            if is_majorant:
                majorants.add(i)
        return majorants

    def find_minorants(self) -> Set[int]:
        """
        Знайти множину мінорант (найменших елементів)
        Елемент є мінорантою, якщо всі інші переважають його
        """
        minorants = set()
        for i in range(self.n):
            is_minorant = True
            for j in range(self.n):
                if i != j and self.B[j][i] == 0:
                    is_minorant = False
                    break
            if is_minorant:
                minorants.add(i)
        return minorants

    def print_matrix(self):
        """Вивід матриці відношення"""
        for row in self.B:
            print("  ".join(f"{val:>3}" for val in row))


# ============================================================================
# БАЗОВИЙ КЛАС ДЛЯ МЕХАНІЗМІВ ВИБОРУ
# ============================================================================

class MechanizmVyboru(ABC):
    """Віртуальний базовий клас для механізмів вибору"""

    def __init__(self, alternatives: List[str], criteria: List[str],
                 evaluations: List[List[float]]):
        """
        Ініціалізація механізму вибору

        Параметри:
        alternatives - список назв альтернатив
        criteria - список назв критеріїв
        evaluations - матриця оцінок [альтернатива][критерій]
        """
        self.alternatives = alternatives
        self.criteria = criteria
        self.evaluations = evaluations
        self.n_alternatives = len(alternatives)
        self.n_criteria = len(criteria)

    @abstractmethod
    def find_solution(self) -> List[int]:
        """
        Віртуальна функція пошуку рішення
        Повертає: список індексів обраних альтернатив
        """
        pass

    def print_evaluations(self):
        """Вивід таблиці оцінок"""
        print("\n📊 ТАБЛИЦЯ ОЦІНОК АЛЬТЕРНАТИВ:")
        print("-" * 70)

        header = f"{'Альтернатива':<15}"
        for criterion in self.criteria:
            header += f"{criterion:>12}"
        print(header)
        print("-" * 70)

        for i, alt in enumerate(self.alternatives):
            row = f"{alt:<15}"
            for j in range(self.n_criteria):
                row += f"{self.evaluations[i][j]:>12.0f}"
            print(row)
        print("-" * 70)

    def print_solution(self, solution: List[int], method_name: str):
        """Вивід результату"""
        print(f"\n✅ {method_name}:")
        if solution:
            print(f"   Обрані альтернативи: {[self.alternatives[i] for i in solution]}")
            print(f"   Індекси: {solution}")
        else:
            print("   Рішення не знайдено")


# ============================================================================
# МЕХАНІЗМ ПАРЕТО
# ============================================================================

class MechanizmPareto(MechanizmVyboru):
    """Механізм вибору за принципом Парето"""

    def __init__(self, alternatives: List[str], criteria: List[str],
                 evaluations: List[List[float]], maximize: List[bool] = None):
        super().__init__(alternatives, criteria, evaluations)
        self.maximize = maximize if maximize else [True] * self.n_criteria

    def dominates(self, i: int, j: int) -> bool:
        """
        Перевірка чи альтернатива i домінує j за Парето
        aQb ⇔ ((∀i∈I): Qᵢ(a) ≥ Qᵢ(b)) ∧ ((∃i∈I): Qᵢ(a) > Qᵢ(b))
        """
        at_least_one_better = False

        for k in range(self.n_criteria):
            if self.maximize[k]:
                if self.evaluations[i][k] < self.evaluations[j][k]:
                    return False
                if self.evaluations[i][k] > self.evaluations[j][k]:
                    at_least_one_better = True
            else:
                if self.evaluations[i][k] > self.evaluations[j][k]:
                    return False
                if self.evaluations[i][k] < self.evaluations[j][k]:
                    at_least_one_better = True

        return at_least_one_better

    def find_solution(self) -> List[int]:
        """Знайти множину Парето-оптимальних альтернатив"""
        pareto_set = []

        for i in range(self.n_alternatives):
            is_pareto = True
            for j in range(self.n_alternatives):
                if i != j and self.dominates(j, i):
                    is_pareto = False
                    break

            if is_pareto:
                pareto_set.append(i)

        return pareto_set


# ============================================================================
# МЕХАНІЗМ СЛЕЙТЕРА
# ============================================================================

class MechanizmSlater(MechanizmVyboru):
    """Механізм вибору за принципом Слейтера"""

    def __init__(self, alternatives: List[str], criteria: List[str],
                 evaluations: List[List[float]], maximize: List[bool] = None):
        super().__init__(alternatives, criteria, evaluations)
        self.maximize = maximize if maximize else [True] * self.n_criteria

    def weakly_dominates(self, i: int, j: int) -> bool:
        """
        Перевірка чи альтернатива i слабко домінує j
        aQb ⇔ ((∀i∈I): Qᵢ(a) ≥ Qᵢ(b))
        """
        for k in range(self.n_criteria):
            if self.maximize[k]:
                if self.evaluations[i][k] < self.evaluations[j][k]:
                    return False
            else:
                if self.evaluations[i][k] > self.evaluations[j][k]:
                    return False
        return True

    def find_solution(self) -> List[int]:
        """Знайти множину оптимальних альтернатив за Слейтером"""
        slater_set = []

        for i in range(self.n_alternatives):
            is_slater = True
            for j in range(self.n_alternatives):
                if i != j and self.weakly_dominates(j, i) and not self.weakly_dominates(i, j):
                    is_slater = False
                    break

            if is_slater:
                slater_set.append(i)

        return slater_set


# ============================================================================
# МЕХАНІЗМ НАЙКРАЩОГО РЕЗУЛЬТАТУ
# ============================================================================

class MechanizmNaikraschogoResultatu(MechanizmVyboru):
    """Механізм вибору найкращого результату (оптимістичний)"""

    def __init__(self, alternatives: List[str], criteria: List[str],
                 evaluations: List[List[float]], maximize: List[bool] = None):
        super().__init__(alternatives, criteria, evaluations)
        self.maximize = maximize if maximize else [True] * self.n_criteria

    def find_solution(self) -> List[int]:
        """
        aQb ⇔ max Qᵢ(a) ≥ max Qᵢ(b)
        """
        best_values = []

        for i in range(self.n_alternatives):
            best_val = max(self.evaluations[i])
            best_values.append(best_val)

        max_value = max(best_values)
        return [i for i, val in enumerate(best_values) if val == max_value]


# ============================================================================
# МЕХАНІЗМ ГАРАНТОВАНОГО РЕЗУЛЬТАТУ
# ============================================================================

class MechanizmGarantovanogoResultatu(MechanizmVyboru):
    """Механізм вибору гарантованого результату (песимістичний)"""

    def __init__(self, alternatives: List[str], criteria: List[str],
                 evaluations: List[List[float]], maximize: List[bool] = None):
        super().__init__(alternatives, criteria, evaluations)
        self.maximize = maximize if maximize else [True] * self.n_criteria

    def find_solution(self) -> List[int]:
        """
        aQb ⇔ min Qᵢ(a) ≥ min Qᵢ(b)
        """
        worst_values = []

        for i in range(self.n_alternatives):
            worst_val = min(self.evaluations[i])
            worst_values.append(worst_val)

        max_worst = max(worst_values)
        return [i for i, val in enumerate(worst_values) if val == max_worst]


# ============================================================================
# МЕХАНІЗМ ГУРВІЦА
# ============================================================================

class MechanizmHurvica(MechanizmVyboru):
    """Механізм вибору Гурвіца"""

    def __init__(self, alternatives: List[str], criteria: List[str],
                 evaluations: List[List[float]], alpha: float = 0.5,
                 maximize: List[bool] = None):
        super().__init__(alternatives, criteria, evaluations)
        self.alpha = alpha
        self.maximize = maximize if maximize else [True] * self.n_criteria

    def find_solution(self) -> List[int]:
        """
        aQb ⇔ (α min Qᵢ(a) + (1-α) max Qᵢ(a)) >
              > (α min Qᵢ(b) + (1-α) max Qᵢ(b))
        """
        hurvic_values = []

        for i in range(self.n_alternatives):
            max_val = max(self.evaluations[i])
            min_val = min(self.evaluations[i])

            hurvic_val = self.alpha * min_val + (1 - self.alpha) * max_val
            hurvic_values.append(hurvic_val)

        max_hurvic = max(hurvic_values)
        return [i for i, val in enumerate(hurvic_values) if abs(val - max_hurvic) < 1e-6]


# ============================================================================
# МЕХАНІЗМ ВИБОРУ ЗА ЕТАЛОНОМ
# ============================================================================

class MechanizmZaEtalonom(MechanizmVyboru):
    """Механізм вибору за еталоном"""

    def __init__(self, alternatives: List[str], criteria: List[str],
                 evaluations: List[List[float]], etalon: List[float],
                 weights: List[float] = None):
        super().__init__(alternatives, criteria, evaluations)
        self.etalon = etalon
        self.weights = weights if weights else [1.0] * self.n_criteria

    def find_solution(self) -> List[int]:
        """
        aQb ⇔ d(Q(a) - Qᴱ) ≤ |Q(b) - Qᴱ|
        """
        distances = []

        for i in range(self.n_alternatives):
            distance = 0
            for j in range(self.n_criteria):
                diff = self.etalon[j] - self.evaluations[i][j]
                distance += self.weights[j] * diff * diff
            distances.append(distance ** 0.5)

        min_distance = min(distances)
        return [i for i, d in enumerate(distances) if abs(d - min_distance) < 1e-6]


# ============================================================================
# МЕХАНІЗМ ЗГОРТКИ КРИТЕРІЇВ
# ============================================================================

class MechanizmZgortkiKriteriiv(MechanizmVyboru):
    """Механізм вибору через згортку критеріїв"""

    def __init__(self, alternatives: List[str], criteria: List[str],
                 evaluations: List[List[float]], weights: List[float] = None,
                 maximize: List[bool] = None):
        super().__init__(alternatives, criteria, evaluations)
        self.weights = weights if weights else [1.0 / self.n_criteria] * self.n_criteria
        self.maximize = maximize if maximize else [True] * self.n_criteria

    def find_solution(self) -> List[int]:
        """
        aQb ⇔ f(Q(a)) ≥ f(Q(b))
        де f - функція згортки (зважена сума)
        """
        scores = []

        for i in range(self.n_alternatives):
            score = 0
            for j in range(self.n_criteria):
                score += self.weights[j] * self.evaluations[i][j]
            scores.append(score)

        max_score = max(scores)
        return [i for i, s in enumerate(scores) if abs(s - max_score) < 1e-6]


# ============================================================================
# ЛЕКСИКОГРАФІЧНИЙ МЕХАНІЗМ
# ============================================================================

class MechanizmLeksikografichnyi(MechanizmVyboru):
    """Лексикографічний механізм вибору"""

    def __init__(self, alternatives: List[str], criteria: List[str],
                 evaluations: List[List[float]], priority: List[int] = None,
                 maximize: List[bool] = None):
        super().__init__(alternatives, criteria, evaluations)
        self.priority = priority if priority else list(range(self.n_criteria))
        self.maximize = maximize if maximize else [True] * self.n_criteria

    def find_solution(self) -> List[int]:
        """
        aQb ⇔ ((Qⱼ(a) = Qⱼ(b)) ∧ (Qⱼ₊₁(a)>Qⱼ₊₁(b)))
        де j - критерій за порядком пріоритету
        """
        candidates = list(range(self.n_alternatives))

        for criterion_idx in self.priority:
            if len(candidates) == 1:
                break

            values = [self.evaluations[i][criterion_idx] for i in candidates]

            if self.maximize[criterion_idx]:
                best_value = max(values)
            else:
                best_value = min(values)

            candidates = [i for i in candidates
                          if abs(self.evaluations[i][criterion_idx] - best_value) < 1e-6]

        return candidates


# ============================================================================
# МЕХАНІЗМ ГОЛОВНОГО КРИТЕРІЮ
# ============================================================================

class MechanizmGolovnogoKriteriyu(MechanizmVyboru):
    """Механізм вибору за головним критерієм з обмеженнями"""

    def __init__(self, alternatives: List[str], criteria: List[str],
                 evaluations: List[List[float]], main_criterion: int,
                 constraints: List[float] = None, maximize_main: bool = True):
        super().__init__(alternatives, criteria, evaluations)
        self.main_criterion = main_criterion
        self.constraints = constraints if constraints else [0] * self.n_criteria
        self.maximize_main = maximize_main

    def find_solution(self) -> List[int]:
        feasible = []

        for i in range(self.n_alternatives):
            is_feasible = True
            # Перевіряємо ВСІ обмеження, включно з головним критерієм
            for j in range(self.n_criteria):
                if self.evaluations[i][j] < self.constraints[j]:
                    is_feasible = False
                    break

            if is_feasible:
                feasible.append(i)

        if not feasible:
            return []

        # Серед допустимих обираємо найкращі за головним критерієм
        main_values = [self.evaluations[i][self.main_criterion] for i in feasible]

        if self.maximize_main:
            best_value = max(main_values)
        else:
            best_value = min(main_values)

        return [i for i in feasible
                if abs(self.evaluations[i][self.main_criterion] - best_value) < 1e-6]


# ============================================================================
# МЕХАНІЗМ ПОСЛІДОВНОЇ ПОСТУПКИ
# ============================================================================

class MechanizmPoslidovnoiPostupky(MechanizmVyboru):
    """Механізм вибору послідовної поступки"""

    def __init__(self, alternatives: List[str], criteria: List[str],
                 evaluations: List[List[float]],
                 delta: List[float] = None):
        super().__init__(alternatives, criteria, evaluations)
        self.delta = delta if delta else [0.1] * self.n_criteria

    def find_solution(self) -> List[int]:
        """
        Метод послідовної поступки
        Послідовно розглядаємо критерії в порядку Q₁, Q₂, Q₃...
        На кожному кроці залишаємо тільки альтернативи, які не гірші за (max - delta)
        """
        candidates = list(range(self.n_alternatives))

        for criterion_idx in range(self.n_criteria):
            if len(candidates) <= 1:
                break

            # Знаходимо максимальне значення за поточним критерієм серед кандидатів
            values = [self.evaluations[i][criterion_idx] for i in candidates]
            max_value = max(values)

            # Обчислюємо поріг: максимум мінус допустима поступка
            threshold = max_value - self.delta[criterion_idx]

            # Залишаємо тільки альтернативи не гірші за поріг
            new_candidates = [i for i in candidates
                              if self.evaluations[i][criterion_idx] >= threshold]

            if new_candidates:
                candidates = new_candidates

        return candidates



def main():
    print("=" * 90)
    print(" " * 25 + "ЛАБОРАТОРНА РОБОТА №4")
    print(" " * 15 + "МЕХАНІЗМИ ВИБОРУ ПОРОДЖЕНІ БІНАРНИМИ ВІДНОШЕННЯМИ")
    print(" " * 35 + "ВАРІАНТ 4")
    print("=" * 90)

    alternatives = ["a₁", "a₂", "a₃", "a₄", "E"]
    criteria = ["Q₁", "Q₂", "Q₃"]

    evaluations = [
        [2, 4, 6],  # a₁
        [3, 2, 5],  # a₂
        [1, 3, 6],  # a₃
        [2, 5, 4],  # a₄
        [2, 3, 5],  # E (еталон)
    ]

    maximize = [True, True, True]

    print("\n📋 ВИХІДНІ ДАНІ (ВАРІАНТ 4):")
    print(f"Альтернативи: {alternatives}")
    print(f"Критерії: {criteria}")
    print(f"Напрямок оптимізації: всі критерії МАКСИМІЗУЮТЬСЯ")

    base_mechanism = MechanizmPareto(alternatives, criteria, evaluations, maximize)
    base_mechanism.print_evaluations()

    print("\n" + "=" * 90)
    print("ЗАСТОСУВАННЯ МЕХАНІЗМІВ ВИБОРУ")
    print("=" * 90)

    all_results = {}

    print("\n" + "-" * 90)
    print("1️⃣  МЕХАНІЗМ ПАРЕТО")
    print("-" * 90)
    print("Бінарне відношення: aQb ⇔ ((∀i∈I): Qᵢ(a)≥Qᵢ(b)) ∧ ((∃i∈I): Qᵢ(a)>Qᵢ(b))")

    pareto = MechanizmPareto(alternatives, criteria, evaluations, maximize)
    pareto_solution = pareto.find_solution()
    pareto.print_solution(pareto_solution, "Множина Парето")
    all_results["Парето"] = pareto_solution

    print("\n" + "-" * 90)
    print("2️⃣  МЕХАНІЗМ СЛЕЙТЕРА")
    print("-" * 90)
    print("Бінарне відношення: aQb ⇔ ((∀i∈I): Qᵢ(a)≥Qᵢ(b))")

    slater = MechanizmSlater(alternatives, criteria, evaluations, maximize)
    slater_solution = slater.find_solution()
    slater.print_solution(slater_solution, "Множина Слейтера")
    all_results["Слейтер"] = slater_solution

    print("\n" + "-" * 90)
    print("3️⃣  МЕХАНІЗМ НАЙКРАЩОГО РЕЗУЛЬТАТУ")
    print("-" * 90)
    print("Бінарне відношення: aQb ⇔ max Qᵢ(a) ≥ max Qᵢ(b)")

    best = MechanizmNaikraschogoResultatu(alternatives, criteria, evaluations, maximize)
    best_solution = best.find_solution()
    best.print_solution(best_solution, "Найкращий результат")
    all_results["Найкращий результат"] = best_solution

    # Виведення детальних значень
    print("\n   Детально:")
    for i in range(len(alternatives)):
        max_val = max(evaluations[i])
        print(f"   {alternatives[i]}: max = {max_val}")

    print("\n" + "-" * 90)
    print("4️⃣  МЕХАНІЗМ ГАРАНТОВАНОГО РЕЗУЛЬТАТУ")
    print("-" * 90)
    print("Бінарне відношення: aQb ⇔ min Qᵢ(a) ≥ min Qᵢ(b)")

    guaranteed = MechanizmGarantovanogoResultatu(alternatives, criteria, evaluations, maximize)
    guaranteed_solution = guaranteed.find_solution()
    guaranteed.print_solution(guaranteed_solution, "Гарантований результат")
    all_results["Гарантований результат"] = guaranteed_solution

    print("\n   Детально:")
    for i in range(len(alternatives)):
        min_val = min(evaluations[i])
        print(f"   {alternatives[i]}: min = {min_val}")

    print("\n" + "-" * 90)
    print("5️⃣  МЕХАНІЗМ ГУРВІЦА (α=0.5)")
    print("-" * 90)
    print("Бінарне відношення: aQb ⇔ (α min Qᵢ(a)+(1-α) max Qᵢ(a)) >")
    print("                          > (α min Qᵢ(b)+(1-α) max Qᵢ(b))")

    hurvic = MechanizmHurvica(alternatives, criteria, evaluations, alpha=0.5, maximize=maximize)
    hurvic_solution = hurvic.find_solution()
    hurvic.print_solution(hurvic_solution, "Критерій Гурвіца")
    all_results["Гурвіца"] = hurvic_solution

    print("\n   Детально:")
    for i in range(len(alternatives)):
        min_val = min(evaluations[i])
        max_val = max(evaluations[i])
        hurvic_val = 0.5 * min_val + 0.5 * max_val
        print(f"   {alternatives[i]}: H = 0.5×{min_val} + 0.5×{max_val} = {hurvic_val}")

    print("\n" + "-" * 90)
    print("6️⃣  МЕХАНІЗМ ВИБОРУ ЗА ЕТАЛОНОМ")
    print("-" * 90)
    print("Бінарне відношення: aQb ⇔ d(Q(a)-Qᴱ) ≤ |Q(b)-Qᴱ|")

    # Еталон не повинен бути серед альтернатив для порівняння
    alternatives_without_etalon = ["a₁", "a₂", "a₃", "a₄"]
    evaluations_without_etalon = [
        [2, 4, 6],  # a₁
        [3, 2, 5],  # a₂
        [1, 3, 6],  # a₃
        [2, 5, 4],  # a₄
    ]

    etalon = [3, 4, 5]  # Еталонні значення
    weights_etalon = [1 / 3, 1 / 3, 1 / 3]  # Рівні ваги

    print(f"   Еталон Qᴱ: {etalon}")
    print(f"   Ваги: {weights_etalon}")
    print(f"   Альтернативи для порівняння: {alternatives_without_etalon}")

    etalon_mech = MechanizmZaEtalonom(alternatives_without_etalon, criteria,
                                       evaluations_without_etalon, etalon, weights_etalon)
    etalon_solution = etalon_mech.find_solution()
    etalon_mech.print_solution(etalon_solution, "Вибір за еталоном")
    all_results["За еталоном"] = etalon_solution

    print("\n   Детально (відстані до еталону):")
    for i in range(len(alternatives_without_etalon)):
        distance = 0
        for j in range(len(criteria)):
            diff = etalon[j] - evaluations_without_etalon[i][j]
            distance += weights_etalon[j] * diff * diff
        distance = distance ** 0.5
        print(f"   {alternatives_without_etalon[i]}: d = {distance:.3f}")

    print("\n" + "-" * 90)
    print("7️⃣  МЕХАНІЗМ ЗГОРТКИ КРИТЕРІЇВ")
    print("-" * 90)
    print("Бінарне відношення: aQb ⇔ f(Q(a)) ≥ f(Q(b))")
    print("де f - зважена сума критеріїв")

    weights_zghortka = [1 / 3, 1 / 3, 1 / 3]  # Рівні ваги
    print(f"   Ваги критеріїв: {weights_zghortka}")

    zghortka = MechanizmZgortkiKriteriiv(alternatives, criteria, evaluations, weights_zghortka, maximize)
    zghortka_solution = zghortka.find_solution()
    zghortka.print_solution(zghortka_solution, "Згортка критеріїв")
    all_results["Згортка"] = zghortka_solution

    print("\n   Детально (зважені суми):")
    for i in range(len(alternatives)):
        score = sum(weights_zghortka[j] * evaluations[i][j] for j in range(len(criteria)))
        print(f"   {alternatives[i]}: F = {score:.3f}")

    print("\n" + "-" * 90)
    print("8️⃣  ЛЕКСИКОГРАФІЧНИЙ МЕХАНІЗМ")
    print("-" * 90)
    print("Бінарне відношення: aQb ⇔ ((Qⱼ(a)=Qⱼ(b)) ∧ (Qⱼ₊₁(a)>Qⱼ₊₁(b)))")

    priority = [2, 0, 1]  # Пріоритет: Q₃ → Q₁ → Q₂
    print(f"   Порядок пріоритету: {[criteria[i] for i in priority]}")

    leksiko = MechanizmLeksikografichnyi(alternatives, criteria, evaluations, priority, maximize)
    leksiko_solution = leksiko.find_solution()
    leksiko.print_solution(leksiko_solution, "Лексикографічний вибір")
    all_results["Лексикографічний"] = leksiko_solution

    print("\n   Детально (покроковий відбір):")
    candidates = list(range(len(alternatives)))
    for step, criterion_idx in enumerate(priority):
        if len(candidates) <= 1:
            break
        values = [evaluations[i][criterion_idx] for i in candidates]
        best_value = max(values)
        new_candidates = [i for i in candidates if evaluations[i][criterion_idx] == best_value]
        print(f"   Крок {step + 1} ({criteria[criterion_idx]}): найкраще значення = {best_value}")
        print(f"      Залишилось: {[alternatives[i] for i in new_candidates]}")
        candidates = new_candidates

    print("\n" + "-" * 90)
    print("9️⃣  МЕХАНІЗМ ГОЛОВНОГО КРИТЕРІЮ")
    print("-" * 90)
    print("Бінарне відношення: aQb ⇔ (Qₗ(a) ≥ Qₗ(b))")
    print("за умови (i∈I\\{l}): (Qᵢ(a) ≥ Qᵢᴹ ∧ Qᵢ(b) ≥ Qᵢᴹ)")

    main_criterion = 2  # Q₃ як головний критерій
    constraints = [0, 4, 5]  # Обмеження: Q₁≥0, Q₂≥4, Q₃≥5

    print(f"   Головний критерій: {criteria[main_criterion]}")
    print(f"   Обмеження: Q₁≥{constraints[0]}, Q₂≥{constraints[1]}, Q₃≥{constraints[2]}")

    golovnyi = MechanizmGolovnogoKriteriyu(alternatives, criteria, evaluations,
                                           main_criterion, constraints, maximize_main=True)
    golovnyi_solution = golovnyi.find_solution()
    golovnyi.print_solution(golovnyi_solution, "Головний критерій")
    all_results["Головний критерій"] = golovnyi_solution

    print("\n   Детально (перевірка обмежень):")
    for i in range(len(alternatives)):
        feasible = all(evaluations[i][j] >= constraints[j] for j in range(len(criteria)) if j != main_criterion)
        status = "✓ задовольняє" if feasible else "✗ не задовольняє"
        print(f"   {alternatives[i]}: {status} обмеження, {criteria[main_criterion]}={evaluations[i][main_criterion]}")

    print("\n" + "-" * 90)
    print("🔟 МЕХАНІЗМ ПОСЛІДОВНОЇ ПОСТУПКИ")
    print("-" * 90)

    delta = [0.5, 0.5, 0.5]
    print(f"   Допустимі поступки δ: {delta}")
    print(f"   Порядок критеріїв: Q₁ → Q₂ → Q₃")

    postupka = MechanizmPoslidovnoiPostupky(alternatives, criteria, evaluations, delta)
    postupka_solution = postupka.find_solution()
    postupka.print_solution(postupka_solution, "Послідовна поступка")
    all_results["Послідовна поступка"] = postupka_solution

    print("\n   Детально (покроковий відбір):")
    candidates = list(range(len(alternatives)))
    for criterion_idx in range(len(criteria)):
        if len(candidates) <= 1:
            break
        values = [evaluations[i][criterion_idx] for i in candidates]
        max_value = max(values)
        threshold = max_value - delta[criterion_idx]
        new_candidates = [i for i in candidates if evaluations[i][criterion_idx] >= threshold]
        print(f"   Крок {criterion_idx + 1} ({criteria[criterion_idx]}): max={max_value}, поріг={threshold}")
        print(f"      Залишилось: {[alternatives[i] for i in new_candidates]}")
        if new_candidates:
            candidates = new_candidates

    # ========================================================================
    # ПОРІВНЯЛЬНА ТАБЛИЦЯ РЕЗУЛЬТАТІВ
    # ========================================================================

    print("\n" + "=" * 90)
    print("📊 ПОРІВНЯЛЬНА ТАБЛИЦЯ РЕЗУЛЬТАТІВ")
    print("=" * 90)

    print(f"\n{'№':<3} {'Механізм':<25} {'Обрані альтернативи':<30} {'Кількість':<10}")
    print("-" * 90)

    for idx, (mechanism, solution) in enumerate(all_results.items(), 1):
        alts = [alternatives[i] for i in solution] if solution else ["Немає"]
        print(f"{idx:<3} {mechanism:<25} {str(alts):<30} {len(solution):<10}")

    print("-" * 90)

    # ========================================================================
    # АНАЛІЗ РЕЗУЛЬТАТІВ
    # ========================================================================

    print("\n" + "=" * 90)
    print("📈 АНАЛІЗ РЕЗУЛЬТАТІВ")
    print("=" * 90)

    print("   Альтернатива  Q₁  Q₂  Q₃  | max  min")
    print("   " + "-" * 45)
    for i in range(len(alternatives)):
        max_val = max(evaluations[i])
        min_val = min(evaluations[i])
        print(
            f"   {alternatives[i]:<12} {evaluations[i][0]:>3} {evaluations[i][1]:>3} {evaluations[i][2]:>3}  | {max_val:>3}  {min_val:>3}")


if __name__ == "__main__":
    main()

