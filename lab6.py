from abc import ABC, abstractmethod
from typing import List, Set, Dict, Tuple
import copy
from itertools import permutations


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
        for i in range(self.n):
            if self.B[i][i] == 0:
                return False
        return True

    def is_symmetric(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.B[i][j] != self.B[j][i]:
                    return False
        return True

    def is_antisymmetric(self):
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.B[i][j] != 0 and self.B[j][i] != 0:
                    return False
        return True

    def is_transitive(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.B[i][j]:
                    for k in range(self.n):
                        if self.B[j][k] and not self.B[i][k]:
                            return False
        return True

    def print_matrix(self):
        """Вивід матриці відношення"""
        for row in self.B:
            print("  ".join(f"{val}" for val in row))


# ============================================================================
# БАЗОВИЙ КЛАС ДЛЯ ГРУПОВОГО ВІДНОШЕННЯ
# ============================================================================

class GrupoveVidnoshennya(ABC):
    """Віртуальний базовий клас для групових відношень"""

    def __init__(self, alternatives: List[str], experts: List[str],
                 expert_preferences: List[VidnoshennyaMatr]):
        """
        Ініціалізація групового відношення

        Параметри:
        alternatives - множина альтернатив
        experts - множина експертів
        expert_preferences - множина експертних оцінок (відношень порядку)
        """
        self.alternatives = alternatives
        self.experts = experts
        self.expert_preferences = expert_preferences
        self.n_alternatives = len(alternatives)
        self.n_experts = len(experts)

    @abstractmethod
    def find_solution(self) -> VidnoshennyaMatr:
        """
        Віртуальна функція пошуку групового рішення
        Повертає: групове відношення переваги
        """
        pass

    def print_expert_preferences(self):
        """Вивід відношень переваг експертів"""
        print("\n📊 ВІДНОШЕННЯ ПЕРЕВАГ ЕКСПЕРТІВ:")
        print("=" * 70)

        for i, expert in enumerate(self.experts):
            print(f"\n{expert}:")
            self.expert_preferences[i].print_matrix()

    def print_solution(self, solution: VidnoshennyaMatr, method_name: str):
        """Вивід групового рішення"""
        print(f"\n✅ {method_name}:")
        print("-" * 70)
        solution.print_matrix()

        # Виведення лінійного порядку якщо можливо
        order = self.extract_linear_order(solution)
        if order:
            order_names = [self.alternatives[i] for i in order]
            print(f"\nЛінійний порядок: {' ≻ '.join(order_names)}")

    def extract_linear_order(self, relation: VidnoshennyaMatr) -> List[int]:
        """
        Витягти лінійний порядок з відношення (якщо можливо)
        Повертає список індексів від найкращого до найгіршого
        """
        # Підрахунок переваг для кожної альтернативи
        scores = []
        for i in range(self.n_alternatives):
            score = sum(relation.B[i][j] for j in range(self.n_alternatives))
            scores.append((score, i))

        # Сортування за спаданням
        scores.sort(reverse=True)

        return [idx for score, idx in scores]


# ============================================================================
# МЕХАНІЗМ БІЛЬШОСТІ ГОЛОСІВ
# ============================================================================

class MechanizmBilshostiGolosiv(GrupoveVidnoshennya):
    """Механізм узгодження за принципом більшості голосів"""

    def find_solution(self) -> VidnoshennyaMatr:
        """
        Групове відношення: aRb ⇔ більшість експертів віддає перевагу a над b

        R[i][j] = 1, якщо більше половини експертів вважають i ≻ j
        """
        result_matrix = [[0] * self.n_alternatives for _ in range(self.n_alternatives)]

        for i in range(self.n_alternatives):
            for j in range(self.n_alternatives):
                if i != j:
                    # Підрахунок голосів за перевагу i над j
                    votes = sum(1 for pref in self.expert_preferences
                                if pref.B[i][j] == 1)

                    # Більшість голосів
                    if votes > self.n_experts / 2:
                        result_matrix[i][j] = 1

        return VidnoshennyaMatr(result_matrix)


# ============================================================================
# МЕХАНІЗМ КОНДОРСЕ
# ============================================================================

class MechanizmKondorse(GrupoveVidnoshennya):
    """Механізм узгодження за принципом Кондорсе"""

    def find_solution(self) -> Tuple[VidnoshennyaMatr, List[int], int]:
        """
        Переможець Кондорсе: альтернатива, яка перемагає всі інші
        в парних порівняннях за більшістю голосів

        Повертає: (матриця переваг, список переможців, -1 якщо немає переможця)
        """
        # Спочатку будуємо матрицю більшості
        majority_matrix = [[0] * self.n_alternatives for _ in range(self.n_alternatives)]
        vote_counts = [[0] * self.n_alternatives for _ in range(self.n_alternatives)]

        for i in range(self.n_alternatives):
            for j in range(self.n_alternatives):
                if i != j:
                    # Підрахунок голосів за i проти j
                    votes_i = sum(1 for pref in self.expert_preferences
                                  if pref.B[i][j] == 1)
                    votes_j = sum(1 for pref in self.expert_preferences
                                  if pref.B[j][i] == 1)

                    vote_counts[i][j] = votes_i

                    # i перемагає j, якщо більше голосів
                    if votes_i > votes_j:
                        majority_matrix[i][j] = 1

        # Шукаємо переможця Кондорсе
        condorcet_winners = []
        for i in range(self.n_alternatives):
            is_winner = True
            for j in range(self.n_alternatives):
                if i != j and majority_matrix[i][j] != 1:
                    is_winner = False
                    break
            if is_winner:
                condorcet_winners.append(i)

        return VidnoshennyaMatr(majority_matrix), condorcet_winners, vote_counts

    def print_vote_matrix(self, vote_counts):
        """Вивід матриці підрахунку голосів"""
        print("\n📊 МАТРИЦЯ ПАРНИХ ПОРІВНЯНЬ (кількість голосів):")
        print("-" * 70)

        # Заголовок
        header = "     " + "  ".join(f"{alt:>4}" for alt in self.alternatives)
        print(header)
        print("-" * 70)

        # Рядки
        for i in range(self.n_alternatives):
            row = f"{self.alternatives[i]:>4} "
            for j in range(self.n_alternatives):
                if i == j:
                    row += "   - "
                else:
                    row += f"{vote_counts[i][j]:>4} "
            print(row)


# ============================================================================
# МЕХАНІЗМ БОРДА
# ============================================================================

class MechanizmBorda(GrupoveVidnoshennya):
    """Механізм узгодження за методом Борда"""

    def find_solution(self) -> Tuple[VidnoshennyaMatr, List[int], List[int]]:
        """
        Метод Борда: кожна альтернатива отримує бали від експертів
        залежно від позиції в їх ранжуванні

        Бали: (n-1) за 1-е місце, (n-2) за 2-е, ..., 0 за останнє

        Повертає: (групове відношення, індекси за балами, список балів)
        """
        # Обчислення балів Борда для кожної альтернативи
        borda_scores = [0] * self.n_alternatives

        for pref in self.expert_preferences:
            # Для кожного експерта підраховуємо бали
            for i in range(self.n_alternatives):
                # Бал = кількість альтернатив, які i перемагає
                score = sum(pref.B[i][j] for j in range(self.n_alternatives))
                borda_scores[i] += score

        # Сортування альтернатив за балами
        sorted_alternatives = sorted(range(self.n_alternatives),
                                     key=lambda i: borda_scores[i],
                                     reverse=True)

        # Побудова групового відношення на основі балів
        result_matrix = [[0] * self.n_alternatives for _ in range(self.n_alternatives)]

        for i in range(self.n_alternatives):
            for j in range(self.n_alternatives):
                if i != j and borda_scores[i] > borda_scores[j]:
                    result_matrix[i][j] = 1

        return VidnoshennyaMatr(result_matrix), sorted_alternatives, borda_scores

    def print_borda_scores(self, borda_scores):
        """Вивід балів Борда"""
        print("\n📊 БАЛИ БОРДА:")
        print("-" * 70)

        scores_with_names = [(self.alternatives[i], borda_scores[i])
                             for i in range(self.n_alternatives)]
        scores_with_names.sort(key=lambda x: x[1], reverse=True)

        for i, (alt, score) in enumerate(scores_with_names, 1):
            print(f"{i}. {alt}: {score} балів")


# ============================================================================
# МЕДІАНА КЕМЕНІ
# ============================================================================

class MedianaKemeni(GrupoveVidnoshennya):
    """Знаходження медіани Кемені"""

    def find_solution(self) -> Tuple[VidnoshennyaMatr, List[int], int]:
        """
        Медіана Кемені: лінійний порядок, який мінімізує суму відстаней
        до всіх експертних оцінок

        Відстань = кількість пар, в яких порядки не збігаються

        Повертає: (оптимальне відношення, оптимальний порядок, мінімальна відстань)
        """
        min_distance = float('inf')
        best_order = None

        # Перебираємо всі можливі лінійні порядки (перестановки)
        for perm in permutations(range(self.n_alternatives)):
            # Обчислюємо відстань Кемені для цього порядку
            distance = self.kemeny_distance(perm)

            if distance < min_distance:
                min_distance = distance
                best_order = list(perm)

        # Будуємо матрицю відношення з оптимального порядку
        result_matrix = [[0] * self.n_alternatives for _ in range(self.n_alternatives)]

        for i in range(len(best_order)):
            for j in range(i + 1, len(best_order)):
                # best_order[i] краще за best_order[j]
                result_matrix[best_order[i]][best_order[j]] = 1

        return VidnoshennyaMatr(result_matrix), best_order, min_distance

    def kemeny_distance(self, order: List[int]) -> int:
        """
        Обчислити відстань Кемені між заданим порядком та всіма експертними оцінками

        Відстань = сума відстаней до кожного експерта
        Відстань до експерта = кількість пар, де порядки різні
        """
        total_distance = 0

        # Для кожного експерта
        for pref in self.expert_preferences:
            expert_distance = 0

            # Перевіряємо всі пари альтернатив
            for i in range(len(order)):
                for j in range(i + 1, len(order)):
                    alt_i = order[i]
                    alt_j = order[j]

                    # В нашому порядку alt_i краще alt_j
                    # Перевіряємо чи це збігається з експертом
                    if pref.B[alt_i][alt_j] != 1:
                        # Якщо не збігається - додаємо до відстані
                        expert_distance += 1

            total_distance += expert_distance

        return total_distance

    def print_all_distances(self):
        """Вивід відстаней для всіх можливих порядків (для малих n)"""
        if self.n_alternatives > 5:
            print("Занадто багато перестановок для виведення")
            return

        print("\n📊 ВІДСТАНІ КЕМЕНІ ДЛЯ ВСІХ МОЖЛИВИХ ПОРЯДКІВ:")
        print("-" * 70)

        distances = []
        for perm in permutations(range(self.n_alternatives)):
            distance = self.kemeny_distance(perm)
            order_names = [self.alternatives[i] for i in perm]
            distances.append((distance, ' ≻ '.join(order_names)))

        # Сортуємо за відстанню
        distances.sort()

        print(f"\n{'Відстань':<12} {'Порядок':<50}")
        print("-" * 70)
        for dist, order in distances[:10]:  # Виводимо топ-10
            print(f"{dist:<12} {order:<50}")


def main():
    print("=" * 90)
    print(" " * 25 + "ЛАБОРАТОРНА РОБОТА №6")
    print(" " * 20 + "МЕТОДИ ПРИЙНЯТТЯ ГРУПОВИХ РІШЕНЬ")
    print(" " * 35 + "ВАРІАНТ 4")
    print("=" * 90)

    alternatives = ["a₁", "a₂", "a₃", "a₄", "a₅"]
    experts = ["P₁", "P₂", "P₃", "P₄", "P₅"]

    P1_matrix = [
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 0, 0]
    ]

    P2_matrix = [
        [0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0]
    ]

    P3_matrix = [
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [1, 1, 1, 1, 0]
    ]

    P4_matrix = [
        [0, 0, 0, 1, 0],
        [1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0]
    ]

    P5_matrix = [
        [0, 0, 0, 1, 1],
        [1, 0, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0]
    ]

    expert_preferences = [
        VidnoshennyaMatr(P1_matrix),
        VidnoshennyaMatr(P2_matrix),
        VidnoshennyaMatr(P3_matrix),
        VidnoshennyaMatr(P4_matrix),
        VidnoshennyaMatr(P5_matrix)
    ]

    print("\n📋 ВИХІДНІ ДАНІ:")
    print(f"Альтернативи: {alternatives}")
    print(f"Експерти: {experts}")
    print(f"Кількість експертів: {len(experts)}")
    print(f"Кількість альтернатив: {len(alternatives)}")

    base_group = MechanizmBilshostiGolosiv(alternatives, experts, expert_preferences)
    base_group.print_expert_preferences()

    print("\n" + "=" * 90)
    print("АНАЛІЗ ІНДИВІДУАЛЬНИХ ПЕРЕВАГ ЕКСПЕРТІВ")
    print("=" * 90)

    for i, expert in enumerate(experts):
        order = base_group.extract_linear_order(expert_preferences[i])
        order_names = [alternatives[idx] for idx in order]
        print(f"\n{expert}: {' ≻ '.join(order_names)}")

        scores = [sum(expert_preferences[i].B[j][k]
                      for k in range(len(alternatives)))
                  for j in range(len(alternatives))]
        print(f"  Бали: {', '.join(f'{alternatives[j]}:{scores[j]}' for j in range(len(alternatives)))}")

    print("\n" + "=" * 90)
    print("МЕТОД 1: ПРИНЦИП БІЛЬШОСТІ ГОЛОСІВ")
    print("=" * 90)

    print("\nОпис: aRb ⇔ більше половини експертів віддає перевагу a над b")
    print(f"Поріг більшості: {len(experts) / 2} голосів")

    majority = MechanizmBilshostiGolosiv(alternatives, experts, expert_preferences)
    majority_solution = majority.find_solution()
    majority.print_solution(majority_solution, "Групове відношення (більшість голосів)")

    print("\n📊 ДЕТАЛЬНИЙ АНАЛІЗ ПАРНИХ ПОРІВНЯНЬ:")
    print("-" * 70)
    for i in range(len(alternatives)):
        for j in range(i + 1, len(alternatives)):
            votes_i = sum(1 for pref in expert_preferences if pref.B[i][j] == 1)
            votes_j = sum(1 for pref in expert_preferences if pref.B[j][i] == 1)

            if majority_solution.B[i][j] == 1:
                winner = alternatives[i]
                result = f"{alternatives[i]} ≻ {alternatives[j]}"
            elif majority_solution.B[j][i] == 1:
                winner = alternatives[j]
                result = f"{alternatives[j]} ≻ {alternatives[i]}"
            else:
                result = f"{alternatives[i]} ~ {alternatives[j]}"

            print(f"  {alternatives[i]} vs {alternatives[j]}: {votes_i}:{votes_j} → {result}")

    print("\n" + "=" * 90)
    print("МЕТОД 2: ПРАВИЛО КОНДОРСЕ")
    print("=" * 90)

    print("\nОпис: Переможець Кондорсе - альтернатива, яка перемагає всі інші")
    print("      в парних порівняннях за більшістю голосів")

    condorcet = MechanizmKondorse(alternatives, experts, expert_preferences)
    condorcet_solution, winners, vote_counts = condorcet.find_solution()

    condorcet.print_vote_matrix(vote_counts)
    condorcet.print_solution(condorcet_solution, "Групове відношення (Кондорсе)")

    print("\n🏆 ПЕРЕМОЖЕЦЬ КОНДОРСЕ:")
    if winners:
        winner_names = [alternatives[i] for i in winners]
        print(f"   {', '.join(winner_names)}")

        for winner in winners:
            print(f"\n   {alternatives[winner]} перемагає:")
            for j in range(len(alternatives)):
                if j != winner:
                    votes_for = vote_counts[winner][j]
                    votes_against = vote_counts[j][winner]
                    print(f"      • {alternatives[j]}: {votes_for}:{votes_against}")
    else:
        print("   ⚠️  ПАРАДОКС КОНДОРСЕ: переможця не існує!")
        print("   Існує циклічність в групових перевагах")

    print("\n" + "=" * 90)
    print("МЕТОД 3: МЕТОД БОРДА")
    print("=" * 90)

    print("\nОпис: Кожна альтернатива отримує бали від експертів")
    print("      Бали = кількість альтернатив, які дана альтернатива перемагає")

    borda = MechanizmBorda(alternatives, experts, expert_preferences)
    borda_solution, sorted_alts, borda_scores = borda.find_solution()

    borda.print_borda_scores(borda_scores)
    borda.print_solution(borda_solution, "Групове відношення (Борда)")

    print("\n📊 ДЕТАЛЬНА ТАБЛИЦЯ БАЛІВ:")
    print("-" * 70)

    print(f"{'Альтернатива':<15}", end="")
    for expert in experts:
        print(f"{expert:>8}", end="")
    print(f"{'СУМА':>10}")
    print("-" * 70)

    for i in range(len(alternatives)):
        print(f"{alternatives[i]:<15}", end="")
        total = 0
        for pref in expert_preferences:
            score = sum(pref.B[i][j] for j in range(len(alternatives)))
            print(f"{score:>8}", end="")
            total += score
        print(f"{total:>10}")

    print("\n🏆 ПЕРЕМОЖЕЦЬ ЗА БОРДОЮ:")
    winner_idx = sorted_alts[0]
    print(f"   {alternatives[winner_idx]} з {borda_scores[winner_idx]} балами")

    print("\n" + "=" * 90)
    print("МЕТОД 4: МЕДІАНА КЕМЕНІ")
    print("=" * 90)

    print("\nОпис: Лінійний порядок, що мінімізує суму відстаней до всіх експертних оцінок")
    print("      Відстань = кількість пар альтернатив з різним порядком")

    kemeny = MedianaKemeni(alternatives, experts, expert_preferences)

    print("\n⏳ Обчислення медіани Кемені...")
    print(f"   Перебір {len(list(permutations(range(len(alternatives)))))} можливих порядків...")

    kemeny_solution, best_order, min_distance = kemeny.find_solution()

    kemeny.print_solution(kemeny_solution, "Групове відношення (медіана Кемені)")

    print(f"\n🏆 ОПТИМАЛЬНИЙ ПОРЯДОК:")
    order_names = [alternatives[i] for i in best_order]
    print(f"   {' ≻ '.join(order_names)}")
    print(f"   Сумарна відстань Кемені: {min_distance}")

    print("\n📊 ВІДСТАНІ ДО КОЖНОГО ЕКСПЕРТА:")
    print("-" * 70)

    for i, expert in enumerate(experts):
        distance = 0
        for j in range(len(best_order)):
            for k in range(j + 1, len(best_order)):
                alt_j = best_order[j]
                alt_k = best_order[k]
                if expert_preferences[i].B[alt_j][alt_k] != 1:
                    distance += 1
        print(f"   {expert}: {distance}")

    if len(alternatives) <= 5:
        kemeny.print_all_distances()

    print("\n" + "=" * 90)
    print("📊 ПОРІВНЯЛЬНА ТАБЛИЦЯ РЕЗУЛЬТАТІВ")
    print("=" * 90)

    majority_order = base_group.extract_linear_order(majority_solution)
    condorcet_order = base_group.extract_linear_order(condorcet_solution)
    borda_order = sorted_alts
    kemeny_order = best_order

    print(f"\n{'Метод':<25} {'Результуючий порядок':<40}")
    print("-" * 90)

    majority_names = [alternatives[i] for i in majority_order]
    print(f"{'Більшість голосів':<25} {' ≻ '.join(majority_names):<40}")

    if winners:
        condorcet_names = [alternatives[i] for i in condorcet_order]
        print(f"{'Кондорсе':<25} {' ≻ '.join(condorcet_names):<40}")
    else:
        print(f"{'Кондорсе':<25} {'Парадокс Кондорсе':<40}")

    borda_names = [alternatives[i] for i in borda_order]
    print(f"{'Борда':<25} {' ≻ '.join(borda_names):<40}")

    kemeny_names = [alternatives[i] for i in kemeny_order]
    print(f"{'Медіана Кемені':<25} {' ≻ '.join(kemeny_names):<40}")

    # Аналіз переможців
    print("\n🏆 ПЕРЕМОЖЦІ ЗА РІЗНИМИ МЕТОДАМИ:")
    print("-" * 70)
    print(f"   Більшість голосів:  {alternatives[majority_order[0]]}")
    if winners:
        print(f"   Кондорсе:           {alternatives[winners[0]]}")
    else:
        print(f"   Кондорсе:           Немає переможця")
    print(f"   Борда:              {alternatives[borda_order[0]]}")
    print(f"   Медіана Кемені:     {alternatives[kemeny_order[0]]}")

    all_winners = [majority_order[0]]
    if winners:
        all_winners.append(winners[0])
    all_winners.extend([borda_order[0], kemeny_order[0]])

    if len(set(all_winners)) == 1:
        print(f"\n✅ ВСІ МЕТОДИ УЗГОДЖЕНІ: переможець {alternatives[all_winners[0]]}")
    else:
        from collections import Counter
        winner_counts = Counter(all_winners)
        most_common = winner_counts.most_common(1)[0]
        print(f"\n⚠️  МЕТОДИ НЕ ПОВНІСТЮ УЗГОДЖЕНІ")
        print(f"   Найчастіший переможець: {alternatives[most_common[0]]} ({most_common[1]} методів)")

    print(f"Аналіз отриманих результатів:")
    print(f"─────────────────────────────────────────────────────────────────────")
    print(f"\n• Кількість експертів: {len(experts)}")
    print(f"• Кількість альтернатив: {len(alternatives)}")

    if len(set(all_winners)) == 1:
        print(f"\n• Всі методи узгоджені - це говорить про стабільність вибору")
        print(f"• Переможець {alternatives[all_winners[0]]} є справедливим компромісом")
    else:
        print(f"\n• Методи дають різні результати - це типово для групового вибору")
        print(f"• Різниця пояснюється різними принципами агрегування")

    if not winners:
        print(f"\n• Виявлено парадокс Кондорсе - циклічність в перевагах")
        print(f"• У такій ситуації рекомендується метод Борда або Кемені")


if __name__ == "__main__":
    main()
