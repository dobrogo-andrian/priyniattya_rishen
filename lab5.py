from abc import ABC, abstractmethod
from typing import List, Set, Dict, Tuple
import copy
import math


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
            print("  ".join(f"{val:>3}" for val in row))


class VidnoshennyaMatrMetr(VidnoshennyaMatr):
    """Клас для роботи з метризованими бінарними відношеннями"""

    def __init__(self, matrix: List[List], relation_type: str = None):
        self.M = copy.deepcopy(matrix)

        binary_matrix = [[1 if matrix[i][j] != 0 else 0
                          for j in range(len(matrix[i]))]
                         for i in range(len(matrix))]

        super().__init__(binary_matrix)

        if relation_type:
            self.relation_type = relation_type
        else:
            self.relation_type = 'additive'

    def get_v(self, i: int, j: int):
        return self.M[i][j]

    def set_v(self, i: int, j: int, value):
        self.M[i][j] = value
        self.B[i][j] = 1 if value != 0 else 0

    def print_matrix(self):
        """Вивід метризованої матриці"""
        for row in self.M:
            formatted_row = []
            for val in row:
                if val == 0:
                    formatted_row.append("0")
                elif isinstance(val, float):
                    if val.is_integer():
                        formatted_row.append(f"{int(val)}")
                    else:
                        formatted_row.append(f"{val:.3f}")
                else:
                    formatted_row.append(str(val))
            print("  ".join(f"{v:>7}" for v in formatted_row))


# ============================================================================
# МЕХАНІЗМИ ОБЧИСЛЕННЯ МІР БЛИЗЬКОСТІ
# ============================================================================

class MiraBlyzkosiLinOrder:
    """Міра близькості між відношеннями лінійного порядку"""

    def __init__(self, Q: VidnoshennyaMatr, R: VidnoshennyaMatr):
        """
        Ініціалізація
        Q, R - відношення лінійного порядку
        """
        self.Q = Q
        self.R = R
        self.n = Q.n

    def kendall_tau(self) -> Tuple[float, int, int]:
        """Коефіцієнт Кендалла - міра узгодженості двох порядків"""
        concordant = 0
        discordant = 0

        # Порівнюємо ВСІ пари елементів (i, j) де i < j
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Перевіряємо чи елемент i передує j в обох відношеннях
                q_order = self.Q.B[i][j] - self.Q.B[j][i]  # +1, 0, -1
                r_order = self.R.B[i][j] - self.R.B[j][i]

                if q_order * r_order > 0:  # Однаковий порядок
                    concordant += 1
                elif q_order * r_order < 0:  # Різний порядок
                    discordant += 1
                # Якщо q_order * r_order == 0, пара не враховується

        total_pairs = concordant + discordant

        if total_pairs == 0:
            return 0.0, 0, 0

        tau = (concordant - discordant) / total_pairs

        return tau, concordant, discordant

    def normalized_distance(self) -> Tuple[float, int]:
        """
        Нормалізована відстань між відношеннями

        d(Q,R) = |Q ⊕ R| / n²
        де ⊕ - симетрична різниця

        Повертає: (normalized_distance, differences_count)
        """
        differences = 0

        for i in range(self.n):
            for j in range(self.n):
                if self.Q.B[i][j] != self.R.B[i][j]:
                    differences += 1

        normalized = differences / (self.n * self.n)

        return normalized, differences

    def hamming_distance(self) -> int:
        """
        Відстань Хеммінга між матрицями відношень

        Кількість позицій, в яких матриці відрізняються
        """
        distance = 0

        for i in range(self.n):
            for j in range(self.n):
                if self.Q.B[i][j] != self.R.B[i][j]:
                    distance += 1

        return distance

    def similarity_coefficient(self) -> float:
        """
        Коефіцієнт подібності

        s(Q,R) = |Q ∩ R| / |Q ∪ R|

        Повертає: коефіцієнт від 0 до 1
        """
        intersection = 0
        union = 0

        for i in range(self.n):
            for j in range(self.n):
                if self.Q.B[i][j] == 1 and self.R.B[i][j] == 1:
                    intersection += 1
                if self.Q.B[i][j] == 1 or self.R.B[i][j] == 1:
                    union += 1

        if union == 0:
            return 1.0

        return intersection / union


class MiraBlyzkosiMetryzovani:
    """Міра близькості між метризованими відношеннями"""

    def __init__(self, S: VidnoshennyaMatrMetr, T: VidnoshennyaMatrMetr):
        """
        Ініціалізація
        S, T - метризовані відношення
        """
        self.S = S
        self.T = T
        self.n = S.n

    def euclidean_distance(self) -> float:
        """
        Евклідова відстань між матрицями

        d_E(S,T) = √(Σᵢⱼ(sᵢⱼ - tᵢⱼ)²)
        """
        sum_squares = 0

        for i in range(self.n):
            for j in range(self.n):
                diff = float(self.S.M[i][j]) - float(self.T.M[i][j])
                sum_squares += diff * diff

        return math.sqrt(sum_squares)

    def manhattan_distance(self) -> float:
        """
        Манхеттенська відстань

        d_M(S,T) = Σᵢⱼ|sᵢⱼ - tᵢⱼ|
        """
        sum_abs = 0

        for i in range(self.n):
            for j in range(self.n):
                diff = abs(float(self.S.M[i][j]) - float(self.T.M[i][j]))
                sum_abs += diff

        return sum_abs

    def chebyshev_distance(self) -> float:
        """
        Відстань Чебишева (максимальна)

        d_C(S,T) = maxᵢⱼ|sᵢⱼ - tᵢⱼ|
        """
        max_diff = 0

        for i in range(self.n):
            for j in range(self.n):
                diff = abs(float(self.S.M[i][j]) - float(self.T.M[i][j]))
                max_diff = max(max_diff, diff)

        return max_diff

    def frobenius_norm(self) -> float:
        """
        Норма Фробеніуса (нормалізована евклідова відстань)

        ||S-T||_F = √(Σᵢⱼ(sᵢⱼ - tᵢⱼ)²) / n
        """
        euclidean = self.euclidean_distance()
        return euclidean / self.n

    def correlation_coefficient(self) -> float:
        """
        Коефіцієнт кореляції між матрицями

        r = Σᵢⱼ(sᵢⱼ - s̄)(tᵢⱼ - t̄) / √(Σᵢⱼ(sᵢⱼ - s̄)² · Σᵢⱼ(tᵢⱼ - t̄)²)
        """
        # Обчислюємо середні
        s_mean = sum(float(self.S.M[i][j]) for i in range(self.n)
                     for j in range(self.n)) / (self.n * self.n)
        t_mean = sum(float(self.T.M[i][j]) for i in range(self.n)
                     for j in range(self.n)) / (self.n * self.n)

        # Обчислюємо кореляцію
        numerator = 0
        s_variance = 0
        t_variance = 0

        for i in range(self.n):
            for j in range(self.n):
                s_dev = float(self.S.M[i][j]) - s_mean
                t_dev = float(self.T.M[i][j]) - t_mean

                numerator += s_dev * t_dev
                s_variance += s_dev * s_dev
                t_variance += t_dev * t_dev

        denominator = math.sqrt(s_variance * t_variance)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def normalized_similarity(self) -> float:
        """
        Нормалізований коефіцієнт подібності

        sim(S,T) = 1 - d_E(S,T) / d_max
        де d_max - максимально можлива відстань
        """
        euclidean = self.euclidean_distance()

        # Максимальна можлива відстань
        max_s = max(abs(float(self.S.M[i][j])) for i in range(self.n)
                    for j in range(self.n))
        max_t = max(abs(float(self.T.M[i][j])) for i in range(self.n)
                    for j in range(self.n))
        d_max = math.sqrt(self.n * self.n) * (max_s + max_t)

        if d_max == 0:
            return 1.0

        return 1.0 - (euclidean / d_max)


class MiraBlyzkosiEkvivalentnist:
    """Структурна міра близькості між відношеннями еквівалентності"""

    def __init__(self, Q1: VidnoshennyaMatr, Q2: VidnoshennyaMatr):
        """
        Ініціалізація
        Q1, Q2 - відношення еквівалентності
        """
        self.Q1 = Q1
        self.Q2 = Q2
        self.n = Q1.n

    def extract_classes(self, Q: VidnoshennyaMatr) -> List[Set[int]]:
        """
        Виділити класи еквівалентності з відношення

        Повертає: список класів (множин елементів)
        """
        visited = [False] * self.n
        classes = []

        for i in range(self.n):
            if not visited[i]:
                # Знаходимо всі елементи еквівалентні i
                eq_class = set()
                for j in range(self.n):
                    if Q.B[i][j] == 1:
                        eq_class.add(j)
                        visited[j] = True
                classes.append(eq_class)

        return classes

    def rand_index(self) -> Tuple[float, int, int, int, int]:
        """
        Індекс Ренда (Rand Index)

        RI = (a + b) / C(n,2)
        де a - пари в одному класі в обох розбиттях
           b - пари в різних класах в обох розбиттях

        Повертає: (RI, a, b, c, d)
        """
        a = 0  # Разом в обох
        b = 0  # Окремо в обох
        c = 0  # Разом в Q1, окремо в Q2
        d = 0  # Окремо в Q1, разом в Q2

        for i in range(self.n):
            for j in range(i + 1, self.n):
                q1_together = self.Q1.B[i][j] == 1
                q2_together = self.Q2.B[i][j] == 1

                if q1_together and q2_together:
                    a += 1
                elif not q1_together and not q2_together:
                    b += 1
                elif q1_together and not q2_together:
                    c += 1
                else:  # not q1_together and q2_together
                    d += 1

        total_pairs = self.n * (self.n - 1) // 2

        if total_pairs == 0:
            ri = 1.0
        else:
            ri = (a + b) / total_pairs

        return ri, a, b, c, d

    def adjusted_rand_index(self) -> float:
        """
        Скоригований індекс Ренда (Adjusted Rand Index)

        Враховує випадкові збіги
        """
        classes1 = self.extract_classes(self.Q1)
        classes2 = self.extract_classes(self.Q2)

        # Таблиця спряженості
        n_ij = [[len(c1 & c2) for c2 in classes2] for c1 in classes1]

        # Суми по рядках та стовпцях
        a_i = [sum(row) for row in n_ij]
        b_j = [sum(n_ij[i][j] for i in range(len(classes1)))
               for j in range(len(classes2))]

        # Обчислення ARI
        sum_comb_n_ij = sum(n_ij[i][j] * (n_ij[i][j] - 1) / 2
                            for i in range(len(classes1))
                            for j in range(len(classes2)))

        sum_comb_a_i = sum(a * (a - 1) / 2 for a in a_i)
        sum_comb_b_j = sum(b * (b - 1) / 2 for b in b_j)

        n_comb = self.n * (self.n - 1) / 2

        expected = sum_comb_a_i * sum_comb_b_j / n_comb
        max_value = (sum_comb_a_i + sum_comb_b_j) / 2

        if max_value - expected == 0:
            return 1.0

        ari = (sum_comb_n_ij - expected) / (max_value - expected)

        return ari

    def jaccard_index(self) -> float:
        """
        Індекс Жаккара для розбиттів

        J = a / (a + c + d)
        де a - пари разом в обох
        """
        ri, a, b, c, d = self.rand_index()

        denominator = a + c + d

        if denominator == 0:
            return 1.0

        return a / denominator

    def fowlkes_mallows_index(self) -> float:
        """
        Індекс Фолкеса-Меллоуза (Fowlkes-Mallows Index)

        FM = √(PPV × TPR)
        де PPV - precision, TPR - recall
        """
        ri, a, b, c, d = self.rand_index()

        if a + c == 0 or a + d == 0:
            return 0.0

        precision = a / (a + c)
        recall = a / (a + d)

        fm = math.sqrt(precision * recall)

        return fm

    def partition_distance(self) -> int:
        """
        Відстань між розбиттями

        Мінімальна кількість елементів, які потрібно перемістити
        """
        ri, a, b, c, d = self.rand_index()

        return c + d

    def print_classes(self):
        """Вивід класів еквівалентності"""
        print("\n  Класи еквівалентності Q1:")
        classes1 = self.extract_classes(self.Q1)
        for i, cls in enumerate(classes1, 1):
            elements = [f"a{j + 1}" for j in sorted(cls)]
            print(f"    Клас {i}: {{{', '.join(elements)}}}")

        print("\n  Класи еквівалентності Q2:")
        classes2 = self.extract_classes(self.Q2)
        for i, cls in enumerate(classes2, 1):
            elements = [f"a{j + 1}" for j in sorted(cls)]
            print(f"    Клас {i}: {{{', '.join(elements)}}}")


# ============================================================================
# ОСНОВНА ПРОГРАМА - ВАРІАНТ 4
# ============================================================================

def main():
    print("=" * 90)
    print(" " * 25 + "ЛАБОРАТОРНА РОБОТА №5")
    print(" " * 20 + "МІРИ БЛИЗЬКОСТІ НА ВІДНОШЕННЯХ")
    print(" " * 35 + "ВАРІАНТ 4")
    print("=" * 90)

    print("\n" + "=" * 90)
    print("ЧАСТИНА 1: МІРА БЛИЗЬКОСТІ МІЖ ВІДНОШЕННЯМИ ЛІНІЙНОГО ПОРЯДКУ")
    print("=" * 90)

    Q_matrix = [
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1]
    ]

    R_matrix = [
        [1, 0, 0, 1, 0],  # a₁: R⁺(a₁) = {a₁, a₄}
        [0, 1, 0, 0, 1],  # a₂: з R⁻(a₂) = {a₂, a₅} випливає a₂ ≤ a₅
        [0, 0, 1, 0, 0],  # a₃: R⁺(a₃) = {a₃}
        [0, 0, 0, 1, 0],  # a₄: R⁺(a₄) = {a₄}
        [0, 0, 0, 0, 1]  # a₅: R⁺(a₅) = {a₅}
    ]

    Q = VidnoshennyaMatr(Q_matrix)
    R = VidnoshennyaMatr(R_matrix)

    print("\n📊 Відношення Q (лінійний порядок):")
    Q.print_matrix()

    print("\n📊 Відношення R (побудоване з множинного представлення):")
    print("   R⁺(a₁) = {a₁, a₄}")
    print("   R⁻(a₂) = {a₂, a₅}")
    print("   R⁺(a₃) = {a₃}")
    print("   R⁺(a₄) = {a₄}")
    print("   R⁺(a₅) = {a₅}")
    print("\nМатриця R:")
    R.print_matrix()

    print("\n" + "-" * 90)
    print("🔍 МІРИ БЛИЗЬКОСТІ МІЖ Q та R:")
    print("-" * 90)

    mira_lin = MiraBlyzkosiLinOrder(Q, R)

    tau, concordant, discordant = mira_lin.kendall_tau()
    print(f"""
    1️⃣  Коефіцієнт Кендалла (Kendall's τ):
       τ = {tau:.4f}
       Узгоджені пари (C): {concordant}
       Неузгоджені пари (D): {discordant}
    """)

    norm_dist, diff_count = mira_lin.normalized_distance()
    print(f"\n2️⃣  Нормалізована відстань:")
    print(f"   d(Q,R) = {norm_dist:.4f}")
    print(f"   Кількість відмінностей: {diff_count} з {Q.n * Q.n}")
    print(f"   Подібність: {(1 - norm_dist):.4f} ({(1 - norm_dist) * 100:.1f}%)")

    hamming = mira_lin.hamming_distance()
    print(f"\n3️⃣  Відстань Хеммінга:")
    print(f"   H(Q,R) = {hamming}")
    print(f"   Максимально можлива: {Q.n * Q.n}")

    similarity = mira_lin.similarity_coefficient()
    print(f"\n4️⃣  Коефіцієнт подібності (Жаккара):")
    print(f"   J(Q,R) = {similarity:.4f} ({similarity * 100:.1f}%)")

    print("\n" + "=" * 90)
    print("ЧАСТИНА 2: МІРА БЛИЗЬКОСТІ МІЖ МЕТРИЗОВАНИМИ ВІДНОШЕННЯМИ")
    print("=" * 90)

    S_matrix = [
        [0, -3, -1, -2, -3],
        [3, 0, 2, 1, 0],
        [1, -2, 0, -1, -2],
        [2, -1, 1, 0, -1],
        [3, 0, 2, 1, 0]
    ]

    T_matrix = [
        [0, -1, 2, 2, 0],
        [1, 0, 3, 3, 1],
        [-2, -3, 0, 0, -2],
        [-2, -3, 0, 0, -2],
        [0, -1, 2, 2, 0]
    ]

    S = VidnoshennyaMatrMetr(S_matrix, 'additive')
    T = VidnoshennyaMatrMetr(T_matrix, 'additive')

    print("\n📊 Метризоване відношення S:")
    S.print_matrix()

    print("\n📊 Метризоване відношення T:")
    T.print_matrix()

    # Обчислення мір близькості
    print("\n" + "-" * 90)
    print("🔍 МІРИ БЛИЗЬКОСТІ МІЖ S та T:")
    print("-" * 90)

    mira_metr = MiraBlyzkosiMetryzovani(S, T)

    euclidean = mira_metr.euclidean_distance()
    print(f"\n1️⃣  Евклідова відстань:")
    print(f"   d_E(S,T) = {euclidean:.4f}")
    print(f"   Формула: √(Σᵢⱼ(sᵢⱼ - tᵢⱼ)²)")

    manhattan = mira_metr.manhattan_distance()
    print(f"\n2️⃣  Манхеттенська відстань:")
    print(f"   d_M(S,T) = {manhattan:.4f}")
    print(f"   Формула: Σᵢⱼ|sᵢⱼ - tᵢⱼ|")

    chebyshev = mira_metr.chebyshev_distance()
    print(f"\n3️⃣  Відстань Чебишева:")
    print(f"   d_C(S,T) = {chebyshev:.4f}")
    print(f"   Формула: maxᵢⱼ|sᵢⱼ - tᵢⱼ|")

    frobenius = mira_metr.frobenius_norm()
    print(f"\n4️⃣  Норма Фробеніуса:")
    print(f"   ||S-T||_F = {frobenius:.4f}")
    print(f"   Нормалізована евклідова відстань")

    correlation = mira_metr.correlation_coefficient()
    print(f"\n5️⃣  Коефіцієнт кореляції:")
    print(f"   r(S,T) = {correlation:.4f}")
    print(
        f"   Інтерпретація: {'Сильна позитивна' if correlation > 0.7 else 'Помірна позитивна' if correlation > 0.3 else 'Слабка позитивна' if correlation > 0 else 'Негативна'} кореляція")

    norm_sim = mira_metr.normalized_similarity()
    print(f"\n6️⃣  Нормалізована подібність:")
    print(f"   sim(S,T) = {norm_sim:.4f} ({norm_sim * 100:.1f}%)")

    print("\n📊 Матриця різниць (S - T):")
    for i in range(S.n):
        row = []
        for j in range(S.n):
            diff = float(S.M[i][j]) - float(T.M[i][j])
            row.append(f"{diff:>7.1f}")
        print("  ".join(row))

    print("\n" + "=" * 90)
    print("ЧАСТИНА 3: СТРУКТУРНА МІРА БЛИЗЬКОСТІ МІЖ ВІДНОШЕННЯМИ ЕКВІВАЛЕНТНОСТІ")
    print("=" * 90)

    Q1_matrix = [
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1]
    ]

    Q2_matrix = [
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ]

    Q1 = VidnoshennyaMatr(Q1_matrix)
    Q2 = VidnoshennyaMatr(Q2_matrix)

    print("\n📊 Відношення еквівалентності Q1:")
    Q1.print_matrix()

    print("\n📊 Відношення еквівалентності Q2:")
    Q2.print_matrix()

    print("\n" + "-" * 90)
    print("🔍 СТРУКТУРНІ МІРИ БЛИЗЬКОСТІ МІЖ Q1 та Q2:")
    print("-" * 90)

    mira_ekv = MiraBlyzkosiEkvivalentnist(Q1, Q2)

    mira_ekv.print_classes()

    ri, a, b, c, d = mira_ekv.rand_index()
    print(f"\n1️⃣  Індекс Ренда (Rand Index):")
    print(f"   RI = {ri:.4f} ({ri * 100:.1f}%)")
    print(f"   a (разом в обох): {a}")
    print(f"   b (окремо в обох): {b}")
    print(f"   c (разом в Q1, окремо в Q2): {c}")
    print(f"   d (окремо в Q1, разом в Q2): {d}")
    print(f"   Формула: RI = (a + b) / C(n,2)")

    ari = mira_ekv.adjusted_rand_index()
    print(f"\n2️⃣  Скоригований індекс Ренда (ARI):")
    print(f"   ARI = {ari:.4f}")
    print(f"   Враховує випадкові збіги")
    print(
        f"   Інтерпретація: {'Відмінна' if ari > 0.9 else 'Хороша' if ari > 0.7 else 'Помірна' if ari > 0.5 else 'Слабка'} узгодженість")

    jaccard = mira_ekv.jaccard_index()
    print(f"\n3️⃣  Індекс Жаккара:")
    print(f"   J = {jaccard:.4f} ({jaccard * 100:.1f}%)")
    print(f"   Формула: J = a / (a + c + d)")

    fm = mira_ekv.fowlkes_mallows_index()
    print(f"\n4️⃣  Індекс Фолкеса-Меллоуза (FM):")
    print(f"   FM = {fm:.4f}")
    print(f"   Формула: FM = √(precision × recall)")

    partition_dist = mira_ekv.partition_distance()
    print(f"\n5️⃣  Відстань між розбиттями:")
    print(f"   d(Q1,Q2) = {partition_dist}")
    print(f"   Мінімальна кількість пар що треба переміст��ти")


    print("\n" + "=" * 90)
    print("📊 ЗВЕДЕНА ТАБЛИЦЯ РЕЗУЛЬТАТІВ")
    print("=" * 90)

    print("\n┌────────────────────────────────────────────────────────────────────┐")
    print("│            МІРИ БЛИЗЬКОСТІ ДЛЯ ЛІНІЙНИХ ПОРЯДКІВ (Q, R)           │")
    print("├────────────────────────────────────┬───────────────────────────────┤")
    print(f"│ Коефіцієнт Кендалла (τ)           │ {tau:>29.4f} │")
    print(f"│ Нормалізована відстань             │ {norm_dist:>29.4f} │")
    print(f"│ Відстань Хеммінга                  │ {hamming:>29} │")
    print(f"│ Коефіцієнт подібності              │ {similarity:>29.4f} │")
    print("└────────────────────────────────────┴───────────────────────────────┘")

    print("\n┌────────────────────────────────────────────────────────────────────┐")
    print("│         МІРИ БЛИЗЬКОСТІ ДЛЯ МЕТРИЗОВАНИХ ВІДНОШЕНЬ (S, T)         │")
    print("├────────────────────────────────────┬───────────────────────────────┤")
    print(f"│ Евклідова відстань                 │ {euclidean:>29.4f} │")
    print(f"│ Манхеттенська відстань             │ {manhattan:>29.4f} │")
    print(f"│ Відстань Чебишева                  │ {chebyshev:>29.4f} │")
    print(f"│ Норма Фробеніуса                   │ {frobenius:>29.4f} │")
    print(f"│ Коефіцієнт кореляції               │ {correlation:>29.4f} │")
    print(f"│ Нормалізована подібність           │ {norm_sim:>29.4f} │")
    print("└────────────────────────────────────┴───────────────────────────────┘")

    print("\n┌────────────────────────────────────────────────────────────────────┐")
    print("│   СТРУКТУРНІ МІРИ БЛИЗЬКОСТІ ДЛЯ ВІДНОШЕНЬ ЕКВІВАЛЕНТНОСТІ (Q1,Q2)│")
    print("├────────────────────────────────────┬───────────────────────────────┤")
    print(f"│ Індекс Ренда (RI)                  │ {ri:>29.4f} │")
    print(f"│ Скоригований індекс Ренда (ARI)    │ {ari:>29.4f} │")
    print(f"│ Індекс Жаккара                     │ {jaccard:>29.4f} │")
    print(f"│ Індекс Фолкеса-Меллоуза            │ {fm:>29.4f} │")
    print(f"│ Відстань між розбиттями            │ {partition_dist:>29} │")
    print("└────────────────────────────────────┴───────────────────────────────┘")

    # ========================================================================
    # ВИСНОВКИ
    # ========================================================================

    print("\n" + "=" * 90)
    print("📋 ВИСНОВКИ")
    print("=" * 90)

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                      АНАЛІЗ МІР БЛИЗЬКОСТІ                                 ║
╚════════════════════════════════════════════════════════════════════════════╝

1. МІРИ БЛИЗЬКОСТІ ДЛЯ ЛІНІЙНИХ ПОРЯДКІВ:
   ────────────────────────────────────────────────────────────────────────
""")
    print(
        f"   • Коефіцієнт Кендалла (τ = {tau:.4f}) показує {'високу' if tau > 0.7 else 'помірну' if tau > 0.3 else 'низьку'}")
    print(f"     узгодженість між порядками Q та R")
    print(f"   • Нормалізована відстань ({norm_dist:.4f}) вказує на {(1 - norm_dist) * 100:.1f}% подібності")
    print(f"   • З {Q.n * Q.n} елементів матриці {diff_count} відрізняються")
    print(f"   • Коефіцієнт Жаккара ({similarity:.4f}) підтверджує цю оцінку")

    print("""
2. МІРИ БЛИЗЬКОСТІ ДЛЯ МЕТРИЗОВАНИХ ВІДНОШЕНЬ:
   ────────────────────────────────────────────────────────────────────────
""")
    print(f"   • Евклідова відстань ({euclidean:.2f}) показує загальну різницю між матрицями")
    print(f"   • Манхеттенська відстань ({manhattan:.2f}) дає суму абсолютних різниць")
    print(f"   • Максимальна різниця в одній позиції (Чебишев): {chebyshev:.2f}")
    print(
        f"   • Коефіцієнт кореляції ({correlation:.4f}) вказує на {'позитивний' if correlation > 0 else 'негативний'}")
    print(f"     взаємозв'язок між відношеннями")
    print(f"   • Нормалізована подібність: {norm_sim * 100:.1f}%")

    print("""
3. СТРУКТУРНІ МІРИ ДЛЯ ВІДНОШЕНЬ ЕКВІВАЛЕНТНОСТІ:
   ────────────────────────────────────────────────────────────────────────
""")
    print(
        f"   • Індекс Ренда ({ri:.4f}) показує {'високу' if ri > 0.8 else 'помірну' if ri > 0.6 else 'низьку'} подібність розбиттів")
    print(f"   • Скоригований індекс Ренда ({ari:.4f}) враховує випадкові збіги")
    print(f"   • З {Q1.n * (Q1.n - 1) // 2} пар елементів:")
    print(f"     - {a} пар разом в обох розбиттях")
    print(f"     - {b} пар окремо в обох розбиттях")
    print(f"     - {c + d} пар мають різне розміщення")
    print(f"   • Індекс Жаккара ({jaccard:.4f}) підтверджує рівень подібності")


if __name__ == "__main__":
    main()