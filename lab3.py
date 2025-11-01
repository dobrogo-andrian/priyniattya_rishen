from abc import ABC, abstractmethod
from typing import List
import copy
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
        """Перевірка рефлексивності: ∀i: (i,i) ∈ R"""
        for i in range(self.n):
            if self.B[i][i] == 0:
                return False
        return True
    def is_symmetric(self):
        """Перевірка симетричності: (i,j) ∈ R ⟹ (j,i) ∈ R"""
        for i in range(self.n):
            for j in range(self.n):
                if self.B[i][j] != self.B[j][i]:
                    return False
        return True
    def is_antisymmetric(self):
        """Перевірка антисиметричності: (i,j) ∈ R ∧ (j,i) ∈ R ⟹ i=j"""
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.B[i][j] != 0 and self.B[j][i] != 0:
                    return False
        return True
    def is_transitive(self):
        """Перевірка транзитивності: (i,j) ∈ R ∧ (j,k) ∈ R ⟹ (i,k) ∈ R"""
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
    def __init__(self, matrix: List[List], relation_type: str = None):
        self.M = copy.deepcopy(matrix)

        binary_matrix = [[1 if matrix[i][j] != 0 else 0
                          for j in range(len(matrix[i]))]
                         for i in range(len(matrix))]

        super().__init__(binary_matrix)
        if relation_type:
            self.relation_type = relation_type
        else:
            self.relation_type = self.determine_type()

    def get_v(self, i: int, j: int):
        return self.M[i][j]

    def set_v(self, i: int, j: int, value):
        self.M[i][j] = value
        self.B[i][j] = 1 if value != 0 else 0
    def determine_type(self) -> str:
        has_fractions = False
        has_negatives = False
        for i in range(self.n):
            for j in range(self.n):
                val = self.M[i][j]
                if val != 0:
                    try:
                        fval = float(val)
                        if fval < 0:
                            has_negatives = True
                        if 0 < fval < 1:
                            has_fractions = True
                    except:
                        pass
        if has_negatives:
            return 'additive'
        if has_fractions:
            return 'multiplicative'
        return 'additive'
    def is_consistent(self) -> bool:
        epsilon = 1e-6
        for i in range(self.n):
            for j in range(self.n):
                val_ij = self.M[i][j]
                val_ji = self.M[j][i]
                if val_ij == 0:
                    continue
                try:
                    if self.relation_type == 'additive':
                        if abs(float(val_ij) + float(val_ji)) > epsilon:
                            return False
                    elif self.relation_type == 'multiplicative':
                        if val_ji != 0:
                            if abs(float(val_ij) * float(val_ji) - 1) > epsilon:
                                return False
                except:
                    continue
        return True
    def is_additive_transitive(self) -> bool:
        epsilon = 1e-6
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    val_ij = self.M[i][j]
                    val_ik = self.M[i][k]
                    val_kj = self.M[k][j]
                    if val_ij == 0 or val_ik == 0 or val_kj == 0:
                        continue
                    try:
                        expected = float(val_ik) + float(val_kj)
                        actual = float(val_ij)
                        if abs(actual - expected) > epsilon:
                            return False
                    except:
                        continue
        return True
    def is_multiplicative_transitive(self) -> bool:
        epsilon = 1e-6
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    val_ij = self.M[i][j]
                    val_ik = self.M[i][k]
                    val_kj = self.M[k][j]
                    if val_ij == 0 or val_ik == 0 or val_kj == 0:
                        continue
                    try:
                        expected = float(val_ik) * float(val_kj)
                        actual = float(val_ij)
                        if abs(actual - expected) > epsilon:
                            return False
                    except:
                        continue
        return True
    def additive_union(self, other: 'VidnoshennyaMatrMetr') -> 'VidnoshennyaMatrMetr':
        result = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                p_val = self.M[i][j]
                q_val = other.M[i][j]
                if p_val != 0 and q_val != 0:
                    result[i][j] = (float(p_val) + float(q_val)) / 2
                elif p_val != 0:
                    result[i][j] = float(p_val)
                elif q_val != 0:
                    result[i][j] = float(q_val)
                else:
                    result[i][j] = 0
        return VidnoshennyaMatrMetr(result, 'additive')
    def additive_intersection(self, other: 'VidnoshennyaMatrMetr') -> 'VidnoshennyaMatrMetr':
        result = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                p_val = self.M[i][j]
                q_val = other.M[i][j]
                if p_val != 0 and q_val != 0:
                    result[i][j] = (float(p_val) + float(q_val)) / 2
                else:
                    result[i][j] = 0
        return VidnoshennyaMatrMetr(result, 'additive')

    def additive_difference(self, other: 'VidnoshennyaMatrMetr') -> 'VidnoshennyaMatrMetr':
        result = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                p_val = self.M[i][j]
                q_val = other.M[i][j]
                if p_val != 0 and q_val == 0:
                    result[i][j] = float(p_val)
                else:
                    result[i][j] = 0
        return VidnoshennyaMatrMetr(result, 'additive')

    def additive_composition(self, other: 'VidnoshennyaMatrMetr') -> 'VidnoshennyaMatrMetr':
        result = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                sum_val = 0
                count = 0
                for k in range(self.n):
                    p_ik = self.M[i][k]
                    q_kj = other.M[k][j]
                    if p_ik != 0 and q_kj != 0:
                        sum_val += float(p_ik) + float(q_kj)
                        count += 1
                if count > 0:
                    result[i][j] = sum_val / count
                else:
                    result[i][j] = 0
        return VidnoshennyaMatrMetr(result, 'additive')
    def multiplicative_union(self, other: 'VidnoshennyaMatrMetr') -> 'VidnoshennyaMatrMetr':
        result = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                p_val = self.M[i][j]
                q_val = other.M[i][j]
                if p_val != 0 and q_val != 0:
                    result[i][j] = float(p_val) * float(q_val)
                elif p_val != 0:
                    result[i][j] = float(p_val)
                elif q_val != 0:
                    result[i][j] = float(q_val)
                else:
                    result[i][j] = 0
        return VidnoshennyaMatrMetr(result, 'multiplicative')
    def multiplicative_intersection(self, other: 'VidnoshennyaMatrMetr') -> 'VidnoshennyaMatrMetr':
        result = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                p_val = self.M[i][j]
                q_val = other.M[i][j]
                if p_val != 0 and q_val != 0:
                    result[i][j] = float(p_val) * float(q_val)
                else:
                    result[i][j] = 0
        return VidnoshennyaMatrMetr(result, 'multiplicative')

    def multiplicative_difference(self, other: 'VidnoshennyaMatrMetr') -> 'VidnoshennyaMatrMetr':
        result = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                p_val = self.M[i][j]
                q_val = other.M[i][j]
                if p_val != 0 and q_val == 0:
                    result[i][j] = float(p_val)
                else:
                    result[i][j] = 0
        return VidnoshennyaMatrMetr(result, 'multiplicative')
    def multiplicative_composition(self, other: 'VidnoshennyaMatrMetr') -> 'VidnoshennyaMatrMetr':
        result = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                product = 1.0
                count = 0
                for k in range(self.n):
                    p_ik = self.M[i][k]
                    q_kj = other.M[k][j]
                    if p_ik != 0 and q_kj != 0:
                        product *= float(p_ik) * float(q_kj)
                        count += 1
                if count > 0:
                    result[i][j] = product ** (1.0 / count)
                else:
                    result[i][j] = 0
        return VidnoshennyaMatrMetr(result, 'multiplicative')
    def union(self, other: 'VidnoshennyaMatrMetr') -> 'VidnoshennyaMatrMetr':
        if self.relation_type == 'additive':
            return self.additive_union(other)
        else:
                return self.multiplicative_union(other)
    def intersection(self, other: 'VidnoshennyaMatrMetr') -> 'VidnoshennyaMatrMetr':
        if self.relation_type == 'additive':
            return self.additive_intersection(other)
        else:
            return self.multiplicative_intersection(other)
    def difference(self, other: 'VidnoshennyaMatrMetr') -> 'VidnoshennyaMatrMetr':
        if self.relation_type == 'additive':
            return self.additive_difference(other)
        else:
            return self.multiplicative_difference(other)
    def composition(self, other: 'VidnoshennyaMatrMetr') -> 'VidnoshennyaMatrMetr':
        if self.relation_type == 'additive':
            return self.additive_composition(other)
        else:
            return self.multiplicative_composition(other)
    def print_matrix(self):
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
    def print_properties(self):
        print(f"  Тип відношення: {'адитивне' if self.relation_type == 'additive' else 'мультиплікативне'}")
        print(f"  Узгодженість: {self.is_consistent()}")
        print(f"  Рефлексивність: {self.is_reflexive()}")
        print(f"  Симетричність: {self.is_symmetric()}")
        print(f"  Антисиметричність: {self.is_antisymmetric()}")
        print(f"  Транзитивність (бінарна): {self.is_transitive()}")
        if self.relation_type == 'additive':
            print(f"  Адитивна транзитивність: {self.is_additive_transitive()}")
        else:
            print(f"  Мультиплікативна транзитивність: {self.is_multiplicative_transitive()}")










def main():
    print("=" * 90)
    print(" " * 25 + "ЛАБОРАТОРНА РОБОТА №3")
    print(" " * 20 + "МЕТРИЗОВАНІ БІНАРНІ ВІДНОШЕННЯ")
    print(" " * 35 + "ВАРІАНТ 4")
    print("=" * 90)
    print("\n" + "=" * 90)
    print("ЧАСТИНА 1: АДИТИВНІ ВІДНОШЕННЯ P та Q")
    print("=" * 90)
    # Вхідні дані з варіанту 4
    P_matrix = [
        [0, 0, 0, 0, 0],
        [7, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0],
        [9, 2, 0, 4, 0]
    ]

    Q_matrix = [
        [0, 0, 0, 5, 0],
        [2, 0, 0, 7, 1],
        [4, 2, 0, 9, 3],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 6, 0]
    ]

    P = VidnoshennyaMatrMetr(P_matrix, 'additive')
    Q = VidnoshennyaMatrMetr(Q_matrix, 'additive')

    print("\n📊 Відношення P:")
    P.print_matrix()
    print("\n🔍 Властивості та тип відношення P:")
    P.print_properties()

    print("\n" + "-" * 90)

    print("\n📊 Відношення Q:")
    Q.print_matrix()
    print("\n🔍 Властивості та тип відношення Q:")
    Q.print_properties()

    print("\n" + "-" * 90)
    print("🔸 ОПЕРАЦІЇ НАД АДИТИВНИМИ ВІДНОШЕННЯМИ")
    print("-" * 90)

    print("\n1️⃣  ОБ'ЄДНАННЯ P ∪ Q (адитивна алгебра):")
    P_union_Q = P.union(Q)
    P_union_Q.print_matrix()

    print("\n2️⃣  ПЕРЕТИН P ∩ Q (адитивна алгебра):")
    P_inter_Q = P.intersection(Q)
    P_inter_Q.print_matrix()

    print("\n3️⃣  РІЗНИЦЯ P \\ Q (адитивна алгебра):")
    P_diff_Q = P.difference(Q)
    P_diff_Q.print_matrix()

    print("\n4️⃣  КОМПОЗИЦІЯ P ∘ Q (адитивна алгебра):")
    P_comp_Q = P.composition(Q)
    P_comp_Q.print_matrix()
    print("\n" + "=" * 90)
    print("ЧАСТИНА 2: МУЛЬТИПЛІКАТИВНІ ВІДНОШЕННЯ P1 та Q1")
    print("=" * 90)

    P1_matrix = [
        [1, 0, 0, 0, 5],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 10, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ]

    Q1_matrix = [
        [1, 3, 6, 6, 12],
        [0, 1, 2, 2, 4],
        [0, 0, 1, 1, 2],
        [0, 0, 1, 1, 2],
        [1, 0, 0, 0, 1]
    ]

    P1 = VidnoshennyaMatrMetr(P1_matrix, 'multiplicative')
    Q1 = VidnoshennyaMatrMetr(Q1_matrix, 'multiplicative')

    print("\n📊 Відношення P1:")
    P1.print_matrix()
    print("\n🔍 Властивості та тип відношення P1:")
    P1.print_properties()

    print("\n" + "-" * 90)
    print("\n📊 Відношення Q1:")
    Q1.print_matrix()
    print("\n🔍 Властивості та тип відношення Q1:")
    Q1.print_properties()

    print("\n" + "-" * 90)
    print("🔸 ОПЕРАЦІЇ НАД МУЛЬТИПЛІКАТИВНИМИ ВІДНОШЕННЯМИ")
    print("-" * 90)

    print("\n1️⃣  ОБ'ЄДНАННЯ P1 ∪ Q1 (мультиплікативна алгебра):")
    P1_union_Q1 = P1.union(Q1)
    P1_union_Q1.print_matrix()

    print("\n2️⃣  ПЕРЕТИН P1 ∩ Q1 (мультиплікативна алгебра):")
    P1_inter_Q1 = P1.intersection(Q1)
    P1_inter_Q1.print_matrix()

    print("\n3️⃣  РІЗНИЦЯ P1 \\ Q1 (мультиплікативна алгебра):")
    P1_diff_Q1 = P1.difference(Q1)
    P1_diff_Q1.print_matrix()

    print("\n4️⃣  КОМПОЗИЦІЯ P1 ∘ Q1 (мультиплікативна алгебра):")
    P1_comp_Q1 = P1.composition(Q1)
    P1_comp_Q1.print_matrix()

if __name__ == "__main__":
    main()