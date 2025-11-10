from abc import ABC, abstractmethod
from typing import List, Set, Dict
import copy
import time
import numpy as np

class Vidnoshennya(ABC):
    """
    Базовий абстрактний клас для бінарних відношень
    Визначає інтерфейс для роботи з відношеннями незалежно від представлення
    """
    def __init__(self, n: int = 0, elements: List[str] = None):
        """
        Конструктор
        n - кількість елементів
        elements - назви елементів (за замовчуванням a1, a2, ...)
        """
        self.n = n
        self.elements = elements if elements else [f"a{i + 1}" for i in range(n)]
    @abstractmethod
    def is_reflexive(self) -> bool:
        """Перевірка рефлексивності: ∀i (i,i)∈R"""
        pass
    @abstractmethod
    def is_symmetric(self) -> bool:
        """Перевірка симетричності: ∀i,j (i,j)∈R ⇒ (j,i)∈R"""
        pass
    @abstractmethod
    def is_antisymmetric(self) -> bool:
        """Перевірка антисиметричності: ∀i,j (i,j)∈R ∧ (j,i)∈R ⇒ i=j"""
        pass
    @abstractmethod
    def is_asymmetric(self) -> bool:
        """Перевірка асиметричності: ∀i,j (i,j)∈R ⇒ (j,i)∉R"""
        pass
    @abstractmethod
    def is_transitive(self) -> bool:
        """Перевірка транзитивності: ∀i,j,k (i,j)∈R ∧ (j,k)∈R ⇒ (i,k)∈R"""
        pass
    @abstractmethod
    def is_acyclic(self) -> bool:
        """Перевірка ациклічності: немає циклів"""
        pass
    @abstractmethod
    def is_connected(self) -> bool:
        """Перевірка зв'язності: ∀i,j i≠j ⇒ (i,j)∈R ∨ (j,i)∈R"""
        pass
    @abstractmethod
    def union(self, other):
        """Об'єднання відношень: R ∪ S"""
        pass
    @abstractmethod
    def intersection(self, other):
        """Перетин відношень: R ∩ S"""
        pass
    @abstractmethod
    def difference(self, other):
        r"""Різниця відношень: R \ S"""
        pass
    @abstractmethod
    def symmetric_difference(self, other):
        """Симетрична різниця: R ⊕ S"""
        pass
    @abstractmethod
    def complement(self):
        """Доповнення: R̄"""
        pass
    @abstractmethod
    def inverse(self):
        """Обернене відношення: R⁻¹"""
        pass
    @abstractmethod
    def composition(self, other):
        """Композиція відношень: R ∘ S"""
        pass
    @abstractmethod
    def transitive_closure(self):
        """Транзитивне замикання: R⁺"""
        pass
    @abstractmethod
    def symmetric_part(self):
        """Симетрична складова: R ∩ R⁻¹"""
        pass
    @abstractmethod
    def asymmetric_part(self):
        r"""Асиметрична складова: R \ R⁻¹"""
        pass
    @abstractmethod
    def has_relation(self, i: int, j: int) -> bool:
        """Перевірка наявності зв'язку (i,j)∈R"""
        pass
    @abstractmethod
    def add_relation(self, i: int, j: int):
        """Додати зв'язок (i,j)"""
        pass
    @abstractmethod
    def remove_relation(self, i: int, j: int):
        """Видалити зв'язок (i,j)"""
        pass
    @abstractmethod
    def print(self):
        """Вивести відношення"""
        pass
    @abstractmethod
    def to_matrix(self) -> List[List[int]]:
        """Отримати матричне представлення"""
        pass
    @abstractmethod
    def to_upper_sections(self) -> Dict[int, Set[int]]:
        """Отримати представлення верхніми перетинами"""
        pass


class VidnoshennyaMatr(Vidnoshennya):
    """
    Представлення бінарного відношення у вигляді матриці B
    B[i][j] = 1, якщо (i,j) ∈ R
    B[i][j] = 0, інакше
    """
    def __init__(self, n: int = 0, elements: List[str] = None):
        """Конструктор порожнього відношення"""
        super().__init__(n, elements)
        self.B = [[0] * n for _ in range(n)]
    @staticmethod
    def empty(n: int, elements: List[str] = None):
        """Порожнє відношення"""
        rel = VidnoshennyaMatr(n, elements)
        return rel
    @staticmethod
    def full(n: int, elements: List[str] = None):
        """Повне відношення (всі пари)"""
        rel = VidnoshennyaMatr(n, elements)
        rel.B = [[1] * n for _ in range(n)]
        return rel
    @staticmethod
    def diagonal(n: int, elements: List[str] = None):
        """Діагональне відношення (рефлексивне)"""
        rel = VidnoshennyaMatr(n, elements)
        for i in range(n):
            rel.B[i][i] = 1
        return rel
    @staticmethod
    def anti_diagonal(n: int, elements: List[str] = None):
        """Антидіагональне відношення"""
        rel = VidnoshennyaMatr(n, elements)
        for i in range(n):
            for j in range(n):
                if i != j:
                    rel.B[i][j] = 1
        return rel
    @staticmethod
    def from_matrix(matrix: List[List[int]], elements: List[str] = None):
        """Створити з матриці"""
        n = len(matrix)
        rel = VidnoshennyaMatr(n, elements)
        rel.B = copy.deepcopy(matrix)
        return rel
    @staticmethod
    def from_upper_sections(upper_sections: Dict[int, Set[int]], n: int,
                            elements: List[str] = None):
        """
        Створити з верхніх перетинів
        upper_sections: {i: {j | (i,j) ∈ R}}
        """
        rel = VidnoshennyaMatr(n, elements)
        for i, section in upper_sections.items():
            for j in section:
                rel.B[i][j] = 1
        return rel
    def is_reflexive(self) -> bool:
        """Рефлексивність"""
        for i in range(self.n):
            if self.B[i][i] == 0:
                return False
        return True
    def is_symmetric(self) -> bool:
        """Симетричність"""
        for i in range(self.n):
            for j in range(self.n):
                if self.B[i][j] != self.B[j][i]:
                    return False
        return True
    def is_antisymmetric(self) -> bool:
        """Антисиметричність"""
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.B[i][j] == 1 and self.B[j][i] == 1:
                    return False
        return True
    def is_asymmetric(self) -> bool:
        """Асиметричність"""
        for i in range(self.n):
            for j in range(self.n):
                if self.B[i][j] == 1 and self.B[j][i] == 1:
                    return False
        return True
    def is_transitive(self) -> bool:
        """Транзитивність"""
        for i in range(self.n):
            for j in range(self.n):
                if self.B[i][j] == 1:
                    for k in range(self.n):
                        if self.B[j][k] == 1 and self.B[i][k] == 0:
                            return False
        return True
    def is_acyclic(self) -> bool:
        """Ациклічність (немає циклів)"""
        visited = [0] * self.n  # 0 - не відвідано, 1 - в процесі, 2 - завершено
        def has_cycle_dfs(v):
            visited[v] = 1
            for u in range(self.n):
                if self.B[v][u] == 1:
                    if visited[u] == 1:
                        return True
                    if visited[u] == 0 and has_cycle_dfs(u):
                        return True
            visited[v] = 2
            return False
        for i in range(self.n):
            if visited[i] == 0:
                if has_cycle_dfs(i):
                    return False
        return True
    def is_connected(self) -> bool:
        """Зв'язність (лінійність)"""
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.B[i][j] == 0 and self.B[j][i] == 0:
                    return False
        return True
    def union(self, other):
        """Об'єднання"""
        result = VidnoshennyaMatr(self.n, self.elements)
        for i in range(self.n):
            for j in range(self.n):
                result.B[i][j] = max(self.B[i][j], other.B[i][j])
        return result
    def intersection(self, other):
        """Перетин"""
        result = VidnoshennyaMatr(self.n, self.elements)
        for i in range(self.n):
            for j in range(self.n):
                result.B[i][j] = min(self.B[i][j], other.B[i][j])
        return result
    def difference(self, other):
        """Різниця"""
        result = VidnoshennyaMatr(self.n, self.elements)
        for i in range(self.n):
            for j in range(self.n):
                result.B[i][j] = 1 if self.B[i][j] == 1 and other.B[i][j] == 0 else 0
        return result
    def symmetric_difference(self, other):
        """Симетрична різниця"""
        result = VidnoshennyaMatr(self.n, self.elements)
        for i in range(self.n):
            for j in range(self.n):
                result.B[i][j] = 1 if self.B[i][j] != other.B[i][j] else 0
        return result
    def complement(self):
        """Доповнення"""
        result = VidnoshennyaMatr(self.n, self.elements)
        for i in range(self.n):
            for j in range(self.n):
                result.B[i][j] = 1 - self.B[i][j]
        return result
    def inverse(self):
        """Обернене відношення (транспонування)"""
        result = VidnoshennyaMatr(self.n, self.elements)
        for i in range(self.n):
            for j in range(self.n):
                result.B[i][j] = self.B[j][i]
        return result
    def composition(self, other):
        """Композиція R ∘ S"""
        result = VidnoshennyaMatr(self.n, self.elements)
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    if self.B[i][k] == 1 and other.B[k][j] == 1:
                        result.B[i][j] = 1
                        break
        return result
    def transitive_closure(self):
        """Транзитивне замикання (алгоритм Уоршолла)"""
        result = VidnoshennyaMatr(self.n, self.elements)
        result.B = copy.deepcopy(self.B)
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    result.B[i][j] = result.B[i][j] or (result.B[i][k] and result.B[k][j])
        return result
    def symmetric_part(self):
        """Симетрична складова: R ∩ R⁻¹"""
        return self.intersection(self.inverse())
    def asymmetric_part(self):
        r"""Асиметрична складова: R \ R⁻¹"""
        return self.difference(self.inverse())
    def reachability(self):
        """Взаємна досягальність: R⁺ ∩ (R⁺)⁻¹"""
        trans_closure = self.transitive_closure()
        return trans_closure.intersection(trans_closure.inverse())
    def factorize(self):
        """
        Факторизація за симетричною складовою
        Повертає: (відношення еквівалентності, фактор-відношення)
        """
        sym = self.symmetric_part()
        equiv = sym.transitive_closure()
        classes = []
        visited = [False] * self.n
        for i in range(self.n):
            if not visited[i]:
                cls = set()
                for j in range(self.n):
                    if equiv.B[i][j] == 1:
                        cls.add(j)
                        visited[j] = True
                classes.append(cls)
        n_classes = len(classes)
        factor = VidnoshennyaMatr(n_classes, [f"K{i + 1}" for i in range(n_classes)])
        for i, cls_i in enumerate(classes):
            for j, cls_j in enumerate(classes):
                for elem_i in cls_i:
                    for elem_j in cls_j:
                        if self.B[elem_i][elem_j] == 1 and i != j:
                            factor.B[i][j] = 1
                            break
                    if factor.B[i][j] == 1:
                        break
        return equiv, factor, classes

    def decompose_to_dominance(self):
        """
        Розкладання на відношення домінування та еквівалентності
        Повертає: (домінування, еквівалентність)
        """
        dominance = self.asymmetric_part()
        equivalence = self.symmetric_part().transitive_closure()
        return dominance, equivalence
    def incomparability(self):
        """Відношення непорівняльності"""
        union_with_inverse = self.union(self.inverse())
        return union_with_inverse.complement()
    def find_maxima(self) -> Set[int]:
        """Максимуми: елементи без наступників"""
        maxima = set()
        for i in range(self.n):
            has_successor = False
            for j in range(self.n):
                if i != j and self.B[i][j] == 1:
                    has_successor = True
                    break
            if not has_successor:
                maxima.add(i)
        return maxima
    def find_minima(self) -> Set[int]:
        """Мінімуми: елементи без попередників"""
        minima = set()
        for j in range(self.n):
            has_predecessor = False
            for i in range(self.n):
                if i != j and self.B[i][j] == 1:
                    has_predecessor = True
                    break
            if not has_predecessor:
                minima.add(j)
        return minima
    def find_majorants(self) -> Set[int]:
        """Мажоранти: елементи, більші за всі інші"""
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
        """Міноранти: елементи, менші за всі інші"""
        minorants = set()
        for j in range(self.n):
            is_minorant = True
            for i in range(self.n):
                if i != j and self.B[i][j] == 0:
                    is_minorant = False
                    break
            if is_minorant:
                minorants.add(j)
        return minorants
    def has_relation(self, i: int, j: int) -> bool:
        """Перевірка наявності зв'язку"""
        return self.B[i][j] == 1
    def add_relation(self, i: int, j: int):
        """Додати зв'язок"""
        self.B[i][j] = 1
    def remove_relation(self, i: int, j: int):
        """Видалити зв'язок"""
        self.B[i][j] = 0
    def to_matrix(self) -> List[List[int]]:
        """Матричне представлення"""
        return copy.deepcopy(self.B)
    def to_upper_sections(self) -> Dict[int, Set[int]]:
        """Верхні перетини"""
        sections = {}
        for i in range(self.n):
            sections[i] = set()
            for j in range(self.n):
                if self.B[i][j] == 1:
                    sections[i].add(j)
        return sections
    def print(self):
        """Вивести матрицю"""
        print("\nМатричне представлення:")
        print("  ", "  ".join(self.elements))
        for i in range(self.n):
            print(f"{self.elements[i]:<3}", "  ".join(str(self.B[i][j]) for j in range(self.n)))
    def get_relation_type(self) -> str:
        """Визначити тип відношення"""
        types = []
        if self.is_reflexive():
            if self.is_symmetric():
                if self.is_transitive():
                    types.append("еквівалентність")
                else:
                    types.append("толерантність")
            elif self.is_antisymmetric():
                if self.is_transitive():
                    if self.is_connected():
                        types.append("лінійний порядок")
                    else:
                        types.append("частковий порядок")
        if self.is_asymmetric() and self.is_transitive():
            if self.is_connected():
                types.append("строгий лінійний порядок")
            else:
                types.append("строгий частковий порядок")
        return ", ".join(types) if types else "загальне відношення"


class VidnoshennyaZriz(Vidnoshennya):
    """
    Представлення бінарного відношення у вигляді верхніх перетинів R⁺
    R⁺(i) = {j | (i,j) ∈ R}
    """
    def __init__(self, n: int = 0, elements: List[str] = None):
        """Конструктор порожнього відношення"""
        super().__init__(n, elements)
        self.R_plus = {i: set() for i in range(n)}
    @staticmethod
    def empty(n: int, elements: List[str] = None):
        """Порожнє відношення"""
        return VidnoshennyaZriz(n, elements)
    @staticmethod
    def full(n: int, elements: List[str] = None):
        """Повне відношення"""
        rel = VidnoshennyaZriz(n, elements)
        for i in range(n):
            rel.R_plus[i] = set(range(n))
        return rel
    @staticmethod
    def diagonal(n: int, elements: List[str] = None):
        """Діагональне відношення"""
        rel = VidnoshennyaZriz(n, elements)
        for i in range(n):
            rel.R_plus[i] = {i}
        return rel
    @staticmethod
    def anti_diagonal(n: int, elements: List[str] = None):
        """Антидіагональне відношення"""
        rel = VidnoshennyaZriz(n, elements)
        for i in range(n):
            rel.R_plus[i] = set(j for j in range(n) if j != i)
        return rel
    @staticmethod
    def from_upper_sections(upper_sections: Dict[int, Set[int]], n: int,
                            elements: List[str] = None):
        """Створити з верхніх перетинів"""
        rel = VidnoshennyaZriz(n, elements)
        rel.R_plus = copy.deepcopy(upper_sections)
        return rel
    @staticmethod
    def from_matrix(matrix: List[List[int]], elements: List[str] = None):
        """Створити з матриці"""
        n = len(matrix)
        rel = VidnoshennyaZriz(n, elements)
        for i in range(n):
            for j in range(n):
                if matrix[i][j] == 1:
                    rel.R_plus[i].add(j)
        return rel
    def is_reflexive(self) -> bool:
        for i in range(self.n):
            if i not in self.R_plus[i]:
                return False
        return True
    def is_symmetric(self) -> bool:
        for i in range(self.n):
            for j in self.R_plus[i]:
                if i not in self.R_plus[j]:
                    return False
        return True
    def is_antisymmetric(self) -> bool:
        for i in range(self.n):
            for j in self.R_plus[i]:
                if i != j and i in self.R_plus[j]:
                    return False
        return True
    def is_asymmetric(self) -> bool:
        for i in range(self.n):
            for j in self.R_plus[i]:
                if i in self.R_plus[j]:
                    return False
        return True
    def is_transitive(self) -> bool:
        for i in range(self.n):
            for j in self.R_plus[i]:
                for k in self.R_plus[j]:
                    if k not in self.R_plus[i]:
                        return False
        return True
    def is_acyclic(self) -> bool:
        visited = [0] * self.n
        def has_cycle_dfs(v):
            visited[v] = 1
            for u in self.R_plus[v]:
                if visited[u] == 1:
                    return True
                if visited[u] == 0 and has_cycle_dfs(u):
                    return True
            visited[v] = 2
            return False
        for i in range(self.n):
            if visited[i] == 0:
                if has_cycle_dfs(i):
                    return False
        return True
    def is_connected(self) -> bool:
        for i in range(self.n):
            for j in range(self.n):
                if i != j and j not in self.R_plus[i] and i not in self.R_plus[j]:
                    return False
        return True
    def union(self, other):
        result = VidnoshennyaZriz(self.n, self.elements)
        for i in range(self.n):
            result.R_plus[i] = self.R_plus[i] | other.R_plus[i]
        return result
    def intersection(self, other):
        result = VidnoshennyaZriz(self.n, self.elements)
        for i in range(self.n):
            result.R_plus[i] = self.R_plus[i] & other.R_plus[i]
        return result
    def difference(self, other):
        result = VidnoshennyaZriz(self.n, self.elements)
        for i in range(self.n):
            result.R_plus[i] = self.R_plus[i] - other.R_plus[i]
        return result
    def symmetric_difference(self, other):
        result = VidnoshennyaZriz(self.n, self.elements)
        for i in range(self.n):
            result.R_plus[i] = self.R_plus[i] ^ other.R_plus[i]
        return result
    def complement(self):
        result = VidnoshennyaZriz(self.n, self.elements)
        all_elements = set(range(self.n))
        for i in range(self.n):
            result.R_plus[i] = all_elements - self.R_plus[i]
        return result
    def inverse(self):
        result = VidnoshennyaZriz(self.n, self.elements)
        for i in range(self.n):
            for j in self.R_plus[i]:
                result.R_plus[j].add(i)
        return result
    def composition(self, other):
        result = VidnoshennyaZriz(self.n, self.elements)
        for i in range(self.n):
            for k in self.R_plus[i]:
                result.R_plus[i] |= other.R_plus[k]
        return result
    def transitive_closure(self):
        result = VidnoshennyaZriz(self.n, self.elements)
        result.R_plus = copy.deepcopy(self.R_plus)
        for k in range(self.n):
            for i in range(self.n):
                if k in result.R_plus[i]:
                    result.R_plus[i] |= result.R_plus[k]
        return result
    def symmetric_part(self):
        return self.intersection(self.inverse())
    def asymmetric_part(self):
        return self.difference(self.inverse())
    def reachability(self):
        trans_closure = self.transitive_closure()
        return trans_closure.intersection(trans_closure.inverse())
    def has_relation(self, i: int, j: int) -> bool:
        return j in self.R_plus[i]
    def add_relation(self, i: int, j: int):
        self.R_plus[i].add(j)
    def remove_relation(self, i: int, j: int):
        self.R_plus[i].discard(j)
    def to_matrix(self) -> List[List[int]]:
        matrix = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in self.R_plus[i]:
                matrix[i][j] = 1
        return matrix
    def to_upper_sections(self) -> Dict[int, Set[int]]:
        return copy.deepcopy(self.R_plus)
    def print(self):
        print("\nПредставлення верхніми перетинами:")
        for i in range(self.n):
            elements_str = ", ".join(self.elements[j] for j in sorted(self.R_plus[i]))
            print(f"R⁺({self.elements[i]}) = {{{elements_str}}}")
    def find_maxima(self) -> Set[int]:
        maxima = set()
        for i in range(self.n):
            if len(self.R_plus[i] - {i}) == 0:
                maxima.add(i)
        return maxima
    def find_minima(self) -> Set[int]:
        inverse = self.inverse()
        return inverse.find_maxima()
    def find_majorants(self) -> Set[int]:
        majorants = set()
        for i in range(self.n):
            if len(self.R_plus[i]) == self.n:
                majorants.add(i)
        return majorants
    def find_minorants(self) -> Set[int]:
        inverse = self.inverse()
        return inverse.find_majorants()
def measure_time(func, *args):
    """Вимірювання часу виконання функції"""
    start = time.time()
    result = func(*args)
    end = time.time()
    return result, (end - start) * 1000  # в мілісекундах
def estimate_complexity(n: int, operation: str) -> str:
    """Оцінка складності операції"""
    complexities = {
        'union': f'O(n²) = O({n}²) = {n * n}',
        'intersection': f'O(n²) = O({n}²) = {n * n}',
        'composition': f'O(n³) = O({n}³) = {n * n * n}',
        'transitive_closure': f'O(n³) = O({n}³) = {n * n * n}',
        'properties': f'O(n²) = O({n}²) = {n * n}',
    }
    return complexities.get(operation, f'O(n²)')
def main():
    elements = ["a1", "a2", "a3", "a4", "a5"]
    n = 5
    P_matrix = [
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1]
    ]
    P = VidnoshennyaMatr.from_matrix(P_matrix, elements)
    Q_matrix = [
        [1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1]
    ]
    Q = VidnoshennyaMatr.from_matrix(Q_matrix, elements)
    R_sections = {
        0: set(),  # R⁺(a1) = {}
        1: {2},  # R⁺(a2) = {a3}
        2: set(),  # R⁺(a3) = {}
        3: set(),  # R⁺(a4) = {}
        4: {1, 2, 4}  # R⁺(a5) = {a2, a3, a5}
    }
    R = VidnoshennyaZriz.from_upper_sections(R_sections, n, elements)
    S_matrix = [
        [0, 2, 0, 1, 2],
        [-2, 0, -2, -1, 0],
        [0, 2, 0, 1, 2],
        [-1, 1, -1, 0, 1],
        [-2, 0, -2, -1, 0]
    ]
    T_matrix = [
        [0, 3, 1, 2, 1],
        [-3, 0, -2, -1, -2],
        [-1, 2, 0, 1, 0],
        [-2, 1, -1, 0, -1],
        [-1, 2, 0, 1, 0]
    ]
    print("\n📊 Відношення P (матриця):")
    P.print()
    print("\n📊 Відношення Q (матриця):")
    Q.print()
    print("\n📊 Відношення R (верхні перетини):")
    R.print()
    print("\n📊 Метризоване відношення S:")
    for row in S_matrix:
        print("  ", "  ".join(f"{v:>3}" for v in row))
    print("\n📊 Метризоване відношення T:")
    for row in T_matrix:
        print("  ", "  ".join(f"{v:>3}" for v in row))

    print("\n" + "=" * 100)
    print("ЗАВДАННЯ 1: Композиція QR та симетрична різниця Q⊕R")
    print("=" * 100)
    R_matr = VidnoshennyaMatr.from_upper_sections(R_sections, n, elements)
    QR, time_comp = measure_time(Q.composition, R_matr)
    print("\n1️⃣  Композиція Q ∘ R:")
    QR.print()
    print(f"\n⏱️  Час виконання: {time_comp:.4f} мс")
    print(f"📊 Складність: {estimate_complexity(n, 'composition')}")
    sym_diff, time_sym = measure_time(Q.symmetric_difference, R_matr)
    print("\n2️⃣  Симетрична різниця Q ⊕ R:")
    sym_diff.print()
    print(f"\n⏱️  Час виконання: {time_sym:.4f} мс")
    print(f"📊 Складність: {estimate_complexity(n, 'union')}")


    print("\n" + "=" * 100)
    print("ЗАВДАННЯ 2: Властивості відношення Q")
    print("=" * 100)
    properties = {
        'Рефлексивне': Q.is_reflexive(),
        'Симетричне': Q.is_symmetric(),
        'Асиметричне': Q.is_asymmetric(),
        'Антисиметричне': Q.is_antisymmetric(),
        'Транзитивне': Q.is_transitive(),
        'Ациклічне': Q.is_acyclic(),
        "Зв'язне": Q.is_connected()
    }
    print("\n📋 Властивості:")
    for prop, value in properties.items():
        symbol = "✓" if value else "✗"
        print(f"   {symbol} {prop}: {'ТАК' if value else 'НІ'}")
    rel_type = Q.get_relation_type()
    print(f"\n🏷️  Тип відношення: {rel_type}")

    print("\n" + "=" * 100)
    print("ЗАВДАННЯ 3: Транзитивне замикання та взаємна досягальність R")
    print("=" * 100)
    R_trans, time_trans = measure_time(R.transitive_closure)
    print("\n1️⃣  Транзитивне замикання R⁺:")
    R_trans.print()
    print("\nМатричне представлення R⁺:")
    R_trans_matr = VidnoshennyaMatr.from_upper_sections(R_trans.R_plus, n, elements)
    R_trans_matr.print()
    print(f"\n⏱️  Час виконання: {time_trans:.4f} мс")
    print(f"📊 Складність: {estimate_complexity(n, 'transitive_closure')}")
    R_reach, time_reach = measure_time(R.reachability)
    print("\n2️⃣  Взаємна досягальність R⁺ ∩ (R⁺)⁻¹:")
    R_reach.print()
    print(f"\n⏱️  Час виконання: {time_reach:.4f} мс")

    print("\n" + "=" * 100)
    print("ЗАВДАННЯ 4: Розкладання P на домінування та еквівалентність")
    print("=" * 100)
    dominance, equivalence = P.decompose_to_dominance()
    print("\n1️⃣  Відношення домінування (асиметрична складова P \\ P⁻¹):")
    dominance.print()
    print("\n2️⃣  Відношення еквівалентності (транз. замикання P ∩ P⁻¹):")
    equivalence.print()
    incomp = P.incomparability()
    print("\n3️⃣  Відношення непорівняльності:")
    incomp.print()

    print("\n" + "=" * 100)
    print("ЗАВДАННЯ 5: Факторизація P за симетричною складовою")
    print("=" * 100)
    equiv, factor, classes = P.factorize()
    print("\n1️⃣  Відношення еквівалентності:")
    equiv.print()
    print("\n2️⃣  Класи еквівалентності:")
    for i, cls in enumerate(classes, 1):
        elements_str = ", ".join(elements[j] for j in sorted(cls))
        print(f"   K{i}: {{{elements_str}}}")
    print("\n3️⃣  Фактор-відношення:")
    factor.print()

    print("\n" + "=" * 100)
    print("ЗАВДАННЯ 6: Екстремальні елементи відношення R")
    print("=" * 100)
    maxima = R.find_maxima()
    minima = R.find_minima()
    majorants = R.find_majorants()
    minorants = R.find_minorants()
    print("\n📊 Результати:")
    print(f"\n   Максимуми: {{{', '.join(elements[i] for i in sorted(maxima))}}}")
    print(f"   (елементи без наступників)")
    print(f"\n   Мінімуми: {{{', '.join(elements[i] for i in sorted(minima))}}}")
    print(f"   (елементи без попередників)")
    print(f"\n   Мажоранти: {{{', '.join(elements[i] for i in sorted(majorants))}}}")
    print(f"   (елементи, більші за всі)")
    print(f"\n   Міноранти: {{{', '.join(elements[i] for i in sorted(minorants))}}}")
    print(f"   (елементи, менші за всі)")

    print("\n" + "=" * 100)
    print("ЗАВДАННЯ 7: Міра близькості між Q та R")
    print("=" * 100)
    # Hamming distance
    Q_matrix_flat = [Q.B[i][j] for i in range(n) for j in range(n)]
    R_matrix_flat = [R_matr.B[i][j] for i in range(n) for j in range(n)]
    hamming = sum(1 for i in range(len(Q_matrix_flat)) if Q_matrix_flat[i] != R_matrix_flat[i])
    normalized_dist = hamming / (n * n)
    similarity = 1 - normalized_dist
    print(f"\n📊 Результати:")
    print(f"\n   Відстань Хеммінга: {hamming}")
    print(f"   Нормалізована відстань: {normalized_dist:.4f}")
    print(f"   Коефіцієнт подібності: {similarity:.4f} ({similarity * 100:.1f}%)")

    print("\n" + "=" * 100)
    print("ЗАВДАННЯ 8: Міра близькості між метризованими S та T")
    print("=" * 100)
    euclidean = np.sqrt(sum((S_matrix[i][j] - T_matrix[i][j]) ** 2
                            for i in range(n) for j in range(n)))
    manhattan = sum(abs(S_matrix[i][j] - T_matrix[i][j])
                    for i in range(n) for j in range(n))
    chebyshev = max(abs(S_matrix[i][j] - T_matrix[i][j])
                    for i in range(n) for j in range(n))
    print(f"\n📊 Результати:")
    print(f"\n   Евклідова відстань: {euclidean:.4f}")
    print(f"   Манхеттенська відстань: {manhattan:.4f}")
    print(f"   Відстань Чебишева: {chebyshev:.4f}")


    print("\n" + "=" * 100)
    print("📊 ПІДСУМКОВА СТАТИСТИКА ВИКОНАННЯ")
    print("=" * 100)

    print(f"\n⏱️  Час виконання операцій:")
    print(f"   • Композиція Q∘R: {time_comp:.4f} мс")
    print(f"   • Симетрична різниця: {time_sym:.4f} мс")
    print(f"   • Транзитивне замикання: {time_trans:.4f} мс")
    print(f"   • Взаємна досягальність: {time_reach:.4f} мс")

    print(f"\n📊 Складність операцій (для n={n}):")
    print(f"   • Бінарні операції (∪,∩,\\,⊕): O(n²) = {n * n} операцій")
    print(f"   • Композиція: O(n³) = {n * n * n} операцій")
    print(f"   • Транзитивне замикання: O(n³) = {n * n * n} операцій")
    print(f"   • Перевірка властивостей: O(n²) = {n * n} операцій")

    print(f"\n💾 Використання пам'яті:")
    print(f"   • Матричне представлення: {n}×{n} = {n * n} елементів")
    print(f"   • Представлення зрізами: до {n * n} елементів (залежить від щільності)")

    print("\n" + "=" * 100)
    print(" " * 35 + "ПРОГРАМУ ЗАВЕРШЕНО")
    print("=" * 100)


if __name__ == "__main__":
    main()

