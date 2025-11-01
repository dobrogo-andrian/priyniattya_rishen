from abc import ABC
from typing import List, Tuple, Dict
import numpy as np
from collections import Counter


# ============================================================================
# БАЗОВИЙ КЛАС ДЛЯ ПРИЙНЯТТЯ РІШЕНЬ В УМОВАХ НЕВИЗНАЧЕНОСТІ
# ============================================================================

class DecisionUnderUncertainty(ABC):
    """Базовий клас для прийняття рішень в умовах невизначеності"""

    def __init__(self, strategies: List[str], states: List[str],
                 payoff_matrix: List[List[float]], probabilities: List[float] = None):
        """
        Ініціалізація

        strategies - список стратегій (альтернатив)
        states - список станів природи
        payoff_matrix - платіжна матриця [стратегія][стан]
        probabilities - ймовірності станів природи
        """
        self.strategies = strategies
        self.states = states
        self.payoff_matrix = np.array(payoff_matrix)
        self.probabilities = np.array(probabilities) if probabilities else None

        self.n_strategies = len(strategies)
        self.n_states = len(states)

        # Обчислюємо матрицю ризиків
        self.risk_matrix = self._calculate_risk_matrix()

    def _calculate_risk_matrix(self) -> np.ndarray:
        """
        Обчислення матриці ризиків
        r_ij = c_j - a_ij, де c_j = max_i(a_ij)
        """
        risk_matrix = np.zeros_like(self.payoff_matrix)

        for j in range(self.n_states):
            max_payoff = np.max(self.payoff_matrix[:, j])
            risk_matrix[:, j] = max_payoff - self.payoff_matrix[:, j]

        return risk_matrix

    def print_payoff_matrix(self):
        """Вивід платіжної матриці"""
        print("\n📊 ПЛАТІЖНА МАТРИЦЯ (виграші):")
        print("=" * 100)

        # Заголовок
        header = f"{'Стратегія':<15}"
        for state in self.states:
            header += f"{state:>12}"
        print(header)
        print("-" * 100)

        # Дані
        for i, strategy in enumerate(self.strategies):
            row = f"{strategy:<15}"
            for j in range(self.n_states):
                row += f"{self.payoff_matrix[i][j]:>12.2f}"
            print(row)
        print("=" * 100)

    def print_risk_matrix(self):
        """Вивід матриці ризиків"""
        print("\n📊 МАТРИЦЯ РИЗИКІВ:")
        print("=" * 100)

        # Заголовок
        header = f"{'Стратегія':<15}"
        for state in self.states:
            header += f"{state:>12}"
        print(header)
        print("-" * 100)

        # Дані
        for i, strategy in enumerate(self.strategies):
            row = f"{strategy:<15}"
            for j in range(self.n_states):
                row += f"{self.risk_matrix[i][j]:>12.2f}"
            print(row)
        print("=" * 100)

    def wald_criterion(self) -> Tuple[int, float]:
        """
        Максимінний критерій Вальда (крайній песимізм)
        W = max_i(min_j(a_ij))
        """
        min_payoffs = np.min(self.payoff_matrix, axis=1)
        best_strategy = np.argmax(min_payoffs)
        best_value = min_payoffs[best_strategy]

        return best_strategy, best_value

    def maximax_criterion(self) -> Tuple[int, float]:
        """
        Максимаксний критерій (крайній оптимізм)
        M = max_i(max_j(a_ij))
        """
        max_payoffs = np.max(self.payoff_matrix, axis=1)
        best_strategy = np.argmax(max_payoffs)
        best_value = max_payoffs[best_strategy]

        return best_strategy, best_value

    def savage_criterion(self) -> Tuple[int, float]:
        """
        Мінімаксний критерій Севіджа
        S = min_i(max_j(r_ij))
        """
        max_risks = np.max(self.risk_matrix, axis=1)
        best_strategy = np.argmin(max_risks)
        best_value = max_risks[best_strategy]

        return best_strategy, best_value

    def laplace_criterion(self) -> Tuple[int, float]:
        """
        Критерій Лапласа (рівноймовірні стани)
        L = max_i(1/n * sum_j(a_ij))
        """
        mean_payoffs = np.mean(self.payoff_matrix, axis=1)
        best_strategy = np.argmax(mean_payoffs)
        best_value = mean_payoffs[best_strategy]

        return best_strategy, best_value

    def bayes_laplace_criterion(self) -> Tuple[int, float]:
        """
        Критерій Байєса-Лапласа
        B = max_i(sum_j(q_j * a_ij))
        """
        if self.probabilities is None:
            raise ValueError("Ймовірності не задані!")

        expected_payoffs = np.sum(self.payoff_matrix * self.probabilities, axis=1)
        best_strategy = np.argmax(expected_payoffs)
        best_value = expected_payoffs[best_strategy]

        return best_strategy, best_value

    def hurwicz_criterion(self, alpha: float = 0.5) -> Tuple[int, float]:
        """
        Критерій песимізму-оптимізму Гурвіца
        H = max_i(alpha * min_j(a_ij) + (1-alpha) * max_j(a_ij))

        alpha - коефіцієнт песимізму (0 - крайній оптимізм, 1 - крайній песимізм)
        """
        min_payoffs = np.min(self.payoff_matrix, axis=1)
        max_payoffs = np.max(self.payoff_matrix, axis=1)

        hurwicz_values = alpha * min_payoffs + (1 - alpha) * max_payoffs
        best_strategy = np.argmax(hurwicz_values)
        best_value = hurwicz_values[best_strategy]

        return best_strategy, best_value

    def hodges_lehmann_criterion(self, lambda_param: float = 0.5) -> Tuple[int, float]:
        """
        Критерій Ходжеса-Лемана
        HL = max_i(lambda * B_i + (1-lambda) * W_i)

        lambda_param - ступінь довіри до ймовірностей
        """
        if self.probabilities is None:
            raise ValueError("Ймовірності не задані!")

        # Байєсівські оцінки
        bayes_values = np.sum(self.payoff_matrix * self.probabilities, axis=1)

        # Вальдівські оцінки (песимістичні)
        wald_values = np.min(self.payoff_matrix, axis=1)

        # Комбінація
        hl_values = lambda_param * bayes_values + (1 - lambda_param) * wald_values
        best_strategy = np.argmax(hl_values)
        best_value = hl_values[best_strategy]

        return best_strategy, best_value

    def analyze_all_criteria(self, alpha_hurwicz: float = 0.6, lambda_hl: float = 0.5):
        """Аналіз за всіма критеріями"""
        print("\n" + "=" * 100)
        print("🔍 АНАЛІЗ ЗА КРИТЕРІЯМИ ПРИЙНЯТТЯ РІШЕНЬ")
        print("=" * 100)

        results = {}

        # 1. Вальда
        wald_idx, wald_val = self.wald_criterion()
        results['Вальда'] = wald_idx
        print(f"\n1️⃣  Критерій ВАЛЬДА (максимінний, крайній песимізм):")
        print(f"   Оптимальна стратегія: {self.strategies[wald_idx]}")
        print(f"   Гарантований виграш: {wald_val:.2f}")
        print(f"   Принцип: «завжди розраховуй на гірше»")

        # 2. Максимаксний
        maximax_idx, maximax_val = self.maximax_criterion()
        results['Максимаксний'] = maximax_idx
        print(f"\n2️⃣  Критерій МАКСИМАКСНИЙ (крайній оптимізм):")
        print(f"   Оптимальна стратегія: {self.strategies[maximax_idx]}")
        print(f"   Максимальний можливий виграш: {maximax_val:.2f}")
        print(f"   Принцип: «розраховуй на найкраще»")

        # 3. Севіджа
        savage_idx, savage_val = self.savage_criterion()
        results['Севіджа'] = savage_idx
        print(f"\n3️⃣  Критерій СЕВІДЖА (мінімаксний ризик):")
        print(f"   Оптимальна стратегія: {self.strategies[savage_idx]}")
        print(f"   Мінімальний максимальний ризик: {savage_val:.2f}")
        print(f"   Принцип: «мінімізуй максимальний ризик»")

        # 4. Лапласа
        laplace_idx, laplace_val = self.laplace_criterion()
        results['Лапласа'] = laplace_idx
        print(f"\n4️⃣  Критерій ЛАПЛАСА (рівноймовірні стани):")
        print(f"   Оптимальна стратегія: {self.strategies[laplace_idx]}")
        print(f"   Середній виграш: {laplace_val:.2f}")
        print(f"   Припущення: всі стани природи рівноймовірні")

        # 5. Байєса-Лапласа
        if self.probabilities is not None:
            bayes_idx, bayes_val = self.bayes_laplace_criterion()
            results['Байєса-Лапласа'] = bayes_idx
            print(f"\n5️⃣  Критерій БАЙЄСА-ЛАПЛАСА (відомі ймовірності):")
            print(f"   Оптимальна стратегія: {self.strategies[bayes_idx]}")
            print(f"   Математичне очікування виграшу: {bayes_val:.2f}")
            print(f"   Використані ймовірності: {self.probabilities}")

        # 6. Гурвіца
        hurwicz_idx, hurwicz_val = self.hurwicz_criterion(alpha_hurwicz)
        results['Гурвіца'] = hurwicz_idx
        print(f"\n6️⃣  Критерій ГУРВІЦА (α = {alpha_hurwicz}):")
        print(f"   Оптимальна стратегія: {self.strategies[hurwicz_idx]}")
        print(f"   Оцінка Гурвіца: {hurwicz_val:.2f}")
        print(f"   Коефіцієнт песимізму α = {alpha_hurwicz}")

        # 7. Ходжеса-Лемана
        if self.probabilities is not None:
            hl_idx, hl_val = self.hodges_lehmann_criterion(lambda_hl)
            results['Ходжеса-Лемана'] = hl_idx
            print(f"\n7️⃣  Критерій ХОДЖЕСА-ЛЕМАНА (λ = {lambda_hl}):")
            print(f"   Оптимальна стратегія: {self.strategies[hl_idx]}")
            print(f"   Комбінована оцінка: {hl_val:.2f}")
            print(f"   Довіра до ймовірностей λ = {lambda_hl}")

        # Зведена таблиця
        self._print_summary_table(results)

        return results

    def _print_summary_table(self, results: Dict[str, int]):
        """Зведена таблиця результатів"""
        print("\n" + "=" * 100)
        print("📊 ЗВЕДЕНА ТАБЛИЦЯ РЕЗУЛЬТАТІВ")
        print("=" * 100)

        print(f"\n{'Критерій':<25} {'Оптимальна стратегія':<30}")
        print("-" * 100)

        for criterion, strategy_idx in results.items():
            print(f"{criterion:<25} {self.strategies[strategy_idx]:<30}")

        # Аналіз узгодженості
        print("\n" + "-" * 100)
        print("🎯 АНАЛІЗ УЗГОДЖЕНОСТІ:")

        from collections import Counter
        strategy_counts = Counter(results.values())

        if len(strategy_counts) == 1:
            print(f"   ✅ ВСІ КРИТЕРІЇ УЗГОДЖЕНІ!")
            print(f"   Рекомендована стратегія: {self.strategies[list(results.values())[0]]}")
        else:
            most_common = strategy_counts.most_common(1)[0]
            print(f"   ⚠️  Критерії дають різні результати")
            print(f"   Найчастіша рекомендація: {self.strategies[most_common[0]]}")
            print(f"   Кількість критеріїв: {most_common[1]} з {len(results)}")

            print("\n   Розподіл голосів:")
            for strategy_idx, count in strategy_counts.most_common():
                print(f"      • {self.strategies[strategy_idx]}: {count} критеріїв")


class ProductionDecision(DecisionUnderUncertainty):
    """Прийняття рішення про виробництво продукції що псується"""

    def __init__(self, production_cost: float, transport_cost: float,
                 selling_price: float, demand_levels: List[int],
                 probabilities: List[float]):
        """
        production_cost - витрати на виробництво одного ящика (A1)
        transport_cost - витрати на транспортування (C)
        selling_price - ціна продажу (A2)
        demand_levels - можливі рівні попиту [B1, B2, B3, B4, B5]
        probabilities - ймовірності попиту [P1, P2, P3, P4, P5]
        """
        self.production_cost = production_cost
        self.transport_cost = transport_cost
        self.selling_price = selling_price
        self.demand_levels = demand_levels

        # Стратегії - виробити B1, B2, B3, B4 або B5 ящиків
        strategies = [f"Виробити {d} ящ." for d in demand_levels]

        # Стани природи - попит буде B1, B2, B3, B4 або B5
        states = [f"Попит {d} ящ." for d in demand_levels]

        # Обчислюємо платіжну матрицю
        payoff_matrix = self._calculate_payoffs()

        super().__init__(strategies, states, payoff_matrix, probabilities)

    def _calculate_payoffs(self) -> List[List[float]]:
        """
        Обчислення прибутків для кожної комбінації стратегії та стану природи

        Прибуток = Дохід - Витрати
        Дохід = min(вироблено, попит) * ціна_продажу
        Витрати = вироблено * (собівартість + транспорт)
        """
        n = len(self.demand_levels)
        payoff_matrix = []

        for produced in self.demand_levels:
            row = []
            for demand in self.demand_levels:
                # Скільки продано
                sold = min(produced, demand)

                # Дохід від продажу
                revenue = sold * self.selling_price

                # Витрати на виробництво та транспортування
                costs = produced * (self.production_cost + self.transport_cost)

                # Прибуток
                profit = revenue - costs

                row.append(profit)

            payoff_matrix.append(row)

        return payoff_matrix

    def print_detailed_analysis(self):
        """Детальний аналіз задачі"""
        print("\n" + "=" * 100)
        print("ЗАВДАННЯ №1: ВИРОБНИЦТВО ПРОДУКЦІЇ ЩО ШВИДКО ПСУЄТЬСЯ")
        print("=" * 100)

        print("\n📋 ВХІДНІ ДАНІ:")
        print(f"   Собівартість виробництва 1 ящика: {self.production_cost} грн")
        print(f"   Витрати на транспортування 1 ящика: {self.transport_cost} грн")
        print(f"   Ціна продажу 1 ящика: {self.selling_price} грн")
        print(f"   Прибуток з 1 проданого ящика: {self.selling_price - self.production_cost - self.transport_cost} грн")

        print(f"\n   Можливі рівні попиту: {self.demand_levels}")
        print(f"   Ймовірності попиту: {self.probabilities}")

        print("\n💡 ЛОГІКА ЗАДАЧІ:")
        print("   • Продукція швидко псується - якщо не продано, прибуток = 0")
        print("   • Виробляємо заздалегідь, не знаючи точного попиту")
        print("   • Якщо виробимо більше попиту - понесемо збитки")
        print("   • Якщо виробимо менше попиту - втратимо можливий прибуток")


class LogisticsDecision(DecisionUnderUncertainty):
    """Прийняття рішення про постачання лісу"""

    def __init__(self, distance: float, cost_price: float,
                 selling_prices: List[float], volumes: List[float],
                 transport_costs: List[float], penalty: float,
                 probabilities: List[float]):
        """
        distance - довжина маршруту (D)
        cost_price - собівартість 1м³ (C)
        selling_prices - ціни реалізації залежно від запізнення [C1, C2, C3, C4, C5]
        volumes - можливі обсяги партій [A1, A2, A3, A4, A5]
        transport_costs - витрати на доставку [H1, H2, H3]
        penalty - штраф за прострочений день (B)
        probabilities - ймовірності запізнень [p1, p2, p3, p4, p5]
        """
        self.distance = distance
        self.cost_price = cost_price
        self.selling_prices = selling_prices
        self.volumes = volumes
        self.transport_costs = transport_costs
        self.penalty = penalty

        strategies = [f"Відправити {v:.0f} м³" for v in volumes]

        states = [f"Запізнення {i} дн." for i in range(5)]

        payoff_matrix = self._calculate_payoffs()

        super().__init__(strategies, states, payoff_matrix, probabilities)

    def _get_transport_cost(self, volume: float) -> float:
        """Визначити вартість транспортування залежно від обсягу"""
        if volume == self.volumes[0]:  # A1
            return self.transport_costs[0]  # H1
        elif volume in [self.volumes[1], self.volumes[2], self.volumes[3]]:  # A2, A3, A4
            return self.transport_costs[1]  # H2
        else:  # A5
            return self.transport_costs[2]  # H3

    def _calculate_payoffs(self) -> List[List[float]]:
        """
        Обчислення прибутків для кожної комбінації обсягу та запізнення

        Прибуток = Дохід - Витрати - Штрафи
        Дохід = обсяг * ціна_залежно_від_запізнення
        Витрати = обсяг * собівартість + відстань * вартість_км
        Штрафи = дні_запізнення * штраф_за_день
        """
        payoff_matrix = []

        for volume in self.volumes:
            row = []

            transport_cost_per_km = self._get_transport_cost(volume)

            for delay_days in range(5):  # 0, 1, 2, 3, 4 дні запізнення
                revenue = volume * self.selling_prices[delay_days]

                purchase_cost = volume * self.cost_price

                transport_cost = self.distance * transport_cost_per_km

                penalty_cost = delay_days * self.penalty

                profit = revenue - purchase_cost - transport_cost - penalty_cost

                row.append(profit)

            payoff_matrix.append(row)

        return payoff_matrix

    def print_detailed_analysis(self):
        """Детальний аналіз задачі"""
        print("\n" + "=" * 100)
        print("ЗАВДАННЯ №2: ПОСТАЧАННЯ ЛІСУ")
        print("=" * 100)

        print("\n📋 ВХІДНІ ДАНІ:")
        print(f"   Довжина маршруту: {self.distance} км")
        print(f"   Собівартість 1м³ лісу: {self.cost_price} грн")
        print(f"   Штраф за прострочений день: {self.penalty} грн")

        print("\n   Ціни реалізації залежно від запізнення:")
        for i, price in enumerate(self.selling_prices):
            print(f"      {i} днів запізнення: {price} грн/м³")

        print("\n   Можливі обсяги партій:")
        for i, volume in enumerate(self.volumes):
            transport_cost = self._get_transport_cost(volume)
            print(f"      {volume:.0f} м³: {transport_cost} грн/км")

        print(f"\n   Ймовірності запізнень: {self.probabilities}")

        print("\n💡 ЛОГІКА ЗАДАЧІ:")
        print("   • Більший обсяг → вища вартість транспортування")
        print("   • Запізнення → нижча ціна реалізації + штраф")
        print("   • Потрібно збалансувати обсяг та ризик запізнення")


def main():
    print("=" * 100)
    print(" " * 30 + "ЛАБОРАТОРНА РОБОТА №7")
    print(" " * 20 + "ПРИЙНЯТТЯ РІШЕНЬ В УМОВАХ НЕВИЗНАЧЕНОСТІ ТА РИЗИКУ")
    print(" " * 40 + "ВАРІАНТ 4")
    print("=" * 100)

    # Завдання №1
    A1 = 260  # Витрати на виробництво
    A2 = 500  # Ціна продажу
    C = 6  # Витрати на транспортування
    B = [55, 65, 75, 85, 95]  # Можливі рівні попиту
    P = [0.15, 0.2, 0.3, 0.2, 0.15]  # Ймовірності попиту

    # Завдання №2
    D = 480  # Довжина маршруту
    C_cost = 120  # Собівартість
    C_prices = [220, 200, 190, 170, 160]  # Ціни реалізації
    A_volumes = [12, 16, 20, 24, 28]  # Обсяги партій
    H_costs = [0.8, 1.0, 1.5]  # Вартості транспортування
    B_penalty = 65  # Штраф за день
    p_delays = [0.4, 0.3, 0.1, 0.1, 0.1]  # Ймовірності запізнень

    # ========================================================================
    # ЗАВДАННЯ №1: ВИРОБНИЦТВО ПРОДУКЦІЇ
    # ========================================================================

    print("\n\n")
    print("╔" + "=" * 98 + "╗")
    print("║" + " " * 30 + "ЗАВДАННЯ №1: ВИРОБНИЦТВО ПРОДУКЦІЇ" + " " * 34 + "║")
    print("╚" + "=" * 98 + "╝")

    task1 = ProductionDecision(
        production_cost=A1,
        transport_cost=C,
        selling_price=A2,
        demand_levels=B,
        probabilities=P
    )

    task1.print_detailed_analysis()
    task1.print_payoff_matrix()
    task1.print_risk_matrix()

    # Аналіз за критеріями
    results1 = task1.analyze_all_criteria(alpha_hurwicz=0.6, lambda_hl=0.6)

    # Додатковий аналіз з різними параметрами
    print("\n" + "=" * 100)
    print("📊 АНАЛІЗ ЧУТЛИВОСТІ КРИТЕРІЮ ГУРВІЦА")
    print("=" * 100)

    print(f"\n{'α (песимізм)':<15} {'Оптимальна стратегія':<40} {'Оцінка':<15}")
    print("-" * 100)

    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        idx, val = task1.hurwicz_criterion(alpha)
        print(f"{alpha:<15.1f} {task1.strategies[idx]:<40} {val:<15.2f}")

    # ========================================================================
    # ЗАВДАННЯ №2: ПОСТАЧАННЯ ЛІСУ
    # ========================================================================

    print("\n\n")
    print("╔" + "=" * 98 + "╗")
    print("║" + " " * 35 + "ЗАВДАННЯ №2: ПОСТАЧАННЯ ЛІСУ" + " " * 35 + "║")
    print("╚" + "=" * 98 + "╝")

    task2 = LogisticsDecision(
        distance=D,
        cost_price=C_cost,
        selling_prices=C_prices,
        volumes=A_volumes,
        transport_costs=H_costs,
        penalty=B_penalty,
        probabilities=p_delays
    )

    task2.print_detailed_analysis()
    task2.print_payoff_matrix()
    task2.print_risk_matrix()

    # Аналіз за критеріями
    results2 = task2.analyze_all_criteria(alpha_hurwicz=0.6, lambda_hl=0.6)

    # Додатковий аналіз
    print("\n" + "=" * 100)
    print("📊 АНАЛІЗ ЧУТЛИВОСТІ КРИТЕРІЮ ГУРВІЦА")
    print("=" * 100)

    print(f"\n{'α (песимізм)':<15} {'Оптимальна стратегія':<40} {'Оцінка':<15}")
    print("-" * 100)

    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        idx, val = task2.hurwicz_criterion(alpha)
        print(f"{alpha:<15.1f} {task2.strategies[idx]:<40} {val:<15.2f}")

    print("\n\n")
    print("=" * 100)
    print("📈 ЗАГАЛЬНІ ВИСНОВКИ ТА РЕКОМЕНДАЦІЇ")
    print("=" * 100)

    print("""ЗАВДАННЯ №1 (Виробництво продукції):""")

    counter1 = Counter(results1.values())
    most_common1 = counter1.most_common(1)[0]

    print(f"   • Найчастіша рекомендація: {task1.strategies[most_common1[0]]}")
    print(f"   • Підтримується {most_common1[1]} критеріями")

    if len(counter1) == 1:
        print("   ✅ Всі критерії узгоджені - рішення стабільне")
    else:
        print("   ⚠️  Критерії дають різні результати")
        print("   Рекомендується:")
        print("      - Якщо є надійні статистичні дані → критерій Байєса-Лапласа")
        print("      - Якщо важлива обережність → критерій Вальда")
        print("      - Для збалансованого підходу → критерій Гурвіца (α=0.6)")

    print("\nЗАВДАННЯ №2 (Постачання лісу):")

    counter2 = Counter(results2.values())
    most_common2 = counter2.most_common(1)[0]

    print(f"   • Найчастіша рекомендація: {task2.strategies[most_common2[0]]}")
    print(f"   • Підтримується {most_common2[1]} критеріями")

    if len(counter2) == 1:
        print("   ✅ Всі критерії узгоджені - рішення стабільне")
    else:
        print("   ⚠️  Критерії дають різні результати")
        print("   Рекомендується:")
        print("      - Використати критерій Байєса-Лапласа (є дані про ймовірності)")



if __name__ == "__main__":
    main()
