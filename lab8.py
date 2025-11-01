import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from typing import List, Tuple, Dict


class DecisionNode:
    """Вузол рішення (квадрат)"""

    def __init__(self, name: str, x: float, y: float):
        self.name = name
        self.x = x
        self.y = y
        self.children = []
        self.expected_value = 0

    def add_child(self, child, label: str = "", probability: float = None, payoff: float = None):
        """Додати дочірній вузол"""
        self.children.append({
            'node': child,
            'label': label,
            'probability': probability,
            'payoff': payoff
        })


class ChanceNode:
    """Вузол випадковості (коло)"""

    def __init__(self, name: str, x: float, y: float):
        self.name = name
        self.x = x
        self.y = y
        self.children = []
        self.expected_value = 0

    def add_child(self, child, label: str, probability: float, payoff: float = None):
        """Додати дочірній вузол"""
        self.children.append({
            'node': child,
            'label': label,
            'probability': probability,
            'payoff': payoff
        })


class EndNode:
    """Кінцевий вузол (трикутник)"""

    def __init__(self, name: str, x: float, y: float, payoff: float):
        self.name = name
        self.x = x
        self.y = y
        self.payoff = payoff
        self.expected_value = payoff


# ============================================================================
# КЛАС ДЛЯ ПОБУДОВИ ТА АНАЛІЗУ ДЕРЕВА РІШЕНЬ
# ============================================================================

class DecisionTree:
    """Клас для побудови та аналізу дерева рішень"""

    def __init__(self, title: str):
        self.title = title
        self.root = None
        self.all_nodes = []

    def calculate_expected_values(self, node):
        """Рекурсивний розрахунок очікуваних значень (зворотній хід)"""
        if isinstance(node, EndNode):
            return node.payoff

        if isinstance(node, ChanceNode):
            # Для вузла випадковості: EV = Σ(p_i * payoff_i)
            expected_value = 0
            for child_info in node.children:
                child_node = child_info['node']
                probability = child_info['probability']

                # Рекурсивно обчислюємо EV для дочірнього вузла
                child_ev = self.calculate_expected_values(child_node)

                # Додаємо виграш на ребрі (якщо є)
                if child_info['payoff'] is not None:
                    child_ev += child_info['payoff']

                expected_value += probability * child_ev

            node.expected_value = expected_value
            return expected_value

        if isinstance(node, DecisionNode):
            # Для вузла рішення: вибираємо максимальне EV
            max_ev = float('-inf')

            for child_info in node.children:
                child_node = child_info['node']

                # Рекурсивно обчислюємо EV для дочірнього вузла
                child_ev = self.calculate_expected_values(child_node)

                # Додаємо виграш на ребрі (якщо є)
                if child_info['payoff'] is not None:
                    child_ev += child_info['payoff']

                max_ev = max(max_ev, child_ev)

            node.expected_value = max_ev
            return max_ev

    def find_optimal_path(self, node, path=[]):
        """Знайти оптимальний шлях у дереві"""
        if isinstance(node, EndNode):
            return path + [node]

        if isinstance(node, ChanceNode):
            # Для вузла випадковості просто проходимо всі гілки
            paths = []
            for child_info in node.children:
                child_path = self.find_optimal_path(
                    child_info['node'],
                    path + [{'node': node, 'choice': child_info['label']}]
                )
                paths.append(child_path)
            return paths

        if isinstance(node, DecisionNode):
            # Для вузла рішення вибираємо найкращу гілку
            best_child = None
            best_ev = float('-inf')

            for child_info in node.children:
                child_node = child_info['node']
                child_ev = child_node.expected_value

                if child_info['payoff'] is not None:
                    child_ev += child_info['payoff']

                if child_ev > best_ev:
                    best_ev = child_ev
                    best_child = child_info

            return self.find_optimal_path(
                best_child['node'],
                path + [{'node': node, 'choice': best_child['label']}]
            )

    def draw_tree(self, filename: str = None):
        """Намалювати дерево рішень"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 10.5)
        ax.axis('off')

        # Заголовок
        ax.text(5, 10, self.title, fontsize=14, weight='bold', ha='center')

        # Малюємо вузли та ребра
        self._draw_node(ax, self.root)

        # Легенда
        legend_elements = [
            mpatches.Rectangle((0, 0), 1, 1, fc='lightblue', label='Вузол рішення'),
            mpatches.Circle((0, 0), 0.5, fc='lightgreen', label='Вузол випадковості'),
            mpatches.Polygon([[0, 0], [1, 0], [0.5, 1]], fc='lightyellow', label='Кінцевий результат')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Дерево збережено у файл: {filename}")
            plt.close(fig)  # Закриваємо фігуру після збереження
        else:
            plt.show()

    def _draw_node(self, ax, node, parent_x=None, parent_y=None, label="", probability=None):
        """Рекурсивно малювати вузол та його дочірні вузли"""

        # Визначаємо точку з'єднання на контурі вузла
        connection_x = node.x
        connection_y = node.y

        if parent_x is not None:
            dx = node.x - parent_x
            dy = node.y - parent_y
            dist = np.sqrt(dx ** 2 + dy ** 2)

            if dist > 0:
                # Нормалізуємо вектор напрямку
                dx_norm = dx / dist
                dy_norm = dy / dist

                # Визначаємо відступ залежно від типу вузла
                if isinstance(node, DecisionNode):
                    offset = 0.35  # Відступ для квадрата
                    connection_x = node.x - dx_norm * offset
                    connection_y = node.y - dy_norm * offset
                elif isinstance(node, ChanceNode):
                    offset = 0.25  # Радіус кола
                    connection_x = node.x - dx_norm * offset
                    connection_y = node.y - dy_norm * offset
                elif isinstance(node, EndNode):
                    offset = 0.3  # Відступ для трикутника
                    connection_x = node.x - dx_norm * offset
                    connection_y = node.y - dy_norm * offset

            # Малюємо ребро від батька до контуру вузла
            ax.plot([parent_x, connection_x], [parent_y, connection_y], 'k-', linewidth=2)

            # Підпис ребра
            mid_x = (parent_x + connection_x) / 2
            mid_y = (parent_y + connection_y) / 2

            if probability is not None:
                label_text = f"{label}\\np={probability:.2f}"
            else:
                label_text = label

            ax.text(mid_x, mid_y + 0.2, label_text, fontsize=9,
                    ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Малюємо вузол
        if isinstance(node, DecisionNode):
            # Квадрат для рішення
            rect = FancyBboxPatch((node.x - 0.3, node.y - 0.2), 0.6, 0.4,
                                  boxstyle="round,pad=0.05",
                                  facecolor='lightblue', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(node.x, node.y, node.name, fontsize=10, ha='center', va='center', weight='bold')

            # EV під вузлом
            ax.text(node.x, node.y - 0.4, f'EV={node.expected_value:.0f}',
                    fontsize=9, ha='center', style='italic', color='blue')

        elif isinstance(node, ChanceNode):
            # Коло для випадковості
            circle = plt.Circle((node.x, node.y), 0.25, facecolor='lightgreen',
                                edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(node.x, node.y, node.name, fontsize=10, ha='center', va='center', weight='bold')

            # EV під вузлом
            ax.text(node.x, node.y - 0.5, f'EV={node.expected_value:.0f}',
                    fontsize=9, ha='center', style='italic', color='green')

        elif isinstance(node, EndNode):
            # Трикутник для кінцевого результату
            triangle = mpatches.Polygon([[node.x, node.y + 0.3],
                                         [node.x - 0.25, node.y - 0.2],
                                         [node.x + 0.25, node.y - 0.2]],
                                        facecolor='lightyellow', edgecolor='black', linewidth=2)
            ax.add_patch(triangle)
            ax.text(node.x, node.y, node.name, fontsize=9, ha='center', va='center')

            # Виграш під вузлом
            ax.text(node.x, node.y - 0.5, f'{node.payoff:.0f}',
                    fontsize=10, ha='center', weight='bold', color='black')

        # Рекурсивно малюємо дочірні вузли
        if hasattr(node, 'children'):
            for child_info in node.children:
                # Визначаємо початкову точку на контурі поточного вузла
                child_x = child_info['node'].x
                child_y = child_info['node'].y

                dx = child_x - node.x
                dy = child_y - node.y
                dist = np.sqrt(dx ** 2 + dy ** 2)

                start_x = node.x
                start_y = node.y

                if dist > 0:
                    dx_norm = dx / dist
                    dy_norm = dy / dist

                    # Визначаємо відступ для початкової точки
                    if isinstance(node, DecisionNode):
                        offset = 0.35
                        start_x = node.x + dx_norm * offset
                        start_y = node.y + dy_norm * offset
                    elif isinstance(node, ChanceNode):
                        offset = 0.25
                        start_x = node.x + dx_norm * offset
                        start_y = node.y + dy_norm * offset

                self._draw_node(ax, child_info['node'], start_x, start_y,
                                child_info['label'], child_info.get('probability'))



def task1_create_production_tree(A1: float, A2: float, A3: float,
                                 B1: float, B2: float, B3: float):
    """
    Побудова дерева рішень для вибору типу виробництва

    A1, A2, A3 - виграші при сприятливих умовах
    B1, B2, B3 - виграші при несприятливих умовах
    """
    print("\n" + "=" * 100)
    print("ЗАВДАННЯ №1: ВИБІР ТИПУ ВИРОБНИЦТВА")
    print("=" * 100)

    print("\n📋 ВХІДНІ ДАНІ:")
    print(f"   Сприятливі умови (p=0.5):")
    print(f"      Велике виробництво: {A1:,.0f} г.о.")
    print(f"      Мале підприємство: {A2:,.0f} г.о.")
    print(f"      Продаж патенту:     {A3:,.0f} г.о.")

    print(f"\n   Несприятливі умови (p=0.5):")
    print(f"      Велике виробництво: {B1:,.0f} г.о.")
    print(f"      Мале підприємство:  {B2:,.0f} г.о.")
    print(f"      Продаж патенту:     {B3:,.0f} г.о.")

    # Створюємо дерево
    tree = DecisionTree("Завдання 1: Вибір типу виробництва")

    # Кореневий вузол - рішення
    root = DecisionNode("Рішення", 1, 5)
    tree.root = root

    # Вузли випадковості для кожного рішення - збільшено відстань
    chance1 = ChanceNode("Ринок", 4, 8)
    chance2 = ChanceNode("Ринок", 4, 5)
    chance3 = ChanceNode("Ринок", 4, 2)

    # Кінцеві вузли - збільшено вертикальну відстань
    end11 = EndNode("Спр.", 7, 9, A1)
    end12 = EndNode("Неспр.", 7, 7, B1)

    end21 = EndNode("Спр.", 7, 6, A2)
    end22 = EndNode("Неспр.", 7, 4, B2)

    end31 = EndNode("Спр.", 7, 3, A3)
    end32 = EndNode("Неспр.", 7, 1, B3)

    # Будуємо зв'язки
    root.add_child(chance1, "Велике виробництво")
    root.add_child(chance2, "Мале підприємство")
    root.add_child(chance3, "Продати патент")

    chance1.add_child(end11, "Сприятливі", 0.5)
    chance1.add_child(end12, "Несприятливі", 0.5)

    chance2.add_child(end21, "Сприятливі", 0.5)
    chance2.add_child(end22, "Несприятливі", 0.5)

    chance3.add_child(end31, "Сприятливі", 0.5)
    chance3.add_child(end32, "Несприятливі", 0.5)

    # Розрахунок очікуваних значень
    tree.calculate_expected_values(root)

    # Аналіз результатів
    print("\n" + "=" * 100)
    print("📊 АНАЛІЗ ОЧІКУВАНИХ ЗНАЧЕНЬ:")
    print("=" * 100)

    ev1 = 0.5 * A1 + 0.5 * B1
    ev2 = 0.5 * A2 + 0.5 * B2
    ev3 = 0.5 * A3 + 0.5 * B3

    print(f"\n1️⃣  Велике виробництво:")
    print(f"   EV = 0.5 × {A1:,.0f} + 0.5 × {B1:,.0f} = {ev1:,.0f} г.о.")

    print(f"\n2️⃣  Мале підприємство:")
    print(f"   EV = 0.5 × {A2:,.0f} + 0.5 × {B2:,.0f} = {ev2:,.0f} г.о.")

    print(f"\n3️⃣  Продаж патенту:")
    print(f"   EV = 0.5 × {A3:,.0f} + 0.5 × {B3:,.0f} = {ev3:,.0f} г.о.")

    # Визначаємо оптимальне рішення
    decisions = [
        ("Велике виробництво", ev1),
        ("Мале підприємство", ev2),
        ("Продаж патенту", ev3)
    ]

    best_decision = max(decisions, key=lambda x: x[1])

    print("\n" + "=" * 100)
    print("✅ ОПТИМАЛЬНЕ РІШЕННЯ:")
    print("=" * 100)
    print(f"\n   {best_decision[0]}")
    print(f"   Очікуване значення: {best_decision[1]:,.0f} г.о.")

    # Малюємо дерево
    tree.draw_tree("task1_decision_tree.png")

    return tree, best_decision


# ============================================================================
# ЗАВДАННЯ №2: ДОСЛІДЖЕННЯ РИНКУ
# ============================================================================

def task2_market_research_tree(A1: float, A2: float, A3: float,
                               B1: float, B2: float, B3: float,
                               P11: float, P12: float, P21: float, P22: float,
                               Q: float):
    """
    Побудова дерева рішень з урахуванням дослідження ринку

    P11 - ймовірність сприятливого прогнозу при сприятливому факті
    P12 - ймовірність сприятливого прогнозу при несприятливому факті
    P21 - ймовірність несприятливого прогнозу при сприятливому факті
    P22 - ймовірність несприятливого прогнозу при несприятливому факті
    Q - вартість консалтингу
    """
    print("\n" + "=" * 100)
    print("ЗАВДАННЯ №2: ДОСЛІДЖЕННЯ РИНКУ")
    print("=" * 100)

    print("\n📋 ВХІДНІ ДАНІ:")
    print(f"   Вартість консалтингу: {Q:,.0f} г.о.")

    print(f"\n   Матриця умовних ймовірностей:")
    print(f"      {'':20} {'Прогноз Спр.':<15} {'Прогноз Неспр.':<15}")
    print(f"      {'Факт Спр.':<20} {P11:<15.2f} {P21:<15.2f}")
    print(f"      {'Факт Неспр.':<20} {P12:<15.2f} {P22:<15.2f}")

    # Апріорні ймовірності
    p_favorable = 0.5
    p_unfavorable = 0.5

    # КРОК 1: Обчислюємо ймовірності прогнозів за теоремою повної ймовірності
    # P(Прогноз Спр.) = P(Прогноз Спр. | Факт Спр.) × P(Факт Спр.) +
    #                   P(Прогноз Спр. | Факт Неспр.) × P(Факт Неспр.)
    p_prog_fav = P11 * p_favorable + P12 * p_unfavorable
    p_prog_unfav = P21 * p_favorable + P22 * p_unfavorable

    print(f"\n📊 КРОК 1: Ймовірності прогнозів (теорема повної ймовірності):")
    print(f"   P(Прогноз Спр.) = {P11}×{p_favorable} + {P12}×{p_unfavorable} = {p_prog_fav:.3f}")
    print(f"   P(Прогноз Неспр.) = {P21}×{p_favorable} + {P22}×{p_unfavorable} = {p_prog_unfav:.3f}")

    # КРОК 2: Обчислюємо апостеріорні ймовірності за теоремою Байєса
    # P(Факт | Прогноз) = P(Прогноз | Факт) × P(Факт) / P(Прогноз)

    # При сприятливому прогнозі:
    p_fav_given_prog_fav = (P11 * p_favorable) / p_prog_fav
    p_unfav_given_prog_fav = (P12 * p_unfavorable) / p_prog_fav

    # При несприятливому прогнозі:
    p_fav_given_prog_unfav = (P21 * p_favorable) / p_prog_unfav
    p_unfav_given_prog_unfav = (P22 * p_unfavorable) / p_prog_unfav

    print(f"\n📊 КРОК 2: Апостеріорні ймовірності (теорема Байєса):")
    print(f"   При сприятливому прогнозі:")
    print(f"      P(Факт Спр. | Прогноз Спр.) = ({P11}×{p_favorable})/{p_prog_fav:.3f} = {p_fav_given_prog_fav:.3f}")
    print(f"      P(Факт Неспр. | Прогноз Спр.) = ({P12}×{p_unfavorable})/{p_prog_fav:.3f} = {p_unfav_given_prog_fav:.3f}")
    print(f"   При несприятливому прогнозі:")
    print(f"      P(Факт Спр. | Прогноз Неспр.) = ({P21}×{p_favorable})/{p_prog_unfav:.3f} = {p_fav_given_prog_unfav:.3f}")
    print(f"      P(Факт Неспр. | Прогноз Неспр.) = ({P22}×{p_unfavorable})/{p_prog_unfav:.3f} = {p_unfav_given_prog_unfav:.3f}")

    # КРОК 3: Розрахунок EV без дослідження
    print(f"\n📊 КРОК 3: Очікувані значення БЕЗ дослідження:")

    ev1_no_research = 0.5 * A1 + 0.5 * B1
    ev2_no_research = 0.5 * A2 + 0.5 * B2
    ev3_no_research = 0.5 * A3 + 0.5 * B3

    print(f"   Велике виробництво: EV = 0.5×{A1:,.0f} + 0.5×{B1:,.0f} = {ev1_no_research:,.0f} г.о.")
    print(f"   Мале підприємство:  EV = 0.5×{A2:,.0f} + 0.5×{B2:,.0f} = {ev2_no_research:,.0f} г.о.")
    print(f"   Продаж патенту:     EV = 0.5×{A3:,.0f} + 0.5×{B3:,.0f} = {ev3_no_research:,.0f} г.о.")

    ev_no_research = max(ev1_no_research, ev2_no_research, ev3_no_research)
    print(f"   ➜ Оптимальне рішення без дослідження: EV = {ev_no_research:,.0f} г.о.")

    # КРОК 4: Розрахунок EV З дослідженням
    print(f"\n📊 КРОК 4: Очікувані значення З дослідженням:")

    # При сприятливому прогнозі
    print(f"\n   4.1. При сприятливому прогнозі (p={p_prog_fav:.3f}):")
    ev1_prog_fav = p_fav_given_prog_fav * A1 + p_unfav_given_prog_fav * B1
    ev2_prog_fav = p_fav_given_prog_fav * A2 + p_unfav_given_prog_fav * B2
    ev3_prog_fav = p_fav_given_prog_fav * A3 + p_unfav_given_prog_fav * B3

    print(f"      Велике: EV = {p_fav_given_prog_fav:.3f}×{A1:,.0f} + {p_unfav_given_prog_fav:.3f}×{B1:,.0f} = {ev1_prog_fav:,.0f} г.о.")
    print(f"      Мале:   EV = {p_fav_given_prog_fav:.3f}×{A2:,.0f} + {p_unfav_given_prog_fav:.3f}×{B2:,.0f} = {ev2_prog_fav:,.0f} г.о.")
    print(f"      Патент: EV = {p_fav_given_prog_fav:.3f}×{A3:,.0f} + {p_unfav_given_prog_fav:.3f}×{B3:,.0f} = {ev3_prog_fav:,.0f} г.о.")

    ev_prog_fav = max(ev1_prog_fav, ev2_prog_fav, ev3_prog_fav)
    print(f"      ➜ Оптимальне рішення: EV = {ev_prog_fav:,.0f} г.о.")

    # При несприятливому прогнозі
    print(f"\n   4.2. При несприятливому прогнозі (p={p_prog_unfav:.3f}):")
    ev1_prog_unfav = p_fav_given_prog_unfav * A1 + p_unfav_given_prog_unfav * B1
    ev2_prog_unfav = p_fav_given_prog_unfav * A2 + p_unfav_given_prog_unfav * B2
    ev3_prog_unfav = p_fav_given_prog_unfav * A3 + p_unfav_given_prog_unfav * B3

    print(f"      Велике: EV = {p_fav_given_prog_unfav:.3f}×{A1:,.0f} + {p_unfav_given_prog_unfav:.3f}×{B1:,.0f} = {ev1_prog_unfav:,.0f} г.о.")
    print(f"      Мале:   EV = {p_fav_given_prog_unfav:.3f}×{A2:,.0f} + {p_unfav_given_prog_unfav:.3f}×{B2:,.0f} = {ev2_prog_unfav:,.0f} г.о.")
    print(f"      Патент: EV = {p_fav_given_prog_unfav:.3f}×{A3:,.0f} + {p_unfav_given_prog_unfav:.3f}×{B3:,.0f} = {ev3_prog_unfav:,.0f} г.о.")

    ev_prog_unfav = max(ev1_prog_unfav, ev2_prog_unfav, ev3_prog_unfav)
    print(f"      ➜ Оптимальне рішення: EV = {ev_prog_unfav:,.0f} г.о.")

    # Загальне EV з дослідженням (до вирахування вартості)
    ev_with_research_before_cost = p_prog_fav * ev_prog_fav + p_prog_unfav * ev_prog_unfav
    print(f"\n   4.3. Загальне EV з дослідженням (до вирахування вартості):")
    print(f"      EV = {p_prog_fav:.3f}×{ev_prog_fav:,.0f} + {p_prog_unfav:.3f}×{ev_prog_unfav:,.0f}")
    print(f"      EV = {ev_with_research_before_cost:,.0f} г.о.")

    # З врахуванням вартості дослідження
    ev_with_research = ev_with_research_before_cost - Q
    print(f"\n   4.4. EV з дослідженням (після вирахування вартості {Q:,.0f} г.о.):")
    print(f"      EV = {ev_with_research_before_cost:,.0f} - {Q:,.0f} = {ev_with_research:,.0f} г.о.")

    # КРОК 5: Цінність інформації
    evpi = ev_with_research - ev_no_research

    print("\n" + "=" * 100)
    print("💡 КРОК 5: ЦІННІСТЬ ДОДАТКОВОЇ ІНФОРМАЦІЇ (EVPI)")
    print("=" * 100)
    print(f"\n   EVPI = EV(з дослідженням) - EV(без дослідження)")
    print(f"   EVPI = {ev_with_research:,.0f} - {ev_no_research:,.0f} = {evpi:,.0f} г.о.")

    if evpi > 0:
        print(f"\n   ✅ Дослідження ринку ВАРТЕ того!")
        print(f"   Додатковий прибуток: {evpi:,.0f} г.о.")
        print(f"   Рентабельність: {(evpi/Q)*100:.1f}% від вартості дослідження")
    else:
        print(f"\n   ❌ Дослідження ринку НЕ ВАРТЕ того!")
        print(f"   Втрата: {abs(evpi):,.0f} г.о.")

    # Створюємо дерево (спрощена версія для візуалізації)
    tree = DecisionTree("Завдання 2: Дослідження ринку")

    root = DecisionNode("Досліджувати?", 0.5, 5)
    tree.root = root

    # Гілка "Не досліджувати"
    decision_no_research = DecisionNode("Рішення", 3, 8)
    chance_no1 = ChanceNode("Ринок", 5.5, 9.5)
    chance_no2 = ChanceNode("Ринок", 5.5, 8)
    chance_no3 = ChanceNode("Ринок", 5.5, 6.5)

    end_no11 = EndNode("С", 8, 10, A1)
    end_no12 = EndNode("Н", 8, 9, B1)
    end_no21 = EndNode("С", 8, 8.5, A2)
    end_no22 = EndNode("Н", 8, 7.5, B2)
    end_no31 = EndNode("С", 8, 7, A3)
    end_no32 = EndNode("Н", 8, 6, B3)

    root.add_child(decision_no_research, "Не досліджувати")
    decision_no_research.add_child(chance_no1, "Велике")
    decision_no_research.add_child(chance_no2, "Мале")
    decision_no_research.add_child(chance_no3, "Патент")

    chance_no1.add_child(end_no11, "Спр.", 0.5)
    chance_no1.add_child(end_no12, "Неспр.", 0.5)
    chance_no2.add_child(end_no21, "Спр.", 0.5)
    chance_no2.add_child(end_no22, "Неспр.", 0.5)
    chance_no3.add_child(end_no31, "Спр.", 0.5)
    chance_no3.add_child(end_no32, "Неспр.", 0.5)

    # Гілка "Досліджувати"
    chance_research = ChanceNode("Прогноз", 3, 2)
    root.add_child(chance_research, "Досліджувати", payoff=-Q)

    # При сприятливому прогнозі
    decision_prog_fav = DecisionNode("Рішення", 5.5, 4)
    chance_research.add_child(decision_prog_fav, "Спр. прогноз", p_prog_fav)

    chance_fav1 = ChanceNode("Ринок", 8, 5.5)
    chance_fav2 = ChanceNode("Ринок", 8, 4)
    chance_fav3 = ChanceNode("Ринок", 8, 2.5)

    decision_prog_fav.add_child(chance_fav1, "Велике")
    decision_prog_fav.add_child(chance_fav2, "Мале")
    decision_prog_fav.add_child(chance_fav3, "Патент")

    end_fav11 = EndNode("С", 10, 6, A1)
    end_fav12 = EndNode("Н", 10, 5, B1)
    end_fav21 = EndNode("С", 10, 4.5, A2)
    end_fav22 = EndNode("Н", 10, 3.5, B2)
    end_fav31 = EndNode("С", 10, 3, A3)
    end_fav32 = EndNode("Н", 10, 2, B3)

    chance_fav1.add_child(end_fav11, "С", p_fav_given_prog_fav)
    chance_fav1.add_child(end_fav12, "Н", p_unfav_given_prog_fav)
    chance_fav2.add_child(end_fav21, "С", p_fav_given_prog_fav)
    chance_fav2.add_child(end_fav22, "Н", p_unfav_given_prog_fav)
    chance_fav3.add_child(end_fav31, "С", p_fav_given_prog_fav)
    chance_fav3.add_child(end_fav32, "Н", p_unfav_given_prog_fav)

    # При несприятливому прогнозі
    decision_prog_unfav = DecisionNode("Рішення", 5.5, 0.5)
    chance_research.add_child(decision_prog_unfav, "Неспр. прогноз", p_prog_unfav)

    chance_unfav1 = ChanceNode("Ринок", 8, 1.5)
    chance_unfav2 = ChanceNode("Ринок", 8, 0)
    chance_unfav3 = ChanceNode("Ринок", 8, -1.5)

    decision_prog_unfav.add_child(chance_unfav1, "Велике")
    decision_prog_unfav.add_child(chance_unfav2, "Мале")
    decision_prog_unfav.add_child(chance_unfav3, "Патент")

    end_unfav11 = EndNode("С", 10, 2, A1)
    end_unfav12 = EndNode("Н", 10, 1, B1)
    end_unfav21 = EndNode("С", 10, 0.5, A2)
    end_unfav22 = EndNode("Н", 10, -0.5, B2)
    end_unfav31 = EndNode("С", 10, -1, A3)
    end_unfav32 = EndNode("Н", 10, -2, B3)

    chance_unfav1.add_child(end_unfav11, "С", p_fav_given_prog_unfav)
    chance_unfav1.add_child(end_unfav12, "Н", p_unfav_given_prog_unfav)
    chance_unfav2.add_child(end_unfav21, "С", p_fav_given_prog_unfav)
    chance_unfav2.add_child(end_unfav22, "Н", p_unfav_given_prog_unfav)
    chance_unfav3.add_child(end_unfav31, "С", p_fav_given_prog_unfav)
    chance_unfav3.add_child(end_unfav32, "Н", p_unfav_given_prog_unfav)

    # Розрахунок очікуваних значень для дерева
    tree.calculate_expected_values(root)

    print("\n" + "=" * 100)
    print("✅ ОПТИМАЛЬНЕ РІШЕННЯ:")
    print("=" * 100)

    if ev_with_research > ev_no_research:
        print(f"\n   Провести дослідження ринку")
        print(f"   Очікуване значення: {ev_with_research:,.0f} г.о.")
        print(f"   Стратегія:")
        print(f"      • При сприятливому прогнозі → оптимальне рішення з EV = {ev_prog_fav:,.0f} г.о.")
        print(f"      • При несприятливому прогнозі → оптимальне рішення з EV = {ev_prog_unfav:,.0f} г.о.")
    else:
        print(f"\n   Не проводити дослідження ринку")
        print(f"   Очікуване значення: {ev_no_research:,.0f} г.о.")

    tree.draw_tree("task2_decision_tree.png")

    return tree, evpi




def task3_supplier_selection_tree(prob_A: List[float], prob_B: List[float],
                                      K: float, N: int, L: float):
        """
        Побудова дерева рішень для вибору постачальника

        prob_A, prob_B - ймовірності відсотків браку для постачальників
        K - витрати на усунення браку одного виробу
        N - кількість виробів у партії
        L - знижка від постачальника B
        """
        print("\n" + "=" * 100)
        print("ЗАВДАННЯ №3: ВИБІР ПОСТАЧАЛЬНИКА")
        print("=" * 100)

        print("\n📋 ВХІДНІ ДАНІ:")
        print(f"   Розмір партії: {N:,} шт.")
        print(f"   Витрати на усунення браку: {K:,.0f} г.о./шт.")
        print(f"   Знижка від постачальника B: {L:,.0f} г.о.")

        print(f"\n   Ймовірності % браку:")
        print(f"      {'% браку':<10} {'Постачальник A':<20} {'Постачальник B':<20}")
        print(f"      {'-' * 50}")
        for i, (pa, pb) in enumerate(zip(prob_A, prob_B), 1):
            print(f"      {i}%{' ' * 8} {pa:<20.2f} {pb:<20.2f}")

        # Створюємо дерево
        tree = DecisionTree("Завдання 3: Вибір постачальника")

        # Кореневий вузол - вибір постачальника
        root = DecisionNode("Постачальник", 1, 5)
        tree.root = root

        # Вузли випадковості для кожного постачальника - більша відстань
        chance_A = ChanceNode("% браку", 4, 8)
        chance_B = ChanceNode("% браку", 4, 2)

        root.add_child(chance_A, "Постачальник A")
        root.add_child(chance_B, "Постачальник B", payoff=L)  # Знижка

        # Кінцеві вузли для постачальника A - збільшено вертикальний розкид
        y_positions_A = np.linspace(10, 6, len(prob_A))
        for i, (prob, y_pos) in enumerate(zip(prob_A, y_positions_A), 1):
            defect_rate = i / 100.0
            cost = -defect_rate * N * K  # Витрати (негативні)
            end_node = EndNode(f"{i}%", 7, y_pos, cost)
            chance_A.add_child(end_node, f"{i}% браку", prob)

        # Кінцеві вузли для постачальника B - збільшено вертикальний розкид
        y_positions_B = np.linspace(4, 0, len(prob_B))
        for i, (prob, y_pos) in enumerate(zip(prob_B, y_positions_B), 1):
            defect_rate = i / 100.0
            cost = L - defect_rate * N * K  # Знижка мінус витрати
            end_node = EndNode(f"{i}%", 7, y_pos, cost)
            chance_B.add_child(end_node, f"{i}% браку", prob)

        # Розрахунок очікуваних значень
        tree.calculate_expected_values(root)

        # Аналіз результатів
        print("\n" + "=" * 100)
        print("📊 АНАЛІЗ ОЧІКУВАНИХ ВИТРАТ:")
        print("=" * 100)

        # Постачальник A
        ev_A = 0
        print(f"\n1️⃣  Постачальник A:")
        for i, prob in enumerate(prob_A, 1):
            defect_rate = i / 100.0
            cost = -defect_rate * N * K
            ev_A += prob * cost
            print(f"   {i}% браку (p={prob:.2f}): витрати = {cost:,.0f} г.о.")
        print(f"   Очікувані витрати: {ev_A:,.0f} г.о.")

        # Постачальник B
        ev_B = L
        print(f"\n2️⃣  Постачальник B:")
        print(f"   Знижка: +{L:,.0f} г.о.")
        for i, prob in enumerate(prob_B, 1):
            defect_rate = i / 100.0
            cost = -defect_rate * N * K
            ev_B += prob * cost
            print(f"   {i}% браку (p={prob:.2f}): витрати = {cost:,.0f} г.о.")
        print(f"   Очікувані витрати (з знижкою): {ev_B:,.0f} г.о.")

        # Порівняння
        print("\n" + "=" * 100)
        print("✅ ОПТИМАЛЬНЕ РІШЕННЯ:")
        print("=" * 100)

        diff = ev_B - ev_A

        if ev_B > ev_A:
            print(f"\n   Вибрати постачальника B")
            print(f"   Переваги: {diff:,.0f} г.о.")
            print(f"   Очікуваний результат: {ev_B:,.0f} г.о.")
        else:
            print(f"\n   Вибрати постачальника A")
            print(f"   Переваги: {abs(diff):,.0f} г.о.")
            print(f"   Очікуваний результат: {ev_A:,.0f} г.о.")

        print(f"\n💡 АНАЛІЗ:")
        avg_defect_A = sum((i / 100.0) * prob for i, prob in enumerate(prob_A, 1))
        avg_defect_B = sum((i / 100.0) * prob for i, prob in enumerate(prob_B, 1))

        print(f"   Середній % браку:")
        print(f"      Постачальник A: {avg_defect_A * 100:.2f}%")
        print(f"      Постачальник B: {avg_defect_B * 100:.2f}%")

        break_even = L / (N * K * (avg_defect_B - avg_defect_A)) * 100
        if avg_defect_B > avg_defect_A:
            print(f"\n   Точка беззбитковості:")
            print(f"   Знижка має покривати різницю у витратах на брак")
            print(f"   Поточна різниця: {(avg_defect_B - avg_defect_A) * 100:.2f}%")

        tree.draw_tree("task3_decision_tree.png")

        return tree, ev_A, ev_B










def main():
    print("=" * 100)
    print(" " * 30 + "ЛАБОРАТОРНА РОБОТА №8")
    print(" " * 25 + "ДЕРЕВО РІШЕНЬ В УМОВАХ РИЗИКУ")
    print(" " * 40 + "ВАРІАНТ 4")
    print("=" * 100)

    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

    # Завдання 1 та 2
    A1 = 550000  # Велике виробництво, сприятливі умови
    A2 = 300000  # Мале підприємство, сприятливі умови
    A3 = 55000  # Продаж патенту, сприятливі умови
    B1 = -250000  # Велике виробництво, несприятливі умови
    B2 = -75000  # Мале підприємство, несприятливі умови
    B3 = 55000  # Продаж патенту, несприятливі умови

    # Завдання 2 - дослідження ринку
    P11 = 0.75  # P(Прогноз спр. | Факт спр.)
    P12 = 0.25  # P(Прогноз спр. | Факт неспр.)
    P21 = 0.3  # P(Прогноз неспр. | Факт спр.)
    P22 = 0.7  # P(Прогноз неспр. | Факт неспр.)
    Q = 10000  # Вартість консалтингу

    # Завдання 3 - вибір постачальника
    prob_A = [0.6, 0.3, 0.15, 0.15, 0.05]  # Ймовірності браку для A
    prob_B = [0.3, 0.25, 0.15, 0.1, 0.05]  # Ймовірності браку для B
    K = 140  # Витрати на усунення браку одного виробу
    N = 15000  # Кількість виробів у партії
    L = 1100  # Знижка від постачальника B

    print("\n\n")
    print("╔" + "=" * 98 + "╗")
    print("║" + " " * 30 + "ЗАВДАННЯ №1: ВИБІР ТИПУ ВИРОБНИЦТВА" + " " * 33 + "║")
    print("╚" + "=" * 98 + "╝")

    tree1, best_decision1 = task1_create_production_tree(A1, A2, A3, B1, B2, B3)

    print("\n\n")
    print("╔" + "=" * 98 + "╗")
    print("║" + " " * 35 + "ЗАВДАННЯ №2: ДОСЛІДЖЕННЯ РИНКУ" + " " * 33 + "║")
    print("╚" + "=" * 98 + "╝")

    tree2, evpi = task2_market_research_tree(A1, A2, A3, B1, B2, B3, P11, P12, P21, P22, Q)


    print("\n\n")
    print("╔" + "=" * 98 + "╗")
    print("║" + " " * 35 + "ЗАВДАННЯ №3: ВИБІР ПОСТАЧАЛЬНИКА" + " " * 31 + "║")
    print("╚" + "=" * 98 + "╝")

    tree3, ev_A, ev_B = task3_supplier_selection_tree(prob_A, prob_B, K, N, L)


    print("\n\n")
    print("=" * 100)
    print("📈 ЗАГАЛЬНІ ВИСНОВКИ")
    print("=" * 100)

    print(f"   ЗАВДАННЯ 1 (Вибір типу виробництва):")


    print(f"   Оптимальне рішення: {best_decision1[0]}")
    print(f"   Очікуваний прибуток: {best_decision1[1]:,.0f} г.о.")
    print(f"""
   Висновок: При відсутності додаткової інформації про стан ринку
   та рівноймовірних сприятливих/несприятливих умовах найкраща
   стратегія - {best_decision1[0]}.
""")

    print(f"ЗАВДАННЯ 2 (Дослідження ринку):")
    print(f"   Цінність додаткової інформації: {evpi:,.0f} г.о.")

    if evpi > 0:
        print(f"   Рекомендація: Провести дослідження ринку")
        print(f"   Обґрунтування: Очікуваний додатковий прибуток ({evpi:,.0f} г.о.)")
        print(f"   перевищує вартість досл��дження ({Q:,.0f} г.о.)")
    else:
        print(f"   Рекомендація: Не проводити дослідження ринку")
        print(f"   Обґрунтування: Вартість дослідження ({Q:,.0f} г.о.) не")
        print(f"   виправдовується отриманою інформацією")

    print(f"""
   Висновок: Дослідження ринку дозволяє уточнити ймовірності станів
   природи, що може суттєво вплинути на вибір оптимальної стратегії.
   Теорема Байєса використовується для обчислення апостеріорних
   ймовірностей на основі прогнозу консалтингової фірми.
""")

    print(f"ЗАВДАННЯ 3 (Вибір постачальника):")
    print(f"   Постачальник A: Очікувані витрати = {ev_A:,.0f} г.о.")
    print(f"   Постачальник B: Очікувані витрати = {ev_B:,.0f} г.о.")

    if ev_B > ev_A:
        print(f"   Рекомендація: Вибрати постачальника B")
        print(f"   Економія: {ev_B - ev_A:,.0f} г.о.")
    else:
        print(f"   Рекомендація: Вибрати постачальника A")
        print(f"   Економія: {ev_A - ev_B:,.0f} г.о.")

    print("   • task1_decision_tree.png - дерево рішень для завдання 1")
    print("   • task3_decision_tree.png - дерево рішень для завдання 3")

if __name__ == "__main__":
    main()