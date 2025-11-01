# ...existing code...
# (Немає змін, якщо результат збігається з Парето для ваших даних)
# ...existing code...
# ...existing code...

class MechanizmPoslidovnoiPostupky(MechanizmBase):
    # ...existing code...
    def find_solution(self, alternatives, criteria, delta):
        remaining = alternatives.copy()
        for i, criterion in enumerate(criteria):
            values = [alt[criterion] for alt in remaining]
            max_value = max(values)
            threshold = max_value - delta[i]
            # Фільтруємо альтернативи, які >= threshold для поточного критерію
            remaining = [alt for alt in remaining if alt[criterion] >= threshold]
        return remaining

# ...existing code...

