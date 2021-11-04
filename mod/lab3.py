import random
from collections import defaultdict
from typing import Optional, List, Set, Tuple, Dict
from numpy import array
from numpy.linalg import solve  # type: ignore

from structlog import get_logger
from copy import copy

from mod.common.processor import BaseProcessor

logger = get_logger(__name__)


class Block:
    def __init__(self) -> None:
        self._prev: Optional["Block"] = None
        self._next: Optional["Block"] = None

    def set_next(self, next: "Block"):
        self._next = next
        next._prev = self

    def set_prev(self, prev: "Block"):
        self._prev = prev
        prev._next = self

    def reset_tick(self) -> None:
        raise NotImplementedError()

    def make_action(self, outcome: bool = None) -> bool:
        raise NotImplementedError()

    @property
    def busy(self) -> bool:
        raise NotImplementedError()

    def add_task(self) -> None:
        raise NotImplementedError()

    @property
    def state(self) -> int:
        raise NotImplementedError()

    @property
    def decline_count(self) -> int:
        return 0

    @property
    def representation(self) -> List[str]:
        raise NotImplementedError()

    @property
    def p(self) -> float:
        raise NotImplementedError()


class BlockingGenerator(Block):
    def __init__(self) -> None:
        super().__init__()
        self._state = 1
        self._used = False
        self._fake = False

    def reset_tick(self) -> None:
        self._used = False
        self._fake = False

    def make_action(self, outcome: bool = None) -> bool:
        if self._state == 2:
            if self._next is not None and not self._next.busy:
                self._next.add_task()
                self._state = 1
                self._used = True
                self._fake = False
                return True
            self._fake = False
            return False
        if self._used:
            return False
        if self._state == 1:
            self._state = 0
            self._used = True
            return True
        self._fake = True
        self._state = 2
        return self.make_action(outcome=outcome)

    @property
    def state(self) -> int:
        return self._state

    @property
    def busy(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"<BlockingGenerator: used={self._used}, state={self._state}>"

    @property
    def representation(self) -> List[str]:
        return [
            "   -----   ",
            "  /     \\  ",
            " /       \\ ",
            "|         |",
            "|    2    |",
            "|         |",
            " \\       / ",
            "  \\     /  ",
            "   -----   ",
        ]


class Queue(Block):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size
        self._state = 0

    def reset_tick(self) -> None:
        pass

    def make_action(self, outcome: bool = None) -> bool:
        if self._state and self._next is not None and not self._next.busy:
            self._next.add_task()
            self._state -= 1
            return True
        return False

    @property
    def busy(self) -> bool:
        return self._state == self.size

    def add_task(self) -> None:
        if self._state >= self.size:
            raise RuntimeError("Max queue size exceeded")
        self._state += 1

    @property
    def state(self) -> int:
        return self._state

    def __repr__(self) -> str:
        return f"<Queue: size={self.size}, state={self._state}>"

    @property
    def representation(self) -> List[str]:
        return [
            "-----------",
            "|         |",
            "|         |",
            "|         |",
            f"|    {self.size}    |",
            "|         |",
            "|         |",
            "|         |",
            "-----------",
        ]


class NonBlockingChannel(Block):
    def __init__(self, p: float):
        super().__init__()
        self._state = 0
        self._used = False
        self._p = p
        self._decline_count = 0

    def reset_tick(self) -> None:
        self._used = False
        self._decline_count = 0

    def make_action(self, outcome: bool = None) -> bool:
        if self._used:
            return False
        self._used = True
        if outcome is None:
            outcome = random.uniform(0, 1) <= self._p
        if outcome:
            if self._state:
                if self._next is not None:
                    if self._next.busy:
                        self._decline_count += 1
                    else:
                        self._next.add_task()
            self._state = 0
        return True

    @property
    def busy(self) -> bool:
        return self._state == 1

    def add_task(self) -> None:
        if self._state == 1:
            raise RuntimeError("Channel already busy")
        self._state = 1

    @property
    def state(self) -> int:
        return self._state

    @property
    def decline_count(self) -> int:
        return self._decline_count

    @property
    def representation(self) -> List[str]:
        return [
            "   -----   ",
            "  /     \\  ",
            " /       \\ ",
            "|         |",
            f"| p={self._p:.2f}  |",
            "|         |",
            " \\       / ",
            "  \\     /  ",
            "   -----   ",
            "     \\     ",
            "      \\    ",
            "       _|  ",
        ]

    @property
    def p(self) -> float:
        return self._p


def get_representation(machine: List[Block]) -> List[str]:
    result: List[str] = []
    max_size = max(len(block.representation) for block in machine)
    min_size = min(len(block.representation) for block in machine)
    for i in range(max_size):
        lines = []
        appender = "->" if i == (min_size // 2) else "  "
        for block in machine:
            if i >= len(block.representation):
                lines.append(" " * (len(block.representation[0])))
            else:
                lines.append(block.representation[i])
        result.append("| " + appender.join(lines) + " |")
    result.insert(0, "|" + (" " * (len(result[0]) - 2)) + "|")
    result.insert(0, "-" * len(result[0]))
    result.append("|" + (" " * (len(result[0]) - 2)) + "|")
    result.append("-" * len(result[0]))

    return result


def get_state(machine: List[Block]) -> str:
    return "".join(str(block.state) for block in machine)


def iterate(machine: List[Block], outcomes: Tuple[bool, bool]) -> None:
    something_changed = True
    for block in machine:
        block.reset_tick()
    while something_changed:
        something_changed = False
        for block, outcome in zip(
            machine[::-1], [None, outcomes[0], None, outcomes[1]][::-1]
        ):
            something_changed = something_changed or block.make_action(outcome=outcome)


def iterate_random(machine: List[Block]) -> None:
    something_changed = True
    for block in machine:
        block.reset_tick()
    while something_changed:
        something_changed = False
        for block in machine[::-1]:
            something_changed = something_changed or block.make_action()


class Simulator:
    def __init__(self):
        self.processed_states: Set[str] = set()
        self.transitions_chances: Dict[
            str, Dict[str, Tuple[str, float, int]]
        ] = defaultdict(dict)

    def generate_transitions(
        self, machine: List[Block]
    ) -> Dict[str, Dict[str, Tuple[str, float, int]]]:
        self.processed_states = set()
        self.transitions_chances = defaultdict(dict)
        self._generate_transitions(machine)
        return self.transitions_chances

    def _generate_transitions(self, machine: List[Block]) -> None:
        machine_state = get_state(machine)
        if machine_state in self.processed_states:
            return
        self.processed_states.add(machine_state)
        chances = [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ]

        for chance in chances:
            machine_copy = [copy(block) for block in machine]
            for i in range(len(machine_copy) - 1):
                machine_copy[i].set_next(machine_copy[i + 1])
            iterate(machine_copy, chance)
            new_state = get_state(machine_copy)
            t1 = (
                ("p1", machine_copy[1].p)
                if chance[0]
                else ("q1", 1 - machine_copy[1].p)
            )
            t2 = (
                ("p2", machine_copy[3].p)
                if chance[1]
                else ("q2", 1 - machine_copy[3].p)
            )
            probability = "*".join((t1[0], t2[0]))
            probability_float = t1[1] * t2[1]  # type: float
            if new_state in self.transitions_chances[machine_state].keys():
                (
                    old_probability_str,
                    old_probability_float,
                    old_loses,
                ) = self.transitions_chances[machine_state][new_state]
                new_probability_str = " + ".join([old_probability_str, probability])
                new_probability_float = old_probability_float + probability_float
                self.transitions_chances[machine_state][new_state] = (
                    new_probability_str,
                    new_probability_float,
                    old_loses + sum(b.decline_count for b in machine_copy),
                )
            else:
                self.transitions_chances[machine_state][new_state] = (
                    probability,
                    probability_float,
                    sum(b.decline_count for b in machine_copy),
                )
            self._generate_transitions(machine_copy)


class Lab3(BaseProcessor):
    def __init__(self, p1: float, p2: float) -> None:
        super().__init__()
        assert 0 <= p1 <= 1
        assert 0 <= p2 <= 1
        self.p1 = p1
        self.p2 = p2
        self.generator = BlockingGenerator()
        self.queue = Queue(size=1)
        self.channel1 = NonBlockingChannel(p=p1)
        self.channel2 = NonBlockingChannel(p=p2)
        self.items = [self.generator, self.channel1, self.queue, self.channel2]
        self.generator.set_next(self.channel1)
        self.channel1.set_next(self.queue)
        self.queue.set_next(self.channel2)

    def get_chances(
        self, transitions_chances: Dict[str, dict[str, Tuple[str, float, int]]]
    ) -> Dict[str, float]:
        all_states = sorted(list(transitions_chances.keys()))
        unsolved_states = copy(all_states)
        result: Dict[str, float] = {}

        got_new_solved = True
        while got_new_solved:
            got_new_solved = False
            new_unsolved = copy(unsolved_states)
            for state in unsolved_states:
                c = 0.0
                found_unknown_transition = False
                for from_state in all_states:
                    stat = transitions_chances[from_state].get(state)
                    if stat is not None:
                        if from_state in result.keys():
                            c += result[from_state] * stat[1]
                        else:
                            found_unknown_transition = True

                if not found_unknown_transition:
                    got_new_solved = True
                    result[state] = c
                    new_unsolved = [
                        unsolved_state
                        for unsolved_state in unsolved_states
                        if unsolved_state != state
                    ]
            unsolved_states = copy(new_unsolved)

        coefficients: List[List[float]] = []
        consts = []
        for state in unsolved_states:
            lin_coefficients: List[float] = [0.0 for j in range(len(unsolved_states))]
            c = 0.0
            for from_state in all_states:
                if state in transitions_chances[from_state].keys():
                    k = transitions_chances[from_state][state][1]
                    if from_state in result.keys():
                        c -= k * result[from_state]
                    else:
                        lin_coefficients[unsolved_states.index(from_state)] = k
                if from_state == state:
                    lin_coefficients[unsolved_states.index(from_state)] -= 1.0
            coefficients.append(lin_coefficients)
            consts.append(c)

        for k in range(len(coefficients[0])):
            coefficients[0][k] += 1.0
        consts[0] += 1

        print("     " + "  ".join(f" {state}" for state in unsolved_states))
        for state, lin_coefficients, c in zip(unsolved_states, coefficients, consts):
            print(
                f"{state} "
                + ", ".join(f"{k:5.2f}" for k in lin_coefficients)
                + f" : {c}"
            )
        a = array(coefficients)
        b = array(consts)

        answer = solve(a, b)

        result.update(
            {state: probability for state, probability in zip(unsolved_states, answer)}
        )
        return result

    def get_decline_chance(
        self,
        chances: Dict[str, float],
        transitions: Dict[str, Dict[str, Tuple[str, float, int]]],
    ) -> float:
        result = 0.0
        probability = self.p1 * (1.0 - self.p2)
        for source_state, outcomes in transitions.items():
            for outcome, (chance_str, chance_float, loses) in outcomes.items():
                if loses == 0:
                    continue
                result += chances[source_state] * probability
        return result

    def get_block_chance(self, chances: Dict[str, float]) -> float:
        result = 0.0
        for state, probability in chances.items():
            if state.startswith("2"):
                result += probability
        return result

    def get_avg_queue_lengch(self, chances: Dict[str, float]) -> float:
        result = 0.0
        for state, probability in chances.items():
            if state[2] == "1":
                result += probability
        return result

    def get_avg_requests_num(self, chances: Dict[str, float]) -> float:
        result = 0.0
        for state, probability in chances.items():
            count = int(state[0]) + int(state[1]) + int(state[2])
            result += count * probability
        return result

    def get_absolute_channel_capacity(self, chances: Dict[str, float]) -> float:
        result = 0.0
        for state, probability in chances.items():
            if state.endswith("1"):
                result += probability * self.p2
        return result

    def get_channel1_load_coefficient(self, chances: Dict[str, float]) -> float:
        result = 0.0
        for state, probability in chances.items():
            if state[1] == "1":
                result += probability
        return result

    def get_channel2_load_coefficient(self, chances: Dict[str, float]) -> float:
        result = 0.0
        for state, probability in chances.items():
            if state[3] == "1":
                result += probability
        return result

    def execute(self) -> None:
        for line in get_representation(self.items):
            print(line)
        transitions_chances = Simulator().generate_transitions(self.items)
        print()
        for state, transitions in transitions_chances.items():
            print(f"{state}:")
            for new_state, stat in transitions.items():
                print(f"\t{new_state} {(stat[0], stat[1])} {stat[2]}")

        print()
        chances = self.get_chances(transitions_chances)
        for state, probability in chances.items():
            print(f"p('{state}') = {probability:.15f}")
        print()
        decline_chance = self.get_decline_chance(chances, transitions_chances)
        print(f"decline probability        = {decline_chance:.20f} (Pотк)")
        block_chance = self.get_block_chance(chances)
        print(f"block probability          = {block_chance:.20f} (Pбл)")
        average_queue_length = self.get_avg_queue_lengch(chances)
        print(f"avgerage queue length      = {average_queue_length:.20f} (Lоч)")
        average_requests_number = self.get_avg_requests_num(chances)
        print(f"average requests number    = {average_requests_number:.20f} (Lс)")
        absolute_channel_capacity = self.get_absolute_channel_capacity(chances)
        print(f"absolute channel capacity  = {absolute_channel_capacity:.20f} (A)")
        relative_channel_capacity = absolute_channel_capacity / (0.5 * (1 - block_chance))
        print(f"relative channel capacity  = {relative_channel_capacity:.20f} (Q)")
        average_in_queue_time = average_queue_length / absolute_channel_capacity
        print(f"average in queue time      = {average_in_queue_time:.20f} (Wоч)")
        average_in_system_time = average_requests_number / absolute_channel_capacity
        print(f"average in system time     = {average_in_system_time:.20f} (Wс)")
        channel_1_load_coefficient = self.get_channel1_load_coefficient(chances)
        print(f"channel 1 load coefficient = {channel_1_load_coefficient:.20f} (Kкан1)")
        channel_2_load_coefficient = self.get_channel2_load_coefficient(chances)
        print(f"channel 2 load coefficient = {channel_2_load_coefficient:.20f} (Ккан2)")

        machine = [copy(block) for block in self.items]
        for i in range(len(machine) - 1):
            machine[i].set_next(machine[i + 1])
        states_stats: Dict[str, int] = defaultdict(int)
        declines_count = 0
        iterations_count = 1000000

        for i in range(iterations_count):
            states_stats[get_state(machine)] += 1
            declines_count += sum(block.decline_count for block in machine)
            iterate_random(machine)

        empirical_chances = {}
        for state, count in states_stats.items():
            empirical_chances[state] = count / iterations_count

        print("\nempirical probabilities:")
        for state, count in states_stats.items():
            print(
                f"{state}: {(count / iterations_count):.6f}, error = {abs((count / iterations_count) - chances[state] ):.15f}"
            )

        print()
        empirical_decline_chance = declines_count / iterations_count
        print(
            f"decline probability        = {empirical_decline_chance:.20f}; "
            f"error = {abs(empirical_decline_chance - decline_chance):.20f}"
        )
        empirical_block_chance = self.get_block_chance(empirical_chances)
        print(
            f"block probability          = {empirical_block_chance:.20f}; "
            f"error = {abs(empirical_block_chance - block_chance):.20f}"
        )
        empirical_average_queue_length = self.get_avg_queue_lengch(empirical_chances)
        print(
            f"avgerage queue length      = {empirical_average_queue_length:.20f}; "
            f"error = {abs(empirical_average_queue_length - average_queue_length):.20f}"
        )
        empirical_average_requests_number = self.get_avg_requests_num(empirical_chances)
        print(
            f"average requests number    = {empirical_average_requests_number:.20f}; "
            f"error = {abs(empirical_average_requests_number - average_requests_number):.20f}"
        )
        empirical_absolute_channel_capacity = self.get_absolute_channel_capacity(
            empirical_chances
        )
        print(
            f"absolute channel capacity  = {empirical_absolute_channel_capacity:.20f}; "
            f"error = {abs(empirical_absolute_channel_capacity - absolute_channel_capacity):.20f}"
        )
        empirical_relative_channel_capacity = empirical_absolute_channel_capacity / (0.5 * (1 - empirical_block_chance))
        print(
            f"relative channel capacity  = {empirical_relative_channel_capacity:.20f}; "
            f"error = {abs(empirical_relative_channel_capacity - relative_channel_capacity):.20f}"
        )
        empirical_average_in_queue_time = (
            empirical_average_queue_length / empirical_absolute_channel_capacity
        )
        print(
            f"average in queue time      = {empirical_average_in_queue_time:.20f}; "
            f"error = {abs(empirical_average_in_queue_time - average_in_queue_time):.20f}"
        )
        empirical_average_in_system_time = (
            empirical_average_requests_number / empirical_absolute_channel_capacity
        )
        print(
            f"average in system time     = {empirical_average_in_system_time:.20f}; "
            f"error = {abs(empirical_average_in_system_time - average_in_system_time):.20f}"
        )
        empirical_channel_1_load_coefficient = self.get_channel1_load_coefficient(
            empirical_chances
        )
        print(
            f"channel 1 load coefficient = {empirical_channel_1_load_coefficient:.20f}; "
            f"error = {abs(empirical_channel_1_load_coefficient - channel_1_load_coefficient):.20f}"
        )
        empirical_channel_2_load_coefficient = self.get_channel2_load_coefficient(
            empirical_chances
        )
        print(
            f"channel 2 load coefficient = {empirical_channel_2_load_coefficient:.20f}; "
            f"error = {abs(empirical_channel_2_load_coefficient - channel_2_load_coefficient):.20f}"
        )
