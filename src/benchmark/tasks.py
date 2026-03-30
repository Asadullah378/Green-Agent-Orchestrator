"""
Green Agent Orchestrator (GAO) — Benchmark task definitions

15 tasks across three difficulty tiers.  Each task has:
  - id            : unique identifier (E1–E5, M1–M5, H1–H5)
  - difficulty     : easy | medium | hard
  - query          : the user prompt sent to the agent
  - expected_values: strings that MUST appear in a correct answer
  - required_tools : tools the agent should invoke
  - description    : short label for reporting
"""

from __future__ import annotations

BENCHMARK_TASKS: list[dict] = [
    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  EASY — single-tool, single-step tasks                              ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    {
        "id": "E1",
        "difficulty": "easy",
        "query": "What is 15% of 230?",
        "expected_values": ["34.5"],
        "required_tools": ["calculator"],
        "description": "Simple percentage calculation",
    },
    {
        "id": "E2",
        "difficulty": "easy",
        "query": "Convert 100 kilometers to miles.",
        "expected_values": ["62.1371"],
        "required_tools": ["unit_converter"],
        "description": "Distance unit conversion",
    },
    {
        "id": "E3",
        "difficulty": "easy",
        "query": "How many days are there between 2024-01-15 and 2024-06-30?",
        "expected_values": ["167"],
        "required_tools": ["date_calculator"],
        "description": "Date difference calculation",
    },
    {
        "id": "E4",
        "difficulty": "easy",
        "query": "Convert 250 USD to EUR.",
        "expected_values": ["230"],
        "required_tools": ["unit_converter"],
        "description": "Currency conversion",
    },
    {
        "id": "E5",
        "difficulty": "easy",
        "query": "How many words are in the following text: 'The quick brown fox jumps over the lazy dog near the riverbank'?",
        "expected_values": ["12"],
        "required_tools": ["text_processor"],
        "description": "Word count",
    },
    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  MEDIUM — two-step or multi-tool tasks with moderate reasoning      ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    {
        "id": "M1",
        "difficulty": "medium",
        "query": (
            "Calculate the total cost of 3 items priced at $24.99, $15.50, "
            "and $8.75, then add 8.5% sales tax. What is the final total?"
        ),
        "expected_values": ["53.42"],
        "required_tools": ["calculator"],
        "description": "Multi-step arithmetic with tax",
    },
    {
        "id": "M2",
        "difficulty": "medium",
        "query": (
            "Look up all employees in the Engineering department from the "
            "database. What is the average salary of engineers?"
        ),
        "expected_values": ["103250"],
        "required_tools": ["data_lookup", "calculator"],
        "description": "Data lookup + average calculation",
    },
    {
        "id": "M3",
        "difficulty": "medium",
        "query": (
            "How many days from 2026-03-15 until 2026-12-25? "
            "Express the answer in weeks and remaining days."
        ),
        "expected_values": ["285", "40", "5"],
        "required_tools": ["date_calculator", "calculator"],
        "description": "Date diff + weeks conversion",
    },
    {
        "id": "M4",
        "difficulty": "medium",
        "query": (
            "Look up all products in the electronics category from the "
            "database. How many are there, and what is their total value "
            "(price × stock for each, then sum)?"
        ),
        "expected_values": ["5"],
        "required_tools": ["data_lookup", "calculator"],
        "description": "Data lookup + aggregation",
    },
    {
        "id": "M5",
        "difficulty": "medium",
        "query": (
            "Calculate the Body Mass Index (BMI) for a person who is 1.75 m "
            "tall and weighs 82 kg. BMI = weight / height². "
            "Is this person underweight, normal, overweight, or obese?"
        ),
        "expected_values": ["26.7"],
        "required_tools": ["calculator"],
        "description": "BMI calculation + categorisation",
    },
    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  HARD — multi-tool, multi-step analysis tasks                       ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    {
        "id": "H1",
        "difficulty": "hard",
        "query": (
            "Analyse the quarterly revenue of ACME Corp from the companies "
            "database. Calculate: (a) total annual revenue, (b) average "
            "quarterly revenue, (c) the growth rate from Q1 to Q4, and "
            "(d) which quarter had the highest quarter-over-quarter growth."
        ),
        "expected_values": ["5700000", "1425000"],
        "required_tools": ["data_lookup", "calculator"],
        "description": "Multi-metric financial analysis",
    },
    {
        "id": "H2",
        "difficulty": "hard",
        "query": (
            "Compare two mortgage options: "
            "Option A — $250,000 at 6.5% annual for 30 years. "
            "Option B — $250,000 at 5.5% annual for 15 years. "
            "For each, calculate the monthly payment using the formula "
            "M = P * r(1+r)^n / ((1+r)^n - 1) where r is monthly rate "
            "and n is total months. Also calculate total amount paid and "
            "total interest for each option. Which option pays less total interest?"
        ),
        "expected_values": ["Option B", "15"],
        "required_tools": ["calculator"],
        "description": "Mortgage comparison analysis",
    },
    {
        "id": "H3",
        "difficulty": "hard",
        "query": (
            "From the products database, find all items in the 'accessories' "
            "category. Calculate the average price, identify the most and "
            "least expensive items, and compute what a 20% discount would be "
            "on each item. Also convert the total accessories value (sum of "
            "all prices) from USD to EUR."
        ),
        "expected_values": ["accessories"],
        "required_tools": ["data_lookup", "calculator", "unit_converter"],
        "description": "Multi-tool product analysis",
    },
    {
        "id": "H4",
        "difficulty": "hard",
        "query": (
            "Analyse employee salaries from the database. "
            "(a) What is the average salary across all departments? "
            "(b) Which department has the highest average salary? "
            "(c) What is the salary range (max - min)? "
            "(d) How many employees earn above the overall average? "
            "Convert the overall average salary from USD to EUR."
        ),
        "expected_values": ["Engineering"],
        "required_tools": ["data_lookup", "calculator", "unit_converter"],
        "description": "HR analytics with currency conversion",
    },
    {
        "id": "H5",
        "difficulty": "hard",
        "query": (
            "Calculate compound interest on $10,000 invested at 7% annual "
            "rate, compounded monthly, for 5 years. Use the formula "
            "A = P(1 + r/n)^(nt). Then calculate simple interest for the "
            "same parameters (I = P*r*t). What is the difference between "
            "compound and simple interest? Express the compound interest "
            "final amount in EUR as well."
        ),
        "expected_values": ["14176", "3500", "676"],
        "required_tools": ["calculator", "unit_converter"],
        "description": "Compound vs simple interest + conversion",
    },
]


def get_tasks_by_difficulty(difficulty: str) -> list[dict]:
    return [t for t in BENCHMARK_TASKS if t["difficulty"] == difficulty]


def get_task_by_id(task_id: str) -> dict | None:
    for t in BENCHMARK_TASKS:
        if t["id"] == task_id:
            return t
    return None
