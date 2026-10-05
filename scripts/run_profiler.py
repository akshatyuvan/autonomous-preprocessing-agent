"""
scripts/run_profiler.py — run the real Profiler (real gpt-4o-mini) on data/raw/sample.csv.

Environment: LOCAL (Mac). Needs OPENAI_API_KEY in .env. Cost: a fraction of a cent.
Run: python -m scripts.run_profiler
"""
from dotenv import load_dotenv

load_dotenv()  # must run before the OpenAI client is created

from agents.profiler import profiler_node
from state.schema import make_initial_state


def main() -> None:
    state = make_initial_state(dataset_path="data/raw/sample.csv", dataset_name="sample")
    result = profiler_node(state)
    p = result["profiler"]
    print(f"\n{p['row_count']} rows x {p['column_count']} cols, {len(p['issues'])} issues\n")
    for i in p["issues"]:
        print(f"{i['issue_id']}  {i['issue_type']:<22} {i['column']:<16} {i['severity']:<7} "
              f"rows={i['affected_rows']:<4} {i['detail']}")
    print("\nSUMMARY:\n" + p["profile_summary"])
    if result["errors"]:
        print("\nERRORS:", result["errors"])


if __name__ == "__main__":
    main()