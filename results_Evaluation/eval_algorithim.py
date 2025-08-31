from evalplus.evaluate import evaluatefrom evalplus.data import get_human_eval_plus

with open('algorithim_solutions.json', 'r') as f:
    solutions = json.load(f)

results = evaluate(
    dataset="humaneval",
    samples=[solutions],
    parallel=4,
    timeout=10
)

print("Evaluation Results:")
print(f"Pass Rate: {results['pass@1']:.2%}")
print(f"Total Problems: {results['total']}")
print(f"Passed Problems: {results['passed']}")
