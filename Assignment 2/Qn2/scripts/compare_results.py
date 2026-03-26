import json

# Load results from each experiment
def load_results(filepath):
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results

baseline = load_results("results/baseline_results.jsonl")
r8_1k = load_results("results/r8_1k_results.jsonl")
r8_2k = load_results("results/r8_2k_results.jsonl")
r32_1k = load_results("results/r32_1k_results.jsonl")

print("="*120)
print("MODEL OUTPUT COMPARISON TABLE")
print("="*120)
print(f"{'ID':<5} {'Prompt':<35} {'True Label':<12} {'Baseline':<25} {'r=8,1k':<25} {'r=8,2k':<25} {'r=32,1k':<25}")
print("-"*120)

# Compare first 5 samples
for i in range(min(5, len(baseline))):
    print(f"{baseline[i]['id']:<5} {baseline[i]['prompt'][:33]:<35} {baseline[i]['true_label']:<12} {baseline[i]['prediction'][:23]:<25} {r8_1k[i]['prediction'][:23]:<25} {r8_2k[i]['prediction'][:23]:<25} {r32_1k[i]['prediction'][:23]:<25}")

print("="*120)

# Also create a more detailed comparison file
with open("results/detailed_comparison.md", "w") as f:
    f.write("# Model Output Comparison\n\n")
    f.write("## Sample Predictions\n\n")
    f.write("| ID | Prompt | True Label | Baseline | r=8, 1k | r=8, 2k | r=32, 1k |\n")
    f.write("|---|--------|------------|----------|---------|---------|----------|\n")
    
    for i in range(min(10, len(baseline))):
        f.write(f"| {baseline[i]['id']} | {baseline[i]['prompt'][:50]} | {baseline[i]['true_label']} | {baseline[i]['prediction'][:30]} | {r8_1k[i]['prediction'][:30]} | {r8_2k[i]['prediction'][:30]} | {r32_1k[i]['prediction'][:30]} |\n")

print("\nDetailed comparison saved to results/detailed_comparison.md")
