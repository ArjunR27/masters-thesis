import pandas as pd
import matplotlib.pyplot as plt


def plot_question_type_distribution(df: pd.DataFrame) -> None:
    counts = df['question_type'].str.strip().value_counts()

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values, color='steelblue', edgecolor='white')

    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_xlabel('Question Type', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Question Types in LPM QA Dataset', fontsize=14)
    ax.tick_params(axis='x', rotation=30)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig('question_type_distribution.png', dpi=150)
    plt.show()


def main():
    df = pd.read_csv('./lpm_qa_labeled.csv', on_bad_lines='skip')
    plot_question_type_distribution(df)


if __name__ == "__main__":
    main()