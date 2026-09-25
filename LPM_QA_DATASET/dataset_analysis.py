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


def plot_per_speaker_question_counts(df: pd.DataFrame) -> None:
    speaker = df['lecture_key'].str.split('/').str[0]
    counts = speaker.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values, color='steelblue', edgecolor='white')

    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_xlabel('Speaker / Course', fontsize=12)
    ax.set_ylabel('Number of Questions', fontsize=12)
    ax.set_title('TinyLPM-QA Question Count per Speaker/Course', fontsize=14)
    ax.tick_params(axis='x', rotation=0)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig('lpm_qa_per_speaker_dist.png', dpi=150)
    plt.show()


def plot_span_count_distribution(df: pd.DataFrame) -> None:
    span_counts = df['answer_timestamps'].str.split('|').str.len()
    counts = span_counts.value_counts().sort_index()
    total = counts.sum()

    cmap = plt.get_cmap('Blues')
    colors = [cmap(0.35 + 0.55 * i / (len(counts) - 1)) for i in range(len(counts))]

    fig, ax = plt.subplots(figsize=(10, 2.6))
    left = 0
    handles = []
    for (n_spans, count), color in zip(counts.items(), colors):
        width = count / total
        bar = ax.barh(0, width, left=left, height=0.6, color=color, edgecolor='white')
        handles.append(bar)
        if width > 0.06:
            ax.text(left + width / 2, 0, f"{count}\n({width*100:.1f}%)",
                    ha='center', va='center', fontsize=9,
                    color='white' if n_spans >= 3 else 'black')
        left += width

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.axis('off')
    ax.set_title('TinyLPM-QA: Number of Answer Spans per Question (n=150)', fontsize=13, pad=10)
    ax.legend(handles, [f"{n} span{'s' if n > 1 else ''}" for n in counts.index],
              loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=len(counts), frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig('lpm_qa_span_distribution.png', dpi=150)
    plt.show()


def main():
    df = pd.read_csv('./lpm_qa_labeled.csv', on_bad_lines='skip')
    plot_question_type_distribution(df)
    plot_per_speaker_question_counts(df)
    plot_span_count_distribution(df)


if __name__ == "__main__":
    main()