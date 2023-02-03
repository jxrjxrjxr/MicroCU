import matplotlib.pyplot as plt
import numpy as np

def plotBar(labels, pravg, rankscore, sample_rate, save=True):
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, pravg, width, label='PR@avg')
    rects2 = ax.bar(x + width/2, rankscore, width, label='RankScore')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title('p={}'.format(sample_rate))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    ax.set_xlim()

    fig.tight_layout()
    print("111111")
    if not save:
        plt.show()
    else:
        print("here")
        plt.savefig("{}-{}.png".format(len(labels, sample_rate)))