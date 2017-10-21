import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.transforms as mtrans
from matplotlib.patches import BoxStyle


def _plot(ax, fig, data, color, label, end_label=None, offset=.15, linestyle='-'):
    trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=offset, y=0.0, units='inches')

    ax.plot(data, linestyle, label=label, color=color, linewidth=0.7)

    if end_label == None:
        ax.text(len(data) - 1, data[-1], truncate(data[-1], 2),
                  size=7, va="center", ha="center", transform=trans_offset,
                  bbox=dict(boxstyle="angled,pad=0.2", alpha=0.6, color=color))
    else:
        ax.text(len(data) - 1, data[-1], end_label,
                  size=7, va="center", ha="center", transform=trans_offset,
                  bbox=dict(boxstyle="angled,pad=0.2", alpha=0.6, color=color))


class MyStyle(BoxStyle._Base):

    def __init__(self, pad=0.3):
        self.pad = pad
        super(MyStyle, self).__init__()

    def transmute(self, x0, y0, width, height, mutation_size):

        # padding
        pad = mutation_size * self.pad

        # width and height with padding added.
        width, height = width + 2.*pad, height + 2.*pad,

        # boundary of the padded box
        # x0 = x0 + 3 + width / 2
        x0, y0 = x0-pad, y0-pad,
        x1, y1 = x0+width, y0 + height

        cp = [(x0, y0),
              (x1, y0), (x1, y1), (x0, y1),
              (x0-pad, (y0+y1)/2.), (x0, y0),
              (x0, y0)]

        com = [Path.MOVETO,
               Path.LINETO, Path.LINETO, Path.LINETO,
               Path.LINETO, Path.LINETO,
               Path.CLOSEPOLY]

        path = Path(cp, com)

        return path


def truncate(f, n):
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


def draw(ticker, dataset_test_y, l_test_pred, data, features, moving_average_1,
         moving_average_2, draw_moving_averages=True):

    dataset_train_length = len(data.index) - len(dataset_test_y)

    # Build the graph
    BoxStyle._style_list["angled"] = MyStyle

    fig = plt.figure(facecolor='#000606')
    plt.subplots_adjust(left=.08, bottom=.08, right=.96, top=.96, hspace=.0, wspace=.06)
    plt.suptitle(ticker, color='#00decc')

    ax_price = plt.subplot2grid((4, 4), (0, 0), rowspan=4, colspan=4, facecolor='#000606')

    # Plot test dataset
    _plot(ax_price, fig, dataset_test_y, '#f600ff', 'Price')

    # Plot indicators
    if draw_moving_averages:
        if moving_average_1 in features:
            ma = data[moving_average_1].as_matrix()[dataset_train_length:]
            _plot(ax_price, fig, ma, '#47b804', label=moving_average_1, end_label=moving_average_1, offset=.21)

        if moving_average_2 in features:
            ma = data[moving_average_2].as_matrix()[dataset_train_length:]
            _plot(ax_price, fig, ma, '#205d01', label=moving_average_2, end_label=moving_average_2, offset=.21)

    # Plot predicted data
    l_test_pred = [x for x in l_test_pred if str(x) != 'nan']
    _plot(ax_price, fig, l_test_pred, '#ffba00', 'Predicted', linestyle='--')

    # Add tracking error box
    tracking_error = truncate(np.std(l_test_pred - dataset_test_y[:len(l_test_pred)]) * 100, 2)
    ax_price.annotate('Tracking Error: ' + str(tracking_error) + '%', xy=(0.7, 0.05),
                      xycoords='axes fraction', fontsize=10, bbox=dict(facecolor='#ffba00', alpha=0.6),
                      ha='left', va='bottom')

    ax_price.grid(linestyle='dotted')
    ax_price.yaxis.label.set_color('#037f7a')
    ax_price.legend(loc='upper left')
    ax_price.spines['left'].set_color('#037f7a')
    ax_price.spines['right'].set_color('#000606')
    ax_price.spines['top'].set_color('#000606')
    ax_price.spines['bottom'].set_color('#000606')
    ax_price.tick_params(axis='y', colors='#037f7a')
    ax_price.tick_params(axis='x', colors='#037f7a')

    ax_price.fill_between(np.arange(0, len(l_test_pred), 1), dataset_test_y[:len(l_test_pred)], l_test_pred,
                          alpha=.05, color='#ffba00')

    legend = ax_price.legend(loc='best', fancybox=True, framealpha=0.5)
    legend.get_frame().set_facecolor('#000606')
    for line,text in zip(legend.get_lines(), legend.get_texts()):
        text.set_color(line.get_color())

    plt.show()
