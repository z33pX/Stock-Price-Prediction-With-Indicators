import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.transforms as mtrans
from matplotlib.patches import BoxStyle
import pandas as pd
from indicators import CalculateIndicators


def truncate(f, n):
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d + '0' * n)[:n]])


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


def _plot(fig, ax, data, color, label, end_label=None, offset=.15, linestyle='-'):
    BoxStyle._style_list["angled"] = MyStyle

    trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=offset, y=0.0, units='inches')

    ax.plot(data, linestyle, label=label, color=color, linewidth=0.7)

    if end_label:
        ax.text(len(data) - 1, data[-1], end_label, size=7, va="center", ha="center", transform=trans_offset,
                bbox=dict(boxstyle="angled,pad=0.2", alpha=0.6, color=color))
    else:
        ax.text(len(data) - 1, data[-1], truncate(data[-1], 2), size=7, va="center", ha="center",
                transform=trans_offset, bbox=dict(boxstyle="angled,pad=0.2", alpha=0.6, color=color))


def _add_graph(fig, data_df, main_grid_y, position_y, label, color, sharex=None, y_label_color=None):

    ax = plt.subplot2grid((main_grid_y, 4), (position_y, 0), sharex=sharex,
                          rowspan=2, colspan=4, facecolor='#070d00')

    y_color = color

    if y_label_color:
        y_color = y_label_color

    _plot(fig, ax, data_df[label].as_matrix(), color, label)

    ax.spines['left'].set_color(y_color)
    ax.spines['right'].set_color('#070d00')
    ax.spines['top'].set_color('#070d00')
    ax.spines['bottom'].set_color('#070d00')
    ax.legend(loc='upper left')
    ax.grid(linestyle='dotted')
    ax.yaxis.label.set_color(y_color)
    ax.tick_params(axis='y', colors=y_color, labelsize=9)
    ax.tick_params(axis='x', colors='#070d00')

    legend = ax.legend(loc='best', fancybox=True, framealpha=0.5)
    legend.get_frame().set_facecolor('#070d00')
    for line,text in zip(legend.get_lines(), legend.get_texts()):
        text.set_color(line.get_color())

    return ax


def draw(ticker, df, predicted_data_y, ci_object, **kwargs):
    fig = plt.figure(facecolor='#070d00')
    fig.canvas.set_window_title(ticker)
    plt.subplots_adjust(left=.08, bottom=.08, right=.96, top=.96, hspace=.3, wspace=.0)

    accent_color = '#c9c9c9'
    indicators_color = '#007b9f'

    test_data_y = df['Close'].as_matrix()

    draw_RSI = False
    draw_ATR = False
    draw_MACD = False
    draw_Stochastics = False
    draw_moving_average_1 = False
    draw_moving_average_2 = False

    main_grid_y = 4
    postition_y = [2, 2, 2, 2]

    for key in kwargs:
        if kwargs[key]:
            if key == 'accent_color':
                accent_color = kwargs[key]
            if key == 'indicators_color':
                indicators_color = kwargs[key]

            if key == 'draw_moving_average_1':
                draw_moving_average_1 = kwargs[key]

            if key == 'draw_moving_average_2':
                draw_moving_average_2 = kwargs[key]

            if key == 'draw_RSI':
                draw_RSI = True
                postition_y = [x + y for x, y in zip(postition_y, [2, 2, 2, 2])]
                main_grid_y += 2
            if key == 'draw_ATR':
                draw_ATR = True
                postition_y = [x + y for x, y in zip(postition_y, [0, 2, 2, 2])]
                main_grid_y += 2
            if key == 'draw_MACD':
                draw_MACD = True
                postition_y = [x + y for x, y in zip(postition_y, [0, 0, 2, 2])]
                main_grid_y += 2
            if key == 'draw_Stochastics':
                draw_Stochastics = True
                postition_y = [x + y for x, y in zip(postition_y, [0, 0, 0, 2])]
                main_grid_y += 2

    position_labels_y = min([i for i, j in enumerate(postition_y) if j == max(postition_y)])
    arange_y = np.arange(0, len(test_data_y), 1)

    plt.suptitle(ticker, color=accent_color)

    ax_main = plt.subplot2grid((main_grid_y, 4), (0, 0), rowspan=4, colspan=4, facecolor='#070d00')

    # Plot test dataset
    _plot(fig, ax_main, test_data_y, '#f600ff', 'Price')

    # Plot indicators
    if draw_moving_average_1:
        ma = df[ci_object.moving_average_1_label].as_matrix()
        _plot(fig=fig, ax=ax_main, data=ma, color='#47b804',
              label=ci_object.moving_average_1_label, end_label=ci_object.moving_average_1_label, offset=.21)

    if draw_moving_average_2:
        ma = df[ci_object.moving_average_2_label].as_matrix()
        _plot(fig=fig, ax=ax_main, data=ma, color='#205d01',
              label=ci_object.moving_average_2_label, end_label=ci_object.moving_average_2_label, offset=.21)

    if draw_RSI:
        ax = _add_graph(fig=fig, data_df=df, main_grid_y=main_grid_y,
                        position_y=postition_y[0], label='RSI', color=indicators_color, sharex=ax_main,
                        y_label_color=accent_color)

        ax.axhline(30, color='#a90000', linewidth=0.6)
        ax.axhline(70, color='#4c7e1b', linewidth=0.6)
        ax.fill_between(arange_y, df['RSI'], 70, where=(df['RSI'] >= 70),
                        facecolors='#007200', alpha=.5, edgecolor='#007200')
        ax.fill_between(arange_y, df['RSI'], 30, where=(df['RSI'] <= 30),
                        facecolors='#a90000', alpha=.5, edgecolor='#a90000')
        if position_labels_y == 0:
            ax.tick_params(axis='x', colors=accent_color)

    if draw_ATR:
        ax = _add_graph(fig=fig, data_df=df, main_grid_y=main_grid_y,
                        position_y=postition_y[1], label='ATR', color=indicators_color, sharex=ax_main,
                        y_label_color=accent_color)

        if position_labels_y == 1:
            ax.tick_params(axis='x', colors=accent_color)

    if draw_MACD:
        ax = _add_graph(fig=fig, data_df=df, main_grid_y=main_grid_y,
                        position_y=postition_y[2], label='MACD', color=indicators_color, sharex=ax_main,
                        y_label_color=accent_color)
        ax.axhline(0, color=indicators_color, linewidth=0.6)

        if position_labels_y == 2:
            ax.tick_params(axis='x', colors=accent_color)

        ema = pd.ewma(df['MACD'], span=ci_object.MACD_signal)
        ax.plot(arange_y, ema, label='ema_9', color='#ffba00', linewidth=0.7)
        ax.fill_between(arange_y, df['MACD'] - ema, 0 , facecolors='#ffba00', alpha=.5, edgecolor='#ffba00')
        ax.legend(loc='upper left')
        legend = ax.legend(loc='best', fancybox=True, framealpha=0.5)
        legend.get_frame().set_facecolor('#070d00')
        for line, text in zip(legend.get_lines(), legend.get_texts()):
            text.set_color(line.get_color())

    if draw_Stochastics:
        ax = _add_graph(fig=fig, data_df=df, main_grid_y=main_grid_y,
                        position_y=postition_y[3], label='Stochastics', color=indicators_color, sharex=ax_main,
                        y_label_color=accent_color)

        if position_labels_y == 3:
            ax.tick_params(axis='x', colors=accent_color)

    # Plot predicted data
    predicted_data_y = [x for x in predicted_data_y if str(x) != 'nan']
    _plot(fig, ax_main, predicted_data_y, '#ffba00', 'Predicted', linestyle='--')

    # Add tracking error box
    tracking_error = truncate(np.std(predicted_data_y - test_data_y[:len(predicted_data_y)]) * 100, 2)
    ax_main.annotate('Tracking Error: ' + str(tracking_error) + '%', xy=(0.7, 0.05),
                      xycoords='axes fraction', fontsize=10, bbox=dict(facecolor='#ffba00', alpha=0.8),
                      ha='left', va='bottom')

    ax_main.grid(linestyle='dotted')
    ax_main.yaxis.label.set_color(accent_color)
    ax_main.legend(loc='upper left')
    ax_main.spines['left'].set_color(accent_color)
    ax_main.spines['right'].set_color('#070d00')
    ax_main.spines['top'].set_color('#070d00')
    ax_main.spines['bottom'].set_color('#070d00')
    ax_main.tick_params(axis='y', colors=accent_color, labelsize=9)
    ax_main.tick_params(axis='x', colors='#070d00', labelsize=9)

    ax_main.fill_between(np.arange(0, len(predicted_data_y), 1), test_data_y[:len(predicted_data_y)], predicted_data_y,
                         alpha=.05, color='#ffba00')

    legend = ax_main.legend(loc='best', fancybox=True, framealpha=0.5)
    legend.get_frame().set_facecolor('#070d00')
    for line, text in zip(legend.get_lines(), legend.get_texts()):
        text.set_color(line.get_color())


def show():
    plt.show()


def save(path):
    plt.savefig(path, facecolor='#070d00')
