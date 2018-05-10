import matplotlib.patches as patches


def truncate(f, n):
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


def draw_verticals(axis, signals):
    for signal in signals:
        if signal[0] is 'SELL':
            axis.axvline(signal[1], color='red',
                         linewidth=0.8,
                         alpha=0.8, linestyle='-')
        if signal[0] is 'BUY':
            axis.axvline(signal[1], color='green',
                         linewidth=0.8,
                         alpha=0.8, linestyle='-')


def draw_signal_evaluation(axis, signals, **kwargs):
    signal_pairs = list()

    # Create list of BUY and SELL pairs --------------
    buy_flag = False
    t = list()
    for signal in signals:
        if signal[0] == 'BUY' and buy_flag is False:
            buy_flag = True
            t.append(signal)
        if signal[0] == 'SELL' and buy_flag is True:
            buy_flag = False
            t.append(signal)
            signal_pairs.append(t)
            t = list()

    # Analysis ----------------------------------------
    # Excelent source of arrow examples:
    # http://matthiaseisen.com/matplotlib/shapes/arrow/

    # Excelent source of rectangle examples:
    # http://matthiaseisen.com/pp/patterns/p0203/

    eval_type = kwargs.get('eval_type', 'rectangle')
    red = kwargs.get('red', 'red')
    green = kwargs.get('green', 'green')

    objects = list()
    ax = axis._make_twin_axes(sharex=axis, sharey=axis)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.patch.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add objects --------------------------
    green_dots_x = list()
    green_dots_y = list()
    red_dots_x = list()
    red_dots_y = list()

    for signal in signal_pairs:
        try:
            x = signal[0][1]
            y = signal[0][2]
            w = signal[1][1] - x
            h = signal[1][2] - y
        except TypeError:
            raise TypeError(
                'Index ' + str(signal[0][1]) +
                ' not in data')

        if h < 0:
            color = red
            red_dots_x.append(x)
            red_dots_y.append(y)
            red_dots_x.append(x + w)
            red_dots_y.append(y + h)
        else:
            color = green
            green_dots_x.append(x)
            green_dots_y.append(y)
            green_dots_x.append(x + w)
            green_dots_y.append(y + h)

        if (h < 0 and kwargs.get(
                'disable_red_signals', False
            ) is False) or \
           (h > 0 and kwargs.get(
               'disable_green_signals', False
           ) is False):

            if eval_type is 'arrow_1':
                patch = patches.FancyArrowPatch(
                        (x, y), (x + w, y + h),
                        arrowstyle='-|>',
                        mutation_scale=20,
                        color=color
                    )

            # eval_type is 'rectangle'
            else:
                patch = patches.Rectangle(
                    (x, y), w, h,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=1,
                    # linestyle='dotted',
                    fill=True,
                    alpha=0.4
                )

            objects.append(patch)

            # Add annotation -----------------------
            ax.add_artist(patch)
            cx = x + w / 2.0
            cy = y + h / 2.0
            change = round(h / y * 100, 3)

            if eval_type is 'arrow_1':
                bbox = {
                    'facecolor': color,
                    'edgecolor': color,
                    'alpha': 0.3,
                    'pad': 2
                }
            else:
                bbox = None

            ax.annotate(
                str(change), (cx, cy),
                color='#535353',
                fontsize=12, ha='center',
                va='center',
                bbox=bbox,
                zorder=100
            )

    # Add dots
    if kwargs.get('dots', True):
        if green_dots_x and kwargs.get(
                'disable_green_signals', False) is False:
            ax.scatter(
                green_dots_x, green_dots_y,
                s=10, marker='o', color=green,
                zorder=100
            )

        if red_dots_x and kwargs.get(
                'disable_red_signals', False) is False:
            ax.scatter(
                red_dots_x, red_dots_y,
                s=10, marker='o', color=red,
                zorder=100
            )

    for rectangle in objects:
        ax.add_patch(rectangle)
