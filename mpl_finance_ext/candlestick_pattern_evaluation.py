import matplotlib.patches as patches


def draw_pattern_evaluation(data_ohlc, axis, cs_patterns, **kwargs):

    # Analysis ----------------------------------------
    # Excelent source of rectangle examples:
    # http://matthiaseisen.com/pp/patterns/p0203/

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
    for pattern in cs_patterns:
        try:
            max_v = list()
            min_v = list()
            start_index = pattern[1]
            stop_index = pattern[2]
            for i in range(stop_index - start_index + 1):
                max_v.append(float(
                    data_ohlc.iloc[[i + start_index]].max(axis=1)))
                min_v.append(float(
                    data_ohlc.iloc[[i + start_index]].min(axis=1)))

            x = pattern[1] - 0.6
            y = max(max_v) - 0.05 * (min(min_v) - max(max_v))
            w = pattern[2] - x + 0.6
            h = min(min_v) - y + 0.05 * (min(min_v) - max(max_v))

        except TypeError:
            raise TypeError(
                'Index ' + str(pattern[1]) +
                ' not in data')

        if 'berepa' in pattern[0] or 'becopa' in pattern[0]:
            color = red
        elif 'burepa' in pattern[0] or 'bucopa' in pattern[0]:
            color = green
        else:
            color = 'white'

        patch = patches.Rectangle(
                (x, y), w, h,
                facecolor=color,
                edgecolor=color,
                linewidth=1,
                linestyle='dotted',
                fill=False,
        )

        objects.append(patch)

        # Add annotation -----------------------
        ax.add_artist(patch)
        cx = x + w / 2.0
        cy = max(y + h, y)

        ax.annotate(
            str(pattern[0]), (cx, cy),
            color=color,
            fontsize=12, ha='center',
            va='bottom',
            zorder=100
        )

    for rectangle in objects:
        ax.add_patch(rectangle)
