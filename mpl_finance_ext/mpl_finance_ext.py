import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import BoxStyle
from six.moves import xrange, zip

from angled_box_style import AngledBoxStyle
from candlestick_pattern_evaluation import draw_pattern_evaluation
from signal_evaluation import draw_signal_evaluation
from signal_evaluation import draw_verticals

# Colors:
label_colors = '#c1c1c1'
background_color = '#ffffff'

red = '#c2c2c2'     # '#fe0000'
green = '#13bebc'   # '#00fc01'

color_set = ['#13bebc', '#b0c113', '#c1139e', '#c17113', '#0d8382']

# Create angled box style
BoxStyle._style_list["angled"] = AngledBoxStyle


def _candlestick2_ohlc(
        ax, opens, highs, lows, closes,
        width=4.0, colorup=green, colordown=red,
        alpha=0.75, index_fix=True
):
    # Functions not supported in macOS
    # colorup = mcolors.to_rgba(colorup, alpha)
    # colordown = mcolors.to_rgba(colordown, alpha)
    line_colors = list()
    poly_colors_up = list()
    poly_colors_down = list()

    count = 0
    delta = width - 0.16
    poly_segments_up = list()
    poly_segments_down = list()
    line_segments = list()
    for i, open, close, high, low in zip(
            xrange(len(opens)), opens, closes, highs, lows):
        if index_fix:
            i = opens.index[count]
            count += 1
        if open != -1 and close != -1:
            # Simple modification to draw a line for open == close
            # if open == close:
            #     open -= 0.01 * abs(high - low)

            if close > open:
                poly_segments_up.append(
                    ((i - delta, open), (i - delta, close),
                     (i + delta, close), (i + delta, open))
                )
                poly_colors_up.append(colorup)
                if close < high:
                    line_segments.append(((i, close), (i, high)))
                    line_colors.append(colorup)
                if low < open:
                    line_segments.append(((i, low), (i, open)))
                    line_colors.append(colorup)

            else:
                poly_segments_down.append(
                    ((i - delta, open), (i - delta, close),
                     (i + delta, close), (i + delta, open))
                )
                poly_colors_down.append(colordown)
                if open < high:
                    line_segments.append(((i, open), (i, high)))
                    line_colors.append(colordown)
                if low < close:
                    line_segments.append(((i, low), (i, close)))
                    line_colors.append(colordown)

    use_aa = 0,  # use tuple here
    line_collection = LineCollection(
        line_segments,
        colors=line_colors,
        linewidths=0.7,
        antialiaseds=use_aa,
        linestyles='solid'
    )

    bar_collection_down = PolyCollection(
        poly_segments_down,
        facecolors=red,
        edgecolors=poly_colors_down,
        antialiaseds=use_aa,
        linewidths=0,
    )

    bar_collection_up = PolyCollection(
        poly_segments_up,
        facecolors=green,
        edgecolors=poly_colors_up,
        antialiaseds=use_aa,
        linewidths=0,
    )

    if index_fix:
        minx, maxx = closes.index[0], closes.index[-1]
    else:
        minx, maxx = 0, len(line_segments)

    miny = min([low for low in lows if low != -1])
    maxy = max([high for high in highs if high != -1])

    corners = (minx, miny), (maxx, maxy)
    ax.update_datalim(corners)
    ax.autoscale_view()

    ax.add_collection(line_collection)
    ax.add_collection(bar_collection_up)
    ax.add_collection(bar_collection_down)
    return line_collection, bar_collection_up, bar_collection_down


def _add_text_box(fig, axis, text, x_p, y_p):
    x = axis.get_xlim()
    y = axis.get_ylim()
    text_x = x[0] / 100 * x_p
    text_y = y[1] / 100 * y_p

    trans_offset = mtrans.offset_copy(
        axis.transData,
        fig=fig,
        x=0.0,
        y=0.0,
        units='inches'
    )

    axis.text(text_x, text_y, text, ha='left', va='center',
              transform=trans_offset, color='#535353',
              bbox=dict(alpha=0.4, color=label_colors))


def _tail(fig, ax, kwa, data=None, plot_columns=None):

    # Vertical span and lines:
    vline = kwa.get('vline', None)
    if vline is not None:
        plot_vline(
            axis=ax, index=vline
        )

    vspan = kwa.get('vspan', None)
    if vspan is not None:
        plot_vspan(
            axis=ax, index=vspan
        )

    # Names, title, labels
    name = kwa.get('name', None)
    if name is not None:
        ax.text(
            0.5, 0.95, name, color=label_colors,
            horizontalalignment='center',
            fontsize=10, transform=ax.transAxes,
            zorder=120
        )

    xlabel = kwa.get('xlabel', None)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ylabel = kwa.get('ylabel', None)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    title = kwa.get('title', None)
    if title is not None:
        ax.set_title(title)

    # Plot columns
    enable_flags = kwa.get('enable_flags', True)
    if kwa.get('set_flags_at_the_end', True):
        last_index = data.index.values[-1]
    else:
        last_index = None

    if plot_columns is not None and data is not None:
        for i, col in enumerate(plot_columns):
            series = data[col]
            ax.plot(series, linewidth=0.7,
                    color=color_set[i])
            if enable_flags:
                add_price_flag(
                    fig=fig, axis=ax,
                    series=data[col],
                    color=color_set[i],
                    last_index=last_index
                )

    xhline = kwa.get('xhline1', None)
    if xhline is not None:
        ax.axhline(xhline, color=label_colors,
                   linewidth=0.5)

    xhline2 = kwa.get('xhline2', None)
    if xhline2 is not None:
        ax.axhline(xhline2, color=label_colors,
                   linewidth=0.5)

    xhline_red = kwa.get('xhline_red', None)
    if xhline_red is not None:
        ax.axhline(xhline_red, color=red,
                   linewidth=0.5)

    xhline_green = kwa.get('xhline_green', None)
    if xhline_green is not None:
        ax.axhline(xhline_green, color=green,
                   linewidth=0.5)

    xhline_dashed_1 = kwa.get('xhline_dashed1', None)
    if xhline_dashed_1 is not None:
        ax.axhline(xhline_dashed_1, color=label_colors,
                   linewidth=0.6, linestyle='--')

    xhline_dashed_2 = kwa.get('xhline_dashed2', None)
    if xhline_dashed_2 is not None:
        ax.axhline(xhline_dashed_2, color=label_colors,
                   linewidth=0.6, linestyle='--')

    xhline_dotted_1 = kwa.get('xhline_dotted1', None)
    if xhline_dotted_1 is not None:
        ax.axhline(xhline_dotted_1, color=label_colors,
                   linewidth=0.9, linestyle=':')

    xhline_dotted_2 = kwa.get('xhline_dotted2', None)
    if xhline_dotted_2 is not None:
        ax.axhline(xhline_dotted_2, color=label_colors,
                   linewidth=0.9, linestyle=':')

    main_spine = kwa.get('main_spine', 'left')
    fancy_design(ax, main_spine=main_spine)
    rotation = kwa.get('xtickrotation', 35)
    plt.setp(ax.get_xticklabels(), rotation=rotation)
    if kwa.get('disable_x_ticks', False):
        # Deactivates labels always for all shared axes
        labels = [
            item.get_text()
            for item in ax.get_xticklabels()
        ]
        ax.set_xticklabels([''] * len(labels))

    save = kwa.get('save', '')
    if save:
        plt.savefig(save, facecolor=fig.get_facecolor())

    if kwa.get('axis', None) is None and \
            kwa.get('show', True):
        plt.show()
    return fig, ax


def _head(kwargs, data=None):
    # Prepare data ------------------------------------------
    if data is not None:
        for col in list(data):
            data[col] = pd.to_numeric(
                data[col], errors='coerce')

    # Build ax ----------------------------------------------
    fig = kwargs.get('fig', None)
    if fig is None:
        fig, _ = plt.subplots(facecolor=background_color)

    ax = kwargs.get('axis', None)
    if ax is None:
        ax = plt.subplot2grid(
            (4, 4), (0, 0),
            rowspan=4, colspan=4,
            facecolor=background_color
        )
    return fig, ax


def _signal_eval(ax, signals, kwargs):
    """
    Plots the signals
    :param ax: Axis
    :param signals: List of patterns with structure:
        [ ..., ['signal', index, price], ...], where
        signal can be either 'BUY' or 'SELL'
    :param kwargs:
        'draw_verticals': Plots vertical lines
            for each BUY and SELL
        'signl_evaluation': Plot signals
        'signl_evaluation_form': 'rectangles' or
            'arrows_1'
        'dots': Plot dots at 'BUY' and 'SELL' points
    :return:
    """
    if signals is not None:
        if kwargs.get('draw_verticals', True):
            draw_verticals(axis=ax, signals=signals)
        if kwargs.get('signal_evaluation', True):
            draw_signal_evaluation(
                axis=ax,
                signals=signals,
                eval_type=kwargs.get(
                    'signal_evaluation_form',
                    'rectangle'),
                dots=kwargs.get('dots', True),
                red=red,
                green=green,
                disable_red_signals=kwargs.get(
                    'disable_red_signals', False),
                disable_green_signals=kwargs.get(
                    'disable_green_signals', False)
            )


def _pattern_eval(data, ax, cs_patterns, kwargs):
    """
    Plots the candlestick patterns
    :param data: Data
    :param ax: Axis
    :param cs_patterns: List of patterns with structure:
        [ ..., ['pattern_name', start_index,
            stop_index], ...]
    :param kwargs:
        'cs_pattern_evaluation': Enable plotting
    :return:
    """
    if cs_patterns is not None:
        if kwargs.get('cs_pattern_evaluation', True):
            df = data[['open', 'high', 'low', 'close']]
            draw_pattern_evaluation(
                axis=ax,
                data_ohlc=df,
                cs_patterns=cs_patterns,
                red=red,
                green=green
            )


def fancy_design(axis, main_spine='left'):
    """
    This function changes the design for
        - the legend
        - spines
        - ticks
        - grid
    :param axis: Axis
    """
    legend = axis.legend(
        loc='best', fancybox=True, framealpha=0.3
    )

    legend.get_frame().set_facecolor(background_color)
    legend.get_frame().set_edgecolor(label_colors)

    for line, text in zip(legend.get_lines(),
                          legend.get_texts()):
        text.set_color(line.get_color())

    axis.grid(linestyle='dotted',
              color=label_colors, alpha=0.7)
    axis.yaxis.label.set_color(label_colors)
    axis.xaxis.label.set_color(label_colors)
    axis.yaxis.label.set_color(label_colors)
    for spine in axis.spines:
        if spine == main_spine:
            axis.spines[spine].set_color(label_colors)
        else:
            axis.spines[spine].set_color(background_color)
    axis.tick_params(
        axis='y', colors=label_colors,
        which='major', labelsize=10,
        direction='in', length=2,
        width=1
    )

    axis.tick_params(
        axis='x', colors=label_colors,
        which='major', labelsize=10,
        direction='in', length=2,
        width=1
    )


def add_price_flag(fig, axis, series, color, last_index=None):
    """
    Add a price flag at the end of the data
    series in the chart
    :param fig: Figure
    :param axis: Axis
    :param series: Pandas Series
    :param color: Color of the flag
    :param last_index: Last index
    """

    series = series.dropna()
    value = series.tail(1)

    index = value.index.tolist()[0]
    if last_index is not None:
        axis.plot(
            [index, last_index], [value.values[0], value.values[0]],
            color=color, linewidth=0.6, linestyle='--', alpha=0.6
        )
    else:
        last_index = index

    trans_offset = mtrans.offset_copy(
        axis.transData, fig=fig,
        x=0.05, y=0.0, units='inches'
    )

    # Add price text box for candlestick
    value_clean = format(value.values[0], '.6f')
    axis.text(
        last_index, value.values, value_clean,
        size=7, va="center", ha="left",
        transform=trans_offset,
        color='white',
        bbox=dict(
            boxstyle="angled,pad=0.2",
            alpha=0.6, color=color
        )
    )


def plot_candlestick(
        data, signals=None, cs_patterns=None,
        plot_columns=None, **kwargs):
    """
    This function plots a candlestick chart
    :param data: Pandas DataFrame
    :param signals: List of signals with structure
        [(signal, index, price), ... ]. Signal can be 'BUY'
        or 'SELL'
    :param cs_patterns: List of candlestick patterns with structure
    patterns = [... , ['pattern_name', start_index, stop_index], ... ]
    :param plot_columns: List of columns in the given DataFrame like
        plot_columns=['bband_upper_20', 'bband_lower_20']
    :param kwargs:
        'fig': Figure.
        'axis': Axis. If axis is not given the chart will
            plt.plot automatically
        'name': Name of the chart
        'draw_verticals': plots vertical lines for each BUY and SELL
        'signl_evaluation': plot signals
        'signl_evaluation_form': 'rectangles' or 'arrows_1'
        'disable_red_signals': Disables red signals if True
        'disable_green_signals': Disables red signals if True
        'cs_pattern_evaluation': plot candlestick pattern
        'dots': Plot dots at 'BUY' and 'SELL' points
        'enable_flags': Enable flags
        'set_flags_at_the_end': Set flags at the end of the chart
        'xhline1': Normal horizontal line 1
        'xhline2': Normal horizontal line 1
        'xhline_red': Red horizontal line
        'xhline_green': Green horizontal line
        'xhline_dashed1': Dashed horizontal line 1
        'xhline_dashed2': Dashed horizontal line 2
        'xhline_dotted1': Dotted horizontal line 1
        'xhline_dotted2': Dotted horizontal line 2
        'vline': Index of vline
        'vspan': [start index, end index]
        'xtickrotation': Angle of the x ticks
        'xlabel': x label
        'ylabel': x label
        'title': title
        'disable_x_ticks': Disables the x ticks
        'show': If true the chart will be plt.show'd
        'save': Save the image to a specified path like
            save='path_to_picture.png'
    :return: fig, ax
    """
    fig, ax = _head(kwargs=kwargs, data=data)

    # Add candlestick
    _candlestick2_ohlc(
        ax,
        data['open'], data['high'],
        data['low'], data['close'],
        width=0.6,
        colorup=green,
        colordown=red,
        alpha=1
    )

    _signal_eval(ax, signals, kwargs)
    _pattern_eval(data, ax, cs_patterns, kwargs)

    return _tail(
        fig=fig,
        ax=ax,
        kwa=kwargs,
        data=data,
        plot_columns=plot_columns
    )


def plot_filled_ohlc(
        data, signals=None, cs_patterns=None,
        plot_columns=None, **kwargs):
    """
    This function plots a filled ohlc chart
    :param data: Pandas DataFrame
    :param signals: List of signals with structure
        [(signal, index, price), ... ]. Signal can be 'BUY'
        or 'SELL'
    :param cs_patterns: List of candlestick patterns with structure
    patterns = [... , ['pattern_name', start_index, stop_index], ... ]
    :param plot_columns: List of columns in the given DataFrame like
        plot_columns=['bband_upper_20', 'bband_lower_20']
    :param kwargs:
        'fig': Figure.
        'axis': Axis. If axis is not given the chart will
            plt.plot automatically
        'name': Name of the chart
        'draw_verticals': plots vertical lines for each BUY and SELL
        'signl_evaluation': plot signals
        'signl_evaluation_form': 'rectangles' or 'arrows_1'
        'disable_red_signals': Disables red signals if True
        'disable_green_signals': Disables red signals if True
        'cs_pattern_evaluation': plot candlestick pattern
        'dots': Plot dots at 'BUY' and 'SELL' points
        'enable_flags': Enable flags
        'set_flags_at_the_end': Set flags at the end of the chart
        'xhline1': Normal horizontal line 1
        'xhline2': Normal horizontal line 1
        'xhline_red': Red horizontal line
        'xhline_green': Green horizontal line
        'xhline_dashed1': Dashed horizontal line 1
        'xhline_dashed2': Dashed horizontal line 2
        'xhline_dotted1': Dotted horizontal line 1
        'xhline_dotted2': Dotted horizontal line 2
        'vline': Index of vline
        'vspan': [start index, end index]
        'xtickrotation': Angle of the x ticks
        'xlabel': x label
        'ylabel': x label
        'title': title
        'disable_x_ticks': Disables the x ticks
        'show': If true the chart will be plt.show'd
        'save': Save the image to a specified path like
            save='path_to_picture.png'
    :return: fig, ax
    """
    fig, ax = _head(kwargs=kwargs, data=data)

    # Add filled_ohlc
    ax.fill_between(
        data.index,
        data['close'],
        data['high'],
        where=data['close'] <= data['high'],
        facecolor=green,
        interpolate=True,
        alpha=0.35,
        edgecolor=green
    )
    ax.fill_between(
        data.index,
        data['close'],
        data['low'],
        where=data['low'] <= data['close'],
        facecolor=red,
        interpolate=True,
        alpha=0.35,
        edgecolor=red
    )

    _signal_eval(ax, signals, kwargs)
    _pattern_eval(data, ax, cs_patterns, kwargs)

    return _tail(
        fig=fig,
        ax=ax,
        kwa=kwargs,
        data=data,
        plot_columns=plot_columns
    )


def plot(data, plot_columns, **kwargs):
    """
    This function provides a simple way to plot time series
    for example data['close'].
    :param data: Pandas DataFrame object
    :param plot_columns: Name of the columns to plot
    :param kwargs:
        'fig': Figure.
        'axis': Axis. If axis is not given the chart will
            plt.plot automatically
        'name': Name of the chart
        'enable_flags': Enable flags
        'set_flags_at_the_end': Set flags at the end of the chart
        'xhline1': Normal horizontal line 1
        'xhline2': Normal horizontal line 1
        'xhline_red': Red horizontal line
        'xhline_green': Green horizontal line
        'xhline_dashed1': Dashed horizontal line 1
        'xhline_dashed2': Dashed horizontal line 2
        'xhline_dotted1': Dotted horizontal line 1
        'xhline_dotted2': Dotted horizontal line 2
        'vline': Index of vline
        'vspan': [start index, end index]
        'xlabel': x label
        'ylabel': x label
        'title': title
        'disable_x_ticks': Disables the x ticks
        'show': If true the chart will be plt.show'd
        'save': Save the image to a specified path like
            save='path_to_picture.png'
    :return: fig, ax
    """
    fig, ax = _head(kwargs=kwargs, data=data)

    return _tail(
        fig=fig,
        ax=ax,
        kwa=kwargs,
        data=data,
        plot_columns=plot_columns
    )


def bars_from_dict(data_dict, **kwargs):
    """
    This function provides a simple way to plot a histogram
    from a dict.
    :param data_dict: Dictionary of the data
        Structure: {'key_1', count of key_1, ... }
    :param kwargs:
        'fig': Figure.
        'axis': Axis. If axis is not given the chart will
            plt.plot automatically
        'name': Name of the chart
        'xlabel': x label
        'ylabel': x label
        'title': Title
        'disable_x_ticks': Disables the x ticks
        'show': If true the chart will be plt.show'd
        'save': Save the image to a specified path like
            save='path_to_picture.png'
    :return: fig, ax
    """
    # prepare data
    x = dict()
    for candle in data_dict:
        if candle in x:
            x[candle] += 1
        else:
            x[candle] = 1
    objects = list(x.keys())

    y_pos = np.arange(len(objects))
    performance = list()
    for key in x:
        performance.append(x[key])

    # Generate chart
    fig, ax = _head(kwargs=kwargs)

    # Plot histogram
    ax.barh(
        y_pos, performance,
        align='center', alpha=0.5,
        color=green
    )

    plt.yticks(y_pos, objects)

    return _tail(
        fig=fig,
        ax=ax,
        kwa=kwargs
    )


def hist_from_dict(data_dict, **kwargs):
    """
    This function provides a simple way to plot a histogram
    from a dict.
    :param data_dict: Dictionary of the data
        Structure: {'key_1', count of key_1, ... }
    :param kwargs:
        'fig': Figure.
        'axis': Axis. If axis is not given the chart will
            plt.plot automatically
        'bins': Bins
        'density': Density
        'threshold': Threshold of the values
        'name': Name of the chart
        'xlabel': x label
        'ylabel': x label
        'title': Title
        'disable_x_ticks': Disables the x ticks
        'show': If true the chart will be plt.show'd
        'save': Save the image to a specified path like
            save='path_to_picture.png'
    :return: fig, ax
    """
    # Generate chart
    fig, ax = _head(kwargs=kwargs)

    # Plot histogram
    bins = kwargs.get('bins', 10)
    density = kwargs.get('density', None)
    ax.hist(
        data_dict, bins, density=density,
        facecolor=green, alpha=0.75,
        align='mid', histtype='bar',
        rwidth=0.9
    )

    # Plot box
    threshold = kwargs.get('threshold', None)
    if threshold is not None:
        count_pos = 0
        count_neg = 0
        for da in data_dict:
            if da > 0:
                count_pos += 1
            else:
                count_neg += 1

        box_text = '<={}: {} \n>{}: {}'.format(
            threshold, count_neg,
            threshold, count_pos)
        _add_text_box(
            fig=fig, axis=ax, text=box_text,
            x_p=80, y_p=90)

    kwargs['main_spine'] = 'bottom'

    return _tail(
        fig=fig,
        ax=ax,
        kwa=kwargs,
    )


def plot_vline(axis, index, linestyle='--', color=color_set[0]):
    """
    Plots a vertical line
    :param axis: Axis
    :param index: Index
    :param linestyle: Can be '-', '--', '-.', ':'
    :param color: Color
    """
    axis.axvline(
        index, color=color,
        linewidth=0.8, alpha=0.8, linestyle=linestyle
    )


def plot_vspan(axis, index, color=color_set[0], alpha=0.05):
    """
    Plots a vertical span
    :param axis: Axis
    :param index: [start index, end index]
    :param color: Color
    :param alpha: Alpha
    :return:
    """
    axis.axvspan(
        index[0], index[1],
        facecolor=color,
        alpha=alpha
    )
