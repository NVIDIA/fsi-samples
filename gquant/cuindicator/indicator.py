from gquant.cuindicator.util import shift, diff
from gquant.cuindicator.rolling import Rolling
from gquant.cuindicator.ewm import Ewm
from gquant.cuindicator.pewm import PEwm
import cudf
import collections
import math
import numba
from gquant.cuindicator.util import (substract, summation, multiply,
                                     division, upDownMove, abs_arr,
                                     true_range, lowhigh_diff, money_flow,
                                     average_price, onbalance_volume,
                                     ultimate_osc, scale)


def moving_average(close_arr, n):
    """Calculate the moving average for the given data.

    :param close_arr: close price of the bar, expect series from cudf
    :param n: time steps
    :return: moving average in cu.Series
    """
    MA = Rolling(n, close_arr).mean()
    return cudf.Series(MA)


def exponential_moving_average(close_arr, n):
    """Calculate the exponential weighted moving average for the given data.

    :param close_arr: close price of the bar, expect series from cudf
    :param n: time steps
    :return: expoential weighted moving average in cu.Series
    """
    EMA = Ewm(n, close_arr).mean()
    return cudf.Series(EMA)


def port_exponential_moving_average(asset_indicator, close_arr, n):
    """Calculate the exponential weighted moving average for the given data.

    :param close_arr: close price of the bar, expect series from cudf
    :param n: time steps
    :return: expoential weighted moving average in cu.Series
    """
    EMA = PEwm(n, close_arr, asset_indicator).mean()
    return cudf.Series(EMA)


def momentum(close_arr, n):
    """Calculate the momentum for the given data.

    :param close_arr: close price of the bar, expect series from cudf
    :param n: time steps
    :return: momentum in cu.Series
    """
    return cudf.Series(diff(close_arr, n))


def rate_of_change(close_arr, n):
    """ Calculate the rate of return

    :param close_arr: close price of the bar, expect series from cudf
    :param n: time steps
    :return: rate of change in cu.Series
    """
    M = diff(close_arr, n - 1)
    N = shift(close_arr, n - 1)
    return cudf.Series(division(M, N))


def bollinger_bands(close_arr, n):
    """Calculate the Bollinger Bands.
    See https://www.investopedia.com/terms/b/bollingerbands.asp for details

    :param close_arr: close price of the bar, expect series from cudf
    :param n: time steps
    :return: b1 b2
    """
    MA = Rolling(n, close_arr).mean()
    MSD = Rolling(n, close_arr).std()
    close_arr_gpu = numba.cuda.device_array_like(close_arr.data.to_gpu_array())
    close_arr_gpu[:] = close_arr.data.to_gpu_array()[:]
    close_arr_gpu[0:n-1] = math.nan
    MSD_4 = scale(MSD, 4.0)
    b1 = division(MSD_4, MA)
    b2 = division(summation(substract(close_arr_gpu, MA), scale(MSD, 2.0)),
                  MSD_4)
    out = collections.namedtuple('Bollinger', 'b1 b2')
    return out(b1=cudf.Series(b1), b2=cudf.Series(b2))


def trix(close_arr, n):
    """Calculate TRIX for given data.

    :param close_arr: close price of the bar, expect series from cudf
    :param n: time steps
    :return: trix indicator in cudf.Series
    """
    EX1 = Ewm(n, close_arr).mean()
    EX2 = Ewm(n, EX1).mean()
    EX3 = Ewm(n, EX2).mean()
    return rate_of_change(cudf.Series(EX3), 2)


def macd(close_arr, n_fast, n_slow):
    """Calculate MACD, MACD Signal and MACD difference

    :param close_arr: close price of the bar, expect series from cudf
    :param n_fast: fast time steps
    :param n_slow: slow time steps
    :return: MACD MACDsign MACDdiff
    """
    EMAfast = Ewm(n_fast, close_arr).mean()
    EMAslow = Ewm(n_slow, close_arr).mean()
    MACD = substract(EMAfast, EMAslow)
    average_window = 9
    MACDsign = Ewm(average_window, MACD).mean()
    MACDdiff = substract(MACD, MACDsign)
    out = collections.namedtuple('MACD', 'MACD MACDsign MACDdiff')
    return out(MACD=cudf.Series(MACD), MACDsign=cudf.Series(MACDsign),
               MACDdiff=cudf.Series(MACDdiff))


def port_macd(asset_indicator, close_arr, n_fast, n_slow):
    """Calculate MACD, MACD Signal and MACD difference

    :param close_arr: close price of the bar, expect series from cudf
    :param n_fast: fast time steps
    :param n_slow: slow time steps
    :return: MACD MACDsign MACDdiff
    """
    EMAfast = PEwm(n_fast, close_arr, asset_indicator).mean()
    EMAslow = PEwm(n_slow, close_arr, asset_indicator).mean()
    MACD = substract(EMAfast, EMAslow)
    average_window = 9
    MACDsign = PEwm(average_window, MACD, asset_indicator).mean()
    MACDdiff = substract(MACD, MACDsign)
    out = collections.namedtuple('MACD', 'MACD MACDsign MACDdiff')
    return out(MACD=cudf.Series(MACD), MACDsign=cudf.Series(MACDsign),
               MACDdiff=cudf.Series(MACDdiff))


def average_true_range(high_arr, low_arr, close_arr, n):
    """Calculate the Average True Range
    See https://www.investopedia.com/terms/a/atr.asp for details

    :param high_arr: high price of the bar, expect series from cudf
    :param low_arr: low price of the bar, expect series from cudf
    :param close_arr: close price of the bar, expect series from cudf
    :param n: time steps
    :return: average true range indicator
    """
    tr = true_range(high_arr.data.to_gpu_array(), low_arr.data.to_gpu_array(),
                    close_arr.data.to_gpu_array())
    ATR = Ewm(n, tr).mean()
    return cudf.Series(ATR)


def ppsr(high_arr, low_arr, close_arr):
    """Calculate Pivot Points, Supports and Resistances for given data

    :param high_arr: high price of the bar, expect series from cudf
    :param low_arr: low price of the bar, expect series from cudf
    :param close_arr: close price of the bar, expect series from cudf
    :return: PP R1 S1 R2 S2 R3 S3
    """
    high_gpu = high_arr.data.to_gpu_array()
    low_gpu = low_arr.data.to_gpu_array()
    close_gpu = close_arr.data.to_gpu_array()
    PP = average_price(high_gpu, low_gpu, close_gpu)
    R1 = substract(scale(PP, 2.0), low_gpu)
    S1 = substract(scale(PP, 2.0), high_gpu)
    R2 = substract(summation(PP, high_gpu), low_gpu)
    S2 = summation(substract(PP, high_gpu), low_gpu)
    R3 = summation(high_gpu, scale(substract(PP, low_gpu), 2.0))
    S3 = substract(low_gpu, scale(substract(high_gpu, PP), 2.0))
    out = collections.namedtuple('PPSR', 'PP R1 S1 R2 S2 R3 S3')
    return out(PP=cudf.Series(PP),
               R1=cudf.Series(R1),
               S1=cudf.Series(S1),
               R2=cudf.Series(R2),
               S2=cudf.Series(S2),
               R3=cudf.Series(R3),
               S3=cudf.Series(S3))


def stochastic_oscillator_k(high_arr, low_arr, close_arr):
    """Calculate stochastic oscillator K for given data.

    :param high_arr: high price of the bar, expect series from cudf
    :param low_arr: low price of the bar, expect series from cudf
    :param close_arr: close price of the bar, expect series from cudf
    :return: stochastic oscillator K in cudf.Series
    """
    SOk = (close_arr - low_arr) / (high_arr - low_arr)
    return SOk


def stochastic_oscillator_d(high_arr, low_arr, close_arr, n):
    """Calculate stochastic oscillator D for given data.

    :param high_arr: high price of the bar, expect series from cudf
    :param low_arr: low price of the bar, expect series from cudf
    :param close_arr: close price of the bar, expect series from cudf
    :param n: time steps
    :return: stochastic oscillator D in cudf.Series
    """
    SOk = stochastic_oscillator_k(high_arr, low_arr, close_arr)
    SOd = Ewm(n, SOk).mean()
    return cudf.Series(SOd)


def average_directional_movement_index(high_arr, low_arr, close_arr, n, n_ADX):
    """Calculate the Average Directional Movement Index for given data.

    :param high_arr: high price of the bar, expect series from cudf
    :param low_arr: low price of the bar, expect series from cudf
    :param close_arr: close price of the bar, expect series from cudf
    :param n: time steps to do EWM average
    :param n_ADX: time steps to do EWM average of ADX
    :return: Average Directional Movement Index in cudf.Series
    """
    UpI, DoI = upDownMove(high_arr.data.to_gpu_array(),
                          low_arr.data.to_gpu_array())
    last_ele = len(high_arr) - 1
    tr = true_range(high_arr.data.to_gpu_array(), low_arr.data.to_gpu_array(),
                    close_arr.data.to_gpu_array())
    ATR = Ewm(n, tr).mean()
    PosDI = division(Ewm(n, UpI).mean(), ATR)
    NegDI = division(Ewm(n, DoI).mean(), ATR)
    NORM = division(abs_arr(substract(PosDI, NegDI)), summation(PosDI, NegDI))
    NORM[last_ele] = math.nan
    ADX = cudf.Series(Ewm(n_ADX, NORM).mean())
    return ADX


def vortex_indicator(high_arr, low_arr, close_arr, n):
    """Calculate the Vortex Indicator for given data.
    Vortex Indicator described here:

        http://www.vortexindicator.com/VFX_VORTEX.PDF

    :param high_arr: high price of the bar, expect series from cudf
    :param low_arr: low price of the bar, expect series from cudf
    :param close_arr: close price of the bar, expect series from cudf
    :param n: time steps to do EWM average
    :return:  Vortex Indicator in cudf.Series
    """
    TR = true_range(high_arr.data.to_gpu_array(), low_arr.data.to_gpu_array(),
                    close_arr.data.to_gpu_array())

    VM = lowhigh_diff(high_arr.data.to_gpu_array(),
                      low_arr.data.to_gpu_array())

    VI = division(Rolling(n, VM).sum(), Rolling(n, TR).sum())
    return cudf.Series(VI)


def kst_oscillator(close_arr, r1, r2, r3, r4, n1, n2, n3, n4):
    """Calculate KST Oscillator for given data.

    :param close_arr: close price of the bar, expect series from cudf
    :param r1: r1 time steps
    :param r2: r2 time steps
    :param r3: r3 time steps
    :param r4: r4 time steps
    :param n1: n1 time steps
    :param n2: n2 time steps
    :param n3: n3 time steps
    :param n4: n4 time steps
    :return:  KST Oscillator in cudf.Series
    """
    M1 = diff(close_arr, r1 - 1)
    N1 = shift(close_arr, r1 - 1)
    M2 = diff(close_arr, r2 - 1)
    N2 = shift(close_arr, r2 - 1)
    M3 = diff(close_arr, r3 - 1)
    N3 = shift(close_arr, r3 - 1)
    M4 = diff(close_arr, r4 - 1)
    N4 = shift(close_arr, r4 - 1)
    term1 = Rolling(n1, division(M1, N1)).sum()
    term2 = scale(Rolling(n2, division(M2, N2)).sum(), 2.0)
    term3 = scale(Rolling(n3, division(M3, N3)).sum(), 3.0)
    term4 = scale(Rolling(n4, division(M4, N4)).sum(), 4.0)
    KST = summation(summation(summation(term1, term2), term3), term4)
    return cudf.Series(KST)


def relative_strength_index(high_arr, low_arr, n):
    """Calculate Relative Strength Index(RSI) for given data.

    :param high_arr: high price of the bar, expect series from cudf
    :param low_arr: low price of the bar, expect series from cudf
    :param n: time steps to do EWM average
    :return: Relative Strength Index in cudf.Series
    """
    UpI, DoI = upDownMove(high_arr.data.to_gpu_array(),
                          low_arr.data.to_gpu_array())
    UpI_s = shift(UpI, 1)
    UpI_s[0] = 0
    DoI_s = shift(DoI, 1)
    DoI_s[0] = 0
    PosDI = Ewm(n, UpI_s).mean()
    NegDI = Ewm(n, DoI_s).mean()
    RSI = division(PosDI, summation(PosDI, NegDI))
    return cudf.Series(RSI)


def port_relative_strength_index(asset_indicator, high_arr, low_arr, n):
    """Calculate Relative Strength Index(RSI) for given data.

    :param high_arr: high price of the bar, expect series from cudf
    :param low_arr: low price of the bar, expect series from cudf
    :param n: time steps to do EWM average
    :return: Relative Strength Index in cudf.Series
    """
    UpI, DoI = upDownMove(high_arr.data.to_gpu_array(),
                          low_arr.data.to_gpu_array())
    UpI_s = shift(UpI, 1)
    UpI_s[0] = 0
    UpI_s = cudf.Series(UpI_s) * (1.0 - asset_indicator)
    DoI_s = shift(DoI, 1)
    DoI_s[0] = 0
    DoI_s = cudf.Series(DoI_s) * (1.0 - asset_indicator)
    PosDI = PEwm(n, UpI_s, asset_indicator).mean()
    NegDI = PEwm(n, DoI_s, asset_indicator).mean()
    RSI = division(PosDI, summation(PosDI, NegDI))
    return cudf.Series(RSI)


def mass_index(high_arr, low_arr, n1, n2):
    """Calculate the Mass Index for given data.

    :param high_arr: high price of the bar, expect series from cudf
    :param low_arr: low price of the bar, expect series from cudf
    :param n1: n1 time steps
    :param n1: n2 time steps
    :return: Mass Index in cudf.Series
    """
    Range = high_arr - low_arr
    EX1 = Ewm(n1, Range).mean()
    EX2 = Ewm(n1, EX1).mean()
    Mass = division(EX1, EX2)
    MassI = Rolling(n2, Mass).sum()
    return cudf.Series(MassI)


def true_strength_index(close_arr, r, s):
    """Calculate True Strength Index (TSI) for given data.

    :param close_arr: close price of the bar, expect series from cudf
    :param r: r time steps
    :param s: s time steps
    :return: True Strength Index in cudf.Series
    """
    M = diff(close_arr, 1)
    aM = abs_arr(M)
    EMA1 = Ewm(r, M).mean()
    aEMA1 = Ewm(r, aM).mean()
    EMA2 = Ewm(s, EMA1).mean()
    aEMA2 = Ewm(s, aEMA1).mean()
    TSI = division(EMA2, aEMA2)
    return cudf.Series(TSI)


def chaikin_oscillator(high_arr, low_arr, close_arr, volume_arr, n1, n2):
    """Calculate Chaikin Oscillator for given data.

    :param high_arr: high price of the bar, expect series from cudf
    :param low_arr: low price of the bar, expect series from cudf
    :param close_arr: close price of the bar, expect series from cudf
    :param volume_arr: volume the bar, expect series from cudf
    :param n1: n1 time steps
    :param n2: n2 time steps
    :return: Chaikin Oscillator indicator in cudf.Series
    """
    ad = (2.0 * close_arr - high_arr - low_arr) / (
        high_arr - low_arr) * volume_arr
    Chaikin = cudf.Series(Ewm(n1, ad).mean()) - cudf.Series(Ewm(n2, ad).mean())
    return Chaikin


def money_flow_index(high_arr, low_arr, close_arr, volume_arr, n):
    """Calculate Money Flow Index and Ratio for given data.

    :param high_arr: high price of the bar, expect series from cudf
    :param low_arr: low price of the bar, expect series from cudf
    :param close_arr: close price of the bar, expect series from cudf
    :param volume_arr: volume the bar, expect series from cudf
    :param n: time steps
    :return: Money Flow Index in cudf.Series
    """
    PP = average_price(high_arr.data.to_gpu_array(),
                       low_arr.data.to_gpu_array(),
                       close_arr.data.to_gpu_array())

    PosMF = money_flow(PP, volume_arr.data.to_gpu_array())
    MFR = division(PosMF,
                   (multiply(PP, volume_arr.data.to_gpu_array())))  # TotMF
    MFI = Rolling(n, MFR).mean()
    return cudf.Series(MFI)


def on_balance_volume(close_arr, volume_arr, n):
    """Calculate On-Balance Volume for given data.

    :param close_arr: close price of the bar, expect series from cudf
    :param volume_arr: volume the bar, expect series from cudf
    :param n: time steps
    :return: On-Balance Volume in cudf.Series
    """
    OBV = onbalance_volume(close_arr.data.to_gpu_array(),
                           volume_arr.data.to_gpu_array())
    OBV_ma = Rolling(n, OBV).mean()
    return cudf.Series(OBV_ma)


def force_index(close_arr, volume_arr, n):
    """Calculate Force Index for given data.

    :param close_arr: close price of the bar, expect series from cudf
    :param volume_arr: volume the bar, expect series from cudf
    :param n: time steps
    :return: Force Index in cudf.Series
    """
    F = multiply(diff(close_arr, n), diff(volume_arr, n))
    return cudf.Series(F)


def ease_of_movement(high_arr, low_arr, volume_arr, n):
    """Calculate Ease of Movement for given data.

    :param high_arr: high price of the bar, expect series from cudf
    :param low_arr: low price of the bar, expect series from cudf
    :param volume_arr: volume the bar, expect series from cudf
    :param n: time steps
    :return: Ease of Movement in cudf.Series
    """
    high_arr_gpu = high_arr.data.to_gpu_array()
    low_arr_gpu = low_arr.data.to_gpu_array()

    EoM = division(multiply(summation(diff(high_arr_gpu, 1),
                                      diff(low_arr_gpu, 1)),
                            substract(high_arr_gpu, low_arr_gpu)),
                   scale(volume_arr.data.to_gpu_array(), 2.0))
    Eom_ma = Rolling(n, EoM).mean()
    return cudf.Series(Eom_ma)


def ultimate_oscillator(high_arr, low_arr, close_arr):
    """Calculate Ultimate Oscillator for given data.

    :param high_arr: high price of the bar, expect series from cudf
    :param low_arr: low price of the bar, expect series from cudf
    :param close_arr: close price of the bar, expect series from cudf
    :return: Ultimate Oscillator in cudf.Series
    """
    TR_l, BP_l = ultimate_osc(high_arr.data.to_gpu_array(),
                              low_arr.data.to_gpu_array(),
                              close_arr.data.to_gpu_array())
    term1 = division(scale(Rolling(7, BP_l).sum(), 4.0),
                     Rolling(7, TR_l).sum())
    term2 = division(scale(Rolling(14, BP_l).sum(), 2.0),
                     Rolling(14, TR_l).sum())
    term3 = division(Rolling(28, BP_l).sum(), Rolling(28, TR_l).sum())
    UltO = summation(summation(term1, term2), term3)
    return cudf.Series(UltO)


def donchian_channel(high_arr, low_arr, n):
    """Calculate donchian channel of given pandas data frame.

    :param high_arr: high price of the bar, expect series from cudf
    :param low_arr: low price of the bar, expect series from cudf
    :param n: time steps
    :return: donchian channel in cudf.Series
    """
    max_high = Rolling(n, high_arr).max()
    min_low = Rolling(n, low_arr).min()
    dc_l = substract(max_high, min_low)
    dc_l[:n-1] = 0.0
    donchian_chan = shift(dc_l, n - 1)
    return cudf.Series(donchian_chan)


def keltner_channel(high_arr, low_arr, close_arr, n):
    """Calculate Keltner Channel for given data.

    :param high_arr: high price of the bar, expect series from cudf
    :param low_arr: low price of the bar, expect series from cudf
    :param close_arr: close price of the bar, expect series from cudf
    :param n: time steps
    :return: Keltner Channel in cudf.Series
    """
    M = ((high_arr + low_arr + close_arr) / 3.0)
    KelChM = cudf.Series(Rolling(n, M).mean())
    U = ((4.0 * high_arr - 2.0 * low_arr + close_arr) / 3.0)
    KelChU = cudf.Series(Rolling(n, U).mean())
    D = ((-2.0 * high_arr + 4.0 * low_arr + close_arr) / 3.0)
    KelChD = cudf.Series(Rolling(n, D).mean())
    out = collections.namedtuple('Keltner', 'KelChM KelChU KelChD')
    return out(KelChM=KelChM, KelChU=KelChU, KelChD=KelChD)


def coppock_curve(close_arr, n):
    """Calculate Coppock Curve for given data.

    :param close_arr: close price of the bar, expect series from cudf
    :param n: time steps
    :return: Coppock Curve in cudf.Series
    """
    M = diff(close_arr, int(n * 11 / 10) - 1)
    N = shift(close_arr, int(n * 11 / 10) - 1)
    ROC1 = division(M, N)
    M = diff(close_arr, int(n * 14 / 10) - 1)
    N = shift(close_arr, int(n * 14 / 10) - 1)
    ROC2 = division(M, N)
    Copp = Ewm(n, summation(ROC1, ROC2)).mean()
    return cudf.Series(Copp)


def accumulation_distribution(high_arr, low_arr, close_arr, vol_arr, n):
    """Calculate Accumulation/Distribution for given data.

    :param high_arr: high price of the bar, expect series from cudf
    :param low_arr: low price of the bar, expect series from cudf
    :param close_arr: close price of the bar, expect series from cudf
    :param vol_arr: volume of the bar, expect series from cudf
    :param n: time steps
    :return: Accumulation/Distribution in cudf.Series
    """
    ad = (2.0 * close_arr - high_arr - low_arr)/(high_arr - low_arr) * vol_arr
    M = diff(ad, n-1)
    N = shift(ad, n-1)
    return cudf.Series(division(M, N))


def commodity_channel_index(high_arr, low_arr, close_arr, n):
    """Calculate Commodity Channel Index for given data.

    :param high_arr: high price of the bar, expect series from cudf
    :param low_arr: low price of the bar, expect series from cudf
    :param close_arr: close price of the bar, expect series from cudf
    :param n: time steps
    :return: Commodity Channel Index in cudf.Series
    """
    PP = average_price(high_arr.data.to_gpu_array(),
                       low_arr.data.to_gpu_array(),
                       close_arr.data.to_gpu_array())
    M = Rolling(n, PP).mean()
    N = Rolling(n, PP).std()
    CCI = division(substract(PP, M), N)
    return cudf.Series(CCI)
