#
#TFBLS - software
#
#Copyright (C) 2026 Tymoteusz Braciszewski <tymoteusz.braciszewski@amu.edu.pl>
#
#
'''
------------
TFBLS v. 1.0
------------

Time-Frequency Brillouin Light Scattering analysis tools,
developed by:
    - Tymoteusz Braciszewski <tymoteusz.braciszewksi@amu.edu.pl>
    - Mikołaj Pochylski <mikolaj.pochylski@amu.edu.pl>
    
Contact e-mail: <tymoteusz.braciszewksi@amu.edu.pl>


Parts of the pt3_reader and process_t3records functions are based on the
phconvert library by Antonino Ingargiola <tritemio@gmail.com> in
compliance with the MIT License under which it was published:
https://photon-hdf5.github.io/phconvert/

'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.signal import fftconvolve

import os

from fractions import Fraction

import time as TIME

from math import log10, floor
def round_to_3(x):
    return round(x, 3-int(floor(log10(abs(x)))))

has_numba = True
try:
    import numba
except ImportError:
    print("You don't have numba. Please, install numba for better performance.")
    print("Without numba, the code will run significantly slower.")
    has_numba = False


c = 299.792458 ## in mm/ns


##########################################################################
##############################PLOT  FUNCTIONS#############################
##########################################################################

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['red',
                                                                'gold',
                                                                '#b7b82a',
                                                                'tab:green',
                                                                'green',
                                                                'darkgreen'])


def ToInch(array):
    '''
    Translates from cm to inches, the default matplotlib unit.
    '''
    return [a/2.54 for a in array]

def CustomizeAxes(ax,
                  grid = False,
                  legend = False, legend_title = None,
                  title = None,
                  xlabel = None, ylabel = None,
                  xlim = None, ylim = None,
                  xticks = None, xticklabels = None,
                  yticks = None, yticklabels = None,
                  remove_minor_ticks = False,
                  xscale = 'linear', yscale = 'linear',
                  remove_axes = False):
    if legend:
        ax.legend(title = legend_title)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(visible=None, 
                which='major', 
                axis='both', 
                linestyle='--',
                color="grey",
                alpha=0.5,
                zorder = -10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if yscale != 'linear':
        ax.set_yscale(yscale)
    if xscale != 'linear':
        ax.set_xscale(xscale)
    if xticks is not None:
        if xticks == []:
            ax.xaxis.set_ticks_position('none')
            ax.xaxis.set_ticklabels([])
        else:
            if xticklabels == None:
                ax.set_xticks(xticks, [str(tick) for tick in xticks])
            else:
                ax.set_xticks(xticks, xticklabels)
    if yticks is not None:
        if yticks == []:
            ax.yaxis.set_ticks_position('none')
            ax.yaxis.set_ticklabels([])
        else:
            if yticklabels == None:
                ax.set_yticks(yticks, [str(tick) for tick in yticks])
            else:
                ax.set_yticks(yticks, yticklabels)
    if remove_minor_ticks:
        ax.minorticks_off()
    ax.set_title(title)
    if remove_axes:
        ax.spines[['top', 'right','bottom','left']].set_visible(False)
        ax.set_xticks([],[])
        ax.set_yticks([],[])

def CustomizeFig(fig,
                 windowsize = [14,10],
                 h_pad = 1.08, 
                 w_pad = 1.08,
                 pad = 1.08,
                 title = None):
    windowsize = ToInch(windowsize)
    fig.set_size_inches(windowsize)
    fig.suptitle(title)
    fig.tight_layout(pad = pad, h_pad = h_pad, w_pad = w_pad)


def CustomizePlot(fig, ax,
                  windowsize = [14,10],
                  cmap = cmap, cbar_show = False, cbar_label = r"$\tau$ (ns)", cbar_norm = [0,1],
                  grid = False,
                  legend = False, legend_title = None,
                  title = None, suptitle = None,
                  xlabel = None, ylabel = None,
                  xlim = None, ylim = None,
                  xticks = None, xticklabels = None,
                  yticks = None, yticklabels = None,
                  remove_minor_ticks = False,
                  yscale = 'linear', xscale = 'linear',
                  h_pad = 1.08, w_pad = 1.08, pad = 1.08,
                  remove_axes = False):
    '''
    Customization of plots. 
    '''
    
    if cbar_show == True:
        fig.colorbar(matplotlib.cm.ScalarMappable(norm = plt.Normalize(cbar_norm[0],cbar_norm[1]),
                                                  cmap = cmap),
                     ax = ax,
                     label = cbar_label)
    CustomizeAxes(ax,
                 grid = grid,
                 legend = legend, legend_title = legend_title,
                 title = title,
                 xlabel = xlabel, ylabel = ylabel,
                 xlim = xlim, ylim = ylim,
                 xticks = xticks, xticklabels = xticklabels,
                 yticks = yticks, yticklabels = yticklabels,
                 remove_minor_ticks = remove_minor_ticks,
                 xscale = xscale, yscale = yscale,
                 remove_axes = remove_axes)
    CustomizeFig(fig,
                  windowsize = windowsize,
                  h_pad = h_pad, w_pad = w_pad,
                  pad = pad,
                  title = suptitle)

##########################################################################
##########################MATHEMATICAL FUNCTIONS##########################
##########################################################################

def SingleFPI(frequency, timeOfFlight = 1, reflectance = 0.5):
    """
    Single-Pass Fabry-Perot transmission function.
    inputs:
        - frequency of the harmonic component 
        - timeOfFlight of light in the vacuum between mirrors 
        - intensity reflectance (R = |r|**2)
    outputs:
        - transmission through the Fabry-Perot etalon, as derived in the thesis, sans the overall phase shift.
    Units of frequency and time are not specified.
    """
    return (1-reflectance)/(1-reflectance*np.exp(-2j*2*np.pi*frequency*timeOfFlight))

def TandemFPI(frequency, 
             mirrorSeparation = 1, 
             reflectance = 0.93, 
             detuning = 0.95):
    """
    3+3 Tandem Fabry-Perot Interferometer transmission function.
    inputs:
        - frequency of the harmonic component 
        - mirrorSeparation in mm
        - intensity reflectance (R = |r|**2)
        - detuning of the second set of mirrors in the Tandem FPI.
    outputs:
        - transmission through the 3+3 Tandem FPI, sans the overall phase factor.
    """
    tau0 = mirrorSeparation/c
    A = SingleFPI(frequency, tau0, reflectance)
    B = SingleFPI(frequency, tau0*detuning, reflectance)
    return A**3*B**3

def Gauss(time, pulseWidth = 1, t0 = 0, amplitude = 1):
    """
    inputs:
        - time
        - pulseWidth of the gaussian signal
        - t0, the position of the peak
        - amplitude of the pulse
    outputs:
        - amp.*exp(-2(t-t0)^2/pulseWidth^2)
    """

    return amplitude*np.exp(-2*((time-t0)/pulseWidth)**2)

def Exp(time,lifetime = 1, t0 = 0, amplitude =1):
    """
    inputs:
        - time
        - lifetime of the exponential decay
        - t0, the start of the signal
        - amplitude of the pulse
    outputs:
        - amp.*exp(-(t-t_max)/lifetime) for t>t0, else 0.
    """
    if time >= t0:
        return amplitude*np.exp(-(time-t0)/lifetime)
    else: 
        return 0
Exp = np.vectorize(Exp, otypes = [np.float64])

def BiExp(time, 
          lifetime = 1, switchOnTime = 0.1,
          t0 = 0, amplitude = 1):
    """
    Bi-exponential decay with decaying (tau1) and growing component (tau2). 

    inputs:
        - time array
        - tau1 - relaxation/decay time constant
        - tau2 - growth time constant
        - t0 - maximum time position
        - amplitude - maximum value
    """
    tau1 = lifetime
    tau2 = switchOnTime
    decay = Exp(time,tau1,t0+tau2*np.log(tau2/(tau1+tau2)))
    growth = 1-Exp(time,tau2,t0+tau2*np.log(tau2/(tau1+tau2)))
    norm = amplitude/((tau1/(tau1+tau2))*(tau2/(tau1+tau2))**(tau2/tau1))
    return growth*decay*norm

def BiExpTwo(time, 
             lifetime = 1, switchOnTime = 0.1,
             t0 = 0, amplitude = 1):
    """
    Bi-exponential decay with decaying (tau1) and growing component (tau2).
    
    inputs:
        - time array
        - tau1 - relaxation/decay time constant
        - tau2 - growth time constant
        - t0 - maximum time position
        - amplitude - maximum value
    """
    tau1 = lifetime
    tau2 = switchOnTime
    decay = Exp(time,tau1,t0)
    growth = 1-Exp(time,tau2,t0)
    norm = amplitude/((tau1/(tau1+tau2))*(tau2/(tau1+tau2))**(tau2/tau1))
    return growth*decay*norm

def Phonon(t,f0,tau,
           phi = 0,
           A = 0.01,
           tau0 = 0.8,
           A0 = 1):
    return A0*Exp(t,tau0)+Exp(t,tau,0,A)*np.cos(2*np.pi*f0*t+phi)
##########################################################################
############################TRANSFORM METHODS#############################
##########################################################################

def FT(time, signal, 
       shift = False):
    """
    The discrete Fast Fourier Transform of the given signal.
    inputs:
        - time, signal - arrays with the signal to be transformed and time coordinates from which the frequencies are calculated.
        - shift - boolean, set to True it shifts the frequency array from the conventional (positive-first) to normal ordering.
    outputs:
        - frequency array, Fast Fourier Transform array
    """
    if shift:
        return fftshift(fftfreq(len(time), (time[len(time)-1]-time[0])/len(time))), fftshift(fft(signal))
    else:
        return fftfreq(len(time), (time[len(time)-1]-time[0])/len(time)), fft(signal)

def Response(time, signal, 
             transmissionFunction,
             filterWidth = None):
    """
    Numerical computation of signal transmission. 
    inputs: 
        - time, signal - time coordinates and the signal to be transmitted
        - transmissionFunction - a function of one variable, the convolution kernel
        - filterWidth - the width of a gaussian filter overlayed with the transmission
    outputs: 
        - FFT convolution array of the signal with the transmissionFunction
    """
    freqs, fourier = FT(time, signal)
    if filterWidth == None:
        return ifft([fourier[i]*transmissionFunction(freqs[i]) for i in range(len(time))])
    else: 
        return ifft([fourier[i]*transmissionFunction(freqs[i])*np.exp(-2*freqs[i]**2/filterWidth**2) for i in range(len(time))])


def IntensityConvolution(time, intensity,
                         IRF,
                         tlimIRF = None,
                         mode = 'same'):
    '''
    Computation of the instrument response to signal's intensity. The convolution is performed by numpy's FFT convolution algorithm.
    inputs: 
        - time, intensity - time coordinates and the intensity to be convolved
        - IRF - instrument response function of time-resolved detector
        - mode - 'same' or 'full'. If 'same', the length of the resulting arrays is the length of intensity array.
        - tlimIRF - 'None' by default, in which case the response array is of the signal array's length.
          If of the form [t_min, t_max], then the convolution pulse is calculated in this range only. This can reduce computation time drastically.
    outputs:
        - time array
        - convolved signal array
    '''

    if tlimIRF == None:
        tlimIRF = [min(time),max(time)]

    dt = time[1]-time[0]
    N = len(time)
    
    T1 = tlimIRF[0]
    T2 = tlimIRF[1]
    conv_time = np.array([T1 + n*dt for n in range(int((T2-T1)/dt))])
    zero_idx = int(round(-T1/(T2-T1)*len(conv_time)))

    new_time = np.concatenate([[time[0]-zero_idx*dt+n*dt for n in range(zero_idx)],
                              [time[k-zero_idx] for k in range(zero_idx, zero_idx+N)],
                              [time[-1]+(n+1)*dt for n in range(N+len(conv_time)-1-(zero_idx+N))]])
    
    IRF = np.vectorize(IRF,otypes = [np.float64])
    irf = IRF(conv_time)
    irf = irf/sum(irf)
    convolved = np.convolve(intensity,irf,mode = 'full')
    if mode == 'full':
        return new_time, convolved
    elif mode == 'same':
        return new_time[zero_idx:zero_idx+N], convolved[zero_idx:zero_idx+N]
    else:
        raise ValueError("'"+str(mode)+"' is not a valid mode! Choose 'full' or 'same'.")
##########################################################################
###################TR-BLS INSTRUMENTAL RESPONSE MODEL#####################
##########################################################################



def AlignedSeparation(coarseD,
                      wavelength,
                      q = 20):
    '''
    The transmittance maximum for given wavelength is defined modulo half-wavelength for SPFP
    or modulo q/2 wavelengths for the Tandem FPI with detuning m = p/q, p<q.

    This function finds the closest mirror separation to the initial guess.
    inputs:
        - coarse_d (mm), the initial value of mirror separation
        - wavelength (nm)
        - q = 20 by default
    outputs:
        - d closest to coarse_d that satisfies maximum transmittance condition for wavelength
    '''
    wavelength = wavelength*10**(-6)
    return np.round(coarseD/(q*wavelength/2))*(q*wavelength/2)


IRFTime = [1.074,1.078,1.082,1.086,1.09,1.094,1.098,1.102,1.106,1.11,1.114,1.118,1.122,1.126,1.13,1.134,1.138,1.142,1.146,1.15,1.154,1.158,1.162,1.166,1.17,1.174,1.178,1.182,1.186,1.19,1.194,1.198,1.202,1.206,1.21,1.214,1.218,1.222,1.226,1.23,1.234,1.238,1.242,1.246,1.25,1.254,1.258,1.262,1.266,1.27,1.274,1.278,1.282,1.286,1.29,1.294,1.298,1.302,1.306,1.31,1.314,1.318,1.322,1.326,1.33,1.334,1.338,1.342,1.346,1.35,1.354,1.358,1.362,1.366,1.37,1.374,1.378,1.382,1.386,1.39,1.394,1.398]
IRFSignal = [1.0,0.9972669246677625,0.9512068052157572,0.8867799738108982,0.793635591407669,0.6925999351578384,0.5960569360979535,0.5006089905718525,0.4188537727202715,0.35162295594860987,0.29571951069847546,0.2559106172936304,0.22375696862033437,0.20009429008596172,0.18318528584366758,0.16954422119433063,0.15935242322633872,0.15145304537578697,0.14420300084175278,0.13972148665723869,0.13435461004115468,0.13029247806106845,0.12614626703999787,0.12208210905892407,0.11922139566446932,0.11573971296732309,0.11360633392740749,0.11090162260899596,0.10786971213109119,0.10563300704080929,0.10351178400681912,0.10169142211948372,0.09972620116153677,0.09819251841394311,0.0960895293888411,0.09399869636966456,0.09228773853565815,0.09105086493274411,0.08858724773185392,0.08787308238373402,0.08618035855861579,0.08441571169843867,0.08320720210935068,0.08201793952964463,0.07979744244726333,0.07911164111296946,0.07809357561671344,0.07594601456988481,0.07516600418966875,0.07412463968205563,0.07324231625196706,0.0717562445275814,0.07036539484961173,0.06964109949655396,0.06894415515682843,0.06654536998754061,0.06616245580088909,0.0652041573337665,0.0642154688518303,0.06352055051309234,0.061747799648964936,0.061474289515642426,0.060147258868781334,0.05928620844906231,0.058738175181923494,0.05751649658641626,0.05742633954246921,0.05625632397214512,0.05533550652329265,0.05459905516430944,0.05354958665274601,0.052928617350054524,0.05243326010859264,0.05159044369776178,0.050432584133363136,0.04974779579956306,0.049443895651426935,0.04882191334824166,0.047949719923090973,0.04714438453053024,0.04627016510439198,0.04562589679034339]

def MPDIRF(x):
    """
    Instrument Response Function of the Micro-Photon Devices photodetector, measured on 14 IV 2026.
    """
    x = x+1.0755
    bound1 = 1.074
    bound2 = 1.394
    if x < bound1:
        return np.exp(-2*(x-1.0755)**2/0.038829226959463316**2)
    elif x >= bound1 and x < bound2:
        index = int((x-bound1)/(1.398-1.074)*len(IRFTime))
        return IRFSignal[index]+(x-IRFTime[index])/(IRFTime[index+1]-IRFTime[index])*(IRFSignal[index+1]-IRFSignal[index])
    elif x>bound2:
        return np.exp(-(x-0.689627)**0.795954/0.245978)
MPDIRF = np.vectorize(MPDIRF, otypes = [np.float64])

def CalcSpectrogram(time, signal,
                    mirrorSeparation,
                    frequencyChannels,
                    freqRange = None,
                    wavelength = 532,
                    scanAmplitude = None,
                    scanPercent = None,
                    shutterPercent = None,
                    symmetric = False,
                    transmissionFunction = None,
                    mode = 'Tandem',
                    R = 0.93,
                    m = 0.95,
                    IRF = None,
                    tlimIRF = None,
                    filterWidth = None,
                    saveFile = True,
                    filenameSuffix = '',
                    filename = '',
                    comments = '',
                    transFuncComments = '',
                    flag = True):
        
    start_time=TIME.time()

    
    if filename == '':
        if filenameSuffix != '':
            filenameSuffix = '-'+filenameSuffix
        filename = ('spectrogram-d-'+
                    str(mirrorSeparation)+"-"+
                    "R-"+str(R)+"-"+
                    "ch-"+str(frequencyChannels)
                    +filenameSuffix+".dat")
    else:
        if filenameSuffix != '':
            print("filenameSuffix is ignored if filename is specified.")
            
    if mode == 'Single':
        q = 1
    else:
        q = Fraction.from_float(m).limit_denominator(100).denominator
    N = frequencyChannels
    
    if N%2 == 0:
        midpoint = int(N/2)
    else:
        midpoint = int(round((N-1)/2))
    
    coarseD = mirrorSeparation ##saving the initial d0
    wvl = wavelength*10**(-6) ##conversion to mm
    f0 = c/wvl ##laser frequency in GHz
    
    d0 = AlignedSeparation(coarseD,wavelength,q)

    if freqRange is None:
        if scanAmplitude is None:
            if scanPercent is None:
                scanPercent = 1
                scanAmplitude = wavelength
            else:
                scanAmplitude = wavelength*scanPercent
        else:
            if scanPercent is not None:
                print("'scanPercent' and 'scanAmplitude' cannot both be specified.\n'scanAmplitude' will override the value of 'scanPercent'.")
            scanPercent = scanAmplitude/wavelength
        FSR = scanPercent * c/(2*d0)
        freqRangeArray = np.array([IndexToFrequency(i,FSR,frequencyChannels) for i in range(frequencyChannels)])
        delta_d = d0/f0*freqRangeArray
        freqRange = [freqRangeArray[0],freqRangeArray[-1]]
        if shutterPercent is not None:
            shutter_d = round_to_3(shutterPercent*max(delta_d))
        else:
            shutter_d = -1
                              
    else:
        scanAmplitude = 'none'
        scanPercent = 'none'
        shutterPercent = 'none'
        FSR = 'none'
        shutter_d = -1
        if symmetric:
            print("'symmetric = True' is not a valid option if freqRange is specified.")
            symmetric = False
        delta_d = d0/f0*np.linspace(*freqRange,N)
        


    delta_t = time[1]-time[0]
    spectrogram_array = [] #initializing the spectrogram
    

    if transmissionFunction is None:
        if mode == 'Tandem':
            CHI = lambda f, d: TandemFPI(f,d,R,m)
        elif mode == 'Single':
            CHI = lambda f, d: SingleFPI(f,d/c,R)
        else:
            raise ValueError("Unknown mode. Keyword 'mode' can only equal 'Tandem' or 'Single'.")
    else:
        print('WARNING: a custom tranmsission function has been supplied.\n'+
              'Be aware that transmission functions must be periodic in mirror separation\n'+
              'in similar way to either single-pass or Tandem multi-pass transmission.\n'
              'If this is not the case, the program will not produce valid results.')
        CHI = lambda f, d: transmissionFunction(f,d)
    if flag:
        print('-------------------')
        print('calculation started')
        print('-------------------')
    
    for i in range(len(delta_d)):
        if flag:
            print(str(wavelength)+" nm",str(coarseD)+" mm",round(1000*i/len(delta_d))/10,"%")
        if np.abs(delta_d[i]) >= shutter_d:
            if i <= midpoint or symmetric == False:     
                period_f = c/(d0+delta_d[i])*q/2 #this is transmission peak period at curren mirror separation in frequency space
                f_prim = f0 - np.floor(f0/period_f)*period_f #shifting laser frequency to nearest peak
                if filterWidth == None:
                    y = np.abs(Response(time,signal,
                                        lambda f: CHI(f+f_prim,d0+delta_d[i]),
                                        filterWidth))**2
                else:
                    y = np.abs(Response(time,signal,
                                        lambda f: CHI(f+f_prim,d0+delta_d[i])*np.exp(-2*f**2/filterWidth**2),
                                        filterWidth))**2
                if IRF is not None:
                    time_discard, y = IntensityConvolution(time,y,IRF,
                                                           tlimIRF = tlimIRF,
                                                           mode = 'same')
                spectrogram_array.append(y)
            if i > midpoint and symmetric == True:
                spectrogram_array.append(spectrogram_array[2*midpoint-(i+1-N%2)])
        else:
            spectrogram_array.append([0]*len(time))
    spectrogram_array = np.transpose(spectrogram_array)
    if saveFile:
        spectrogramToDat(filename,spectrogram_array,
                         origin = 'model',
                         mirrorSeparation = mirrorSeparation,
                         wavelength = wavelength,
                         scanAmplitude = scanAmplitude,
                         scanPercent = scanPercent,
                         FSR = FSR,
                         shutterPercent = shutterPercent,
                         freqN = frequencyChannels,
                         freqMin = freqRange[0],
                         freqMax = freqRange[1],
                         timeN = time.size,
                         timeMin = time[0],
                         timeMax = time[-1],
                         mode = mode,
                         R = R,
                         m = m,
                         comments = comments)

            
    if flag:
        print('Job finished after '+str(round(TIME.time()-start_time,3))+' seconds')
    
    return Spectrogram(spectrogram_array,mirrorSeparation,
                       frequencyChannels,freqRange,
                       time.size,[time[0],time[-1]],
                       wavelength = wavelength,
                       scanAmplitude = scanAmplitude,
                       origin = 'model')


#######################################
########   Reading .pt3 file   ########
#######################################    
def pt3_reader(filename):
    """
    Load raw t3 records and metadata from a PT3 file.
    """

##  This part of the code is based on the
##  phconvert library by Antonino Ingargiola <tritemio@gmail.com> in
##  compliance with the MIT License under which it was published:
##  https://photon-hdf5.github.io/phconvert/
##  -----------------------------------------------------------------------------
##  PHCONVERT LICENSE
##  -----------------------------------------------------------------------------
##
##  The MIT License (MIT)
##  Copyright (c) 2015-2016 The Regents of the University of California,
##                        Antonino Ingargiola and contributors.
##
##  Permission is hereby granted, free of charge, to any person obtaining a copy
##  of this software and associated documentation files (the "Software"), to deal
##  in the Software without restriction, including without limitation the rights
##  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
##  copies of the Software, and to permit persons to whom the Software is
##  furnished to do so, subject to the following conditions:
##
##  The above copyright notice and this permission notice shall be included in
##  all copies or substantial portions of the Software.
##
##  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
##  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
##  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
##  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
##  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
##  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
##  THE SOFTWARE.
##
##  -----------------------------------------------------------------------------


    with open(filename, 'rb') as f:
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Binary file header
        header_dtype = np.dtype([
            ('Ident',             'S16'   ),
            ('FormatVersion',     'S6'    ),
            ('CreatorName',       'S18'   ),
            ('CreatorVersion',    'S12'   ),
            ('FileTime',          'S18'   ),
            ('CRLF',              'S2'    ),
            ('Comment',           'S256'  ),
            ('NumberOfCurves',    'int32' ),
            ('BitsPerRecord',     'int32' ),   # bits in each T3 record
            ('RoutingChannels',   'int32' ),
            ('NumberOfBoards',    'int32' ),
            ('ActiveCurve',       'int32' ),
            ('MeasurementMode',   'int32' ),
            ('SubMode',           'int32' ),
            ('RangeNo',           'int32' ),
            ('Offset',            'int32' ),
            ('AcquisitionTime',   'int32' ),   # in ms
            ('StopAt',            'uint32'),
            ('StopOnOvfl',        'int32' ),
            ('Restart',           'int32' ),
            ('DispLinLog',        'int32' ),
            ('DispTimeAxisFrom',  'int32' ),
            ('DispTimeAxisTo',    'int32' ),
            ('DispCountAxisFrom', 'int32' ),
            ('DispCountAxisTo',   'int32' ),
        ])
        header = np.fromfile(f, dtype=header_dtype, count=1)

        if header['FormatVersion'][0] != b'2.0':
            raise IOError(("Format '%s' not supported. "
                           "Only valid format is '2.0'.") % \
                           header['FormatVersion'][0])

        dispcurve_dtype = np.dtype([
            ('DispCurveMapTo', 'int32'),
            ('DispCurveShow',  'int32')])
        dispcurve = np.fromfile(f, dispcurve_dtype, count=8)

        params_dtype = np.dtype([
            ('ParamStart', 'f4'),
            ('ParamStep',  'f4'),
            ('ParamEnd',   'f4')])
        params = np.fromfile(f, params_dtype, count=3)

        repeat_dtype = np.dtype([
            ('RepeatMode',      'int32'),
            ('RepeatsPerCurve', 'int32'),
            ('RepeatTime',      'int32'),
            ('RepeatWaitTime',  'int32'),
            ('ScriptName',      'S20'  )])
        repeatgroup = np.fromfile(f, repeat_dtype, count=1)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Hardware information header
        hw_dtype = np.dtype([
            ('HardwareIdent',   'S16'  ),
            ('HardwarePartNo',  'S8'   ),
            ('HardwareSerial',  'int32'),
            ('SyncDivider',     'int32'),
            ('CFDZeroCross0',   'int32'),
            ('CFDLevel0',       'int32'),
            ('CFDZeroCross1',   'int32'),
            ('CFDLevel1',       'int32'),
            ('Resolution',      'f4'),
            ('RouterModelCode', 'int32'),
            ('RouterEnabled',   'int32')])
        hardware = np.fromfile(f, hw_dtype, count=1)

        rtr_dtype = np.dtype([
            ('InputType',       'int32'),
            ('InputLevel',      'int32'),
            ('InputEdge',       'int32'),
            ('CFDPresent',      'int32'),
            ('CFDLevel',        'int32'),
            ('CFDZCross',       'int32')])
        router = np.fromfile(f, rtr_dtype, count=4)

        # Time tagging mode specific header
        ttmode_dtype = np.dtype([
            ('ExtDevices',      'int32' ),
            ('Reserved1',       'int32' ),
            ('Reserved2',       'int32' ),
            ('InpRate0',        'int32' ),
            ('InpRate1',        'int32' ),
            ('StopAfter',       'int32' ),
            ('StopReason',      'int32' ),
            ('nRecords',        'int32' ),
            ('ImgHdrSize',      'int32')])
        ttmode = np.fromfile(f, ttmode_dtype, count=1)

        # Special header for imaging. How many of the following ImgHdr
        # array elements are actually present in the file is indicated by
        # ImgHdrSize above.
        ImgHdr = np.fromfile(f, dtype='int32', count=ttmode['ImgHdrSize'][0])

        # The remainings are all T3 records
        t3records = np.fromfile(f, dtype='uint32', count=ttmode['nRecords'][0])

        pump_time = 1./ttmode['InpRate0'][0]/1e-9 #pump repetition time in ns
        tcspc_resolution = hardware['Resolution'][0] #tcspc resolution in ns
        
        metadata = dict(header=header, dispcurve=dispcurve, params=params,
                        repeatgroup=repeatgroup, hardware=hardware,
                        router=router, ttmode=ttmode, imghdr=ImgHdr)
        return t3records, pump_time, tcspc_resolution, metadata

def process_t3records(t3records,pump_time,tcspc_resolution,
                      time_bit=16,
                      dtime_bit=12,
                      frequencyChannels = 1024,
                      ch_bit=4, special_bit=False):
##  This part of the code is adapted from the
##  phconvert library by Antonino Ingargiola <tritemio@gmail.com> in
##  compliance with the MIT License under which it was published:
##  https://photon-hdf5.github.io/phconvert/
##  -----------------------------------------------------------------------------
##  PHCONVERT LICENSE
##  -----------------------------------------------------------------------------
##
##  The MIT License (MIT)
##  Copyright (c) 2015-2016 The Regents of the University of California,
##                        Antonino Ingargiola and contributors.
##
##  Permission is hereby granted, free of charge, to any person obtaining a copy
##  of this software and associated documentation files (the "Software"), to deal
##  in the Software without restriction, including without limitation the rights
##  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
##  copies of the Software, and to permit persons to whom the Software is
##  furnished to do so, subject to the following conditions:
##
##  The above copyright notice and this permission notice shall be included in
##  all copies or substantial portions of the Software.
##
##  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
##  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
##  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
##  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
##  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
##  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
##  THE SOFTWARE.
##
##  -----------------------------------------------------------------------------
    if special_bit:
        ch_bit += 1
    assert ch_bit <= 8
    assert time_bit <= 16
    assert time_bit + dtime_bit + ch_bit == 32

    '''
    ---------------
    TYPES OF EVENTS
    ---------------
    three types of events are registered in t3records:
        - photon counts
        - internal clock overflows
        - FPI frequency channel sync 

    each event in t3 records is composed of three elements collected in arrays: 
        - detector_markers - event marker, either 1 or 15
        - timebins - counts of the internal clock since last overflow
        - timestamps - counts of the internal clock since laser pump sync
        
    the internal clock is (by default) 16-bit and will overflow every 2^16 counts
    this is registered as an event with signature:
        detector_markers[event] == 15
        timestamps[event] == 0

    the FPI sends a frequency channel sync signal to mark the beginning of the channel
    this is registered as an event with signature:
        detector_markers[event] == 15
        timestamps[event] == 1

    if detector_markers[event] == 1, then the event is a photon count.
    '''
    
    detector_markers = np.bitwise_and(np.right_shift(t3records, time_bit + dtime_bit),2**ch_bit - 1).astype('uint8')
    timebins = np.bitwise_and(np.right_shift(t3records, time_bit),2**dtime_bit - 1).astype('uint16')
    dt = np.dtype([('low16', 'uint16'), ('high16', 'uint16')])
    t3records_low16 = np.frombuffer(t3records, dt)['low16']     # View
    timestamps = t3records_low16.astype(np.int64)               # Copy
    np.bitwise_and(timestamps, 2**time_bit - 1, out=timestamps)
    
    overflow_ch = 2**ch_bit - 1
    overflow = 2**time_bit


    '''
    --------------------------
    LABELING PHOTONS AND SYNCS
    --------------------------
    in order to extract the histogram, two loops are performed: find_syncs and index_channels

    first, the job of find_syncs loop is to:
            - collect the indices of events marking the FPI channel sync in sync_idx array
            - collect the indices of events marking photon counts in the photon_idx array
            - correct the internal clock counts in timestamps by adding the overflow counts
    this is done using if statements and event signatures described above
    '''
    
    N = timestamps.size #number of events
    
    timebinsmax = int(pump_time/tcspc_resolution)-1 #highest timebin index allowed by the pump repetition time
                                                    #1 is subtracted due to the fact that eg. max index = 10 means 11 bins are counted 

    init_sync_idx = np.empty(N,dtype = int)
    init_photon_idx = np.empty(N,dtype = int)
    sync_count = np.array([0])
    photon_count = np.array([0])
    
    find_syncs(timestamps,detector_markers,timebins,overflow_ch,overflow,init_sync_idx,init_photon_idx,sync_count,photon_count)
    
    '''     
    ---------------------------------
    FINDING SWEEP STARTS AND RETRACES
    ---------------------------------
    channels are indexed from 0 to 1023
    
    after each series of 1024 channels the FPI needs to retrace back to its original position
    the photons counted while retracing are not properly assigned and as such need to be discarded

    to recognize when the retrace happens, the timestamps array is used
    retrace time is many times longer than the time between channel syncs
    as such, the time from the start of ch1023 to the start of ch0000 is large
    this means beginning of ch0000 is marked by unusually large sync-to-sync time

    to find this in the data:
        1. the sync-to-sync times are calculated as:
            sync_timestamps = timestamps[sync_idx]
            delta = sync_timestamps-np.roll(sync_timestamps,1)
        2. scan start indices are then found by:
            scan_start_idx = sync_idx[np.where(delta > np.median(delta)*5)[0]]
    the 5 x median is arbitrary, but the true time is much greater than the median
    in this way, scan_start_idx holds the indices of sync_idx at which the first channel starts

    since the last channel with index 1023 contains mostly the retrace data it is discarded

    '''

    
    
    

    sync_count = sync_count[0]
    photon_count = photon_count[0]
    sync_idx = init_sync_idx[:sync_count]
    photon_idx = init_photon_idx[:photon_count]
    
    sync_timestamps = timestamps[sync_idx]
    delta = sync_timestamps-np.roll(sync_timestamps,1)
    scan_start_idx = sync_idx[np.where(delta > np.median(delta)*5)[0]]


    '''
    ---------------------
    INDEXING THE CHANNELS
    ---------------------
    using the sync_idx and scan_start_idx arrays, the frequency channels are assigned in the index_channels loop
    the frequency channel the event is assigned to are stored in the freq_channels array

    index_channels loop runs over the sync_idx array, incrementing the freq_index by 1 at each step

    first and final sweeps are most likely incomplete and as such will be discarded by setting freq_channels = 1023
    '''
    
    freq_channels = np.empty(N,dtype = int)
    index_channels(sync_idx,scan_start_idx,freq_channels)

    '''
    ---------------------------------------
    EXCLUDING SYNCS, OVERFLOWS AND RETRACES
    ---------------------------------------
    freq_channels and timebins arrays are of the size of full t3records and contain entries corresponding to overflows and syncs
    for the histogram, only the entries corresponding to photons are important
    
    the arrays can be trimmed using photon_idx array marking the photon events

    retraces are likewise discarded by ensuring freq_channels == 1023 are sieved out
    '''
    
    freq_channels_trim = freq_channels[photon_idx]
    timebins_trim = timebins[photon_idx]
    keep_idx = np.where(freq_channels_trim != frequencyChannels-1)[0]

    '''
    ----------------------
    HISTOGRAMMING THE DATA
    ----------------------
    after the exlusions, the arrays timebins_trim[keep_idx] and freq_channels_trim[keep_idx] contain paired data

    each valid pair (t,f) consists of time bin index t and frequency channel index f in ranges:
        - t = 0,1,...,timebinsmax
        - f = 0,1,...,1023

    pairs are histogrammed into a timebinsmax+1 by 1024 spectrogram
    '''
    
    spectrogram,tbins,fbins = np.histogram2d(timebins_trim[keep_idx],
                                             freq_channels_trim[keep_idx],
                                             bins = (timebinsmax+1,1024),
                                             range = [[-0.5,timebinsmax+0.5],[-0.5,1023.5]])

    '''
    -------------
    FINAL OUTPUTS
    -------------
    the outputs of this function are:
        - spectrogram - the histogrammed events
        - keep_idx.size - number of valid photon events
        - scan_start_idx.size - number of sweeps
    
    '''
    return spectrogram, keep_idx.size, scan_start_idx.size 

def find_syncs(timestamps,detectors,timebins,overflow_ch,overflow,sync_idx,photon_idx,sync_count,photon_count):
    overflow_correction = 0
    for i in range(timestamps.size):
        if detectors[i] == overflow_ch:
            if timebins[i] == 0:
                overflow_correction += overflow
            else:
                sync_idx[sync_count[0]] = i
                sync_count[0]+=1
        else:
            photon_idx[photon_count[0]] = i
            photon_count[0]+=1
        timestamps[i]+=overflow_correction
def index_channels(sync_idx,scan_start_idx,freq_channels):
    freq_index = 0
    freq_channels[:sync_idx[0]] = 0
    freq_index+=1
    for i in range(1,sync_idx.size):
        freq_channels[sync_idx[i-1]:sync_idx[i]] = freq_index
        freq_index+=1
        if sync_idx[i] >= scan_start_idx[0]:
            break
    ## discard the first sweep:        
    freq_channels[:sync_idx[i]] = 1023
    i0 = i
    freq_index = 0
    for i in range(i0+1,sync_idx.size):
        freq_channels[sync_idx[i-1]:sync_idx[i]] = freq_index
        freq_index = (freq_index+1)%1024
    freq_channels[sync_idx[i]:] = freq_index
    ## discard last sweep
    freq_channels[scan_start_idx[-1]:] = 1023
    

if has_numba:
    find_syncs = numba.jit('void(i8[:], u1[:], u2[:],u4,u8,i8[:],i8[:],i8[:],i8[:])')(find_syncs)
    index_channels = numba.jit('void(i8[:],i8[:],i8[:])')(index_channels)



def MakeCmap(colors, weights = None):
    '''
    Creates a matplotlib cmap from the given color list.
    Input:
        - colors: array of matplotlib colors.
        - weights (optional): integer array. By default None, meaning each color in colors is eually spread in the cmap.
    '''
    if weights is not None:
        colors = np.concatenate([[colors[i]]*weights[i] for i in range(len(colors))])
    return matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

spectrum_cmap = MakeCmap(['black','xkcd:dark green','white'])

SpectrogramHeaderLength = 30
SpectrogramKeywords = ["origin",
                       "mirrorSeparation",
                       "freqN",
                       "freqMin",
                       "freqMax",
                       "timeN",
                       "timeMin",
                       "timeMax",
                       "wavelength",
                       "scanAmplitude",
                       "pumpTime",
                       "resolution",
                       "sweeps",
                       "FPImode",
                       "reflectance",
                       "detuning"]

def spectrogramToDat(filename,spectrogram,
                     **kwargs):
    np.savetxt(filename, spectrogram)
    with open(filename,'r') as f:
        spectrogram = f.read()
    count = 0
    with open(filename, 'w') as f:
        for key in SpectrogramKeywords:
            try:
                f.write(str(key)+":"+str(kwargs[key])+'\n')
            except KeyError as e:
                f.write(str(key)+":"+'none'+'\n')
            count+=1
        for i in range(count,30):
            f.write('\n')
        f.write(spectrogram)


class Spectrogram:
    def __init__(self,data,
                 d,
                 freqN,freqRange,
                 timeN,timeRange,
                 wavelength = 'none',
                 scanAmplitude = 'none',
                 laserdt = 'none',
                 resolution = 'none',
                 sweeps = 'none',
                 origin = 'none',
                 info = 'none'):
        self.data = data
        self.wavelength = wavelength
        self.scanAmplitude = scanAmplitude
        self.d = d
        self.laserdt = laserdt
        self.freqRange = freqRange
        self.freqN = freqN
        self.timeRange = timeRange
        self.timeN = timeN
        self.sweeps = sweeps
        if scanAmplitude != 'none' and wavelength != 'none':
            self.FSR = scanAmplitude/wavelength*c/2/d
        else:
            self.FSR = 'none'
        self.resolution = resolution
        self.origin = origin
        self.info = info


    
    def plot(self,ax,
             color = None,
             cmap = spectrum_cmap,
             shutterPercent = 0,
             labels = True):
        if labels:
            extent = [*self.freqRange,*self.timeRange]
            CustomizeAxes(ax,
                      grid = False,
                      ylabel = 'Time (ns)',
                      xlabel = 'Frequency shift (GHz)')
        else:
            extent = None
        if color is not None:
            cmap = MakeCmap(['black',color,'white'])

        Y = self.data.copy()
        
        if shutterPercent != 0:
            shutterN = int(np.ceil(self.freqN*shutterPercent/2))
            Y[:,int(self.freqN/2)-shutterN:int(self.freqN/2)+shutterN] = 0
        Y = np.flipud(Y/Y.max()+1e-5)
            
        ax.imshow(Y,
                  aspect = 'auto',
                  cmap = cmap,
                  extent = extent,
                  interpolation = 'none',
                  vmin = 1e-5,
                  vmax = 1e-5+1,
                  norm = 'log')

    def getSpectrum(self,
                    norm = None):
        Y = np.sum(self.data,axis = 0)
        print(Y.shape)
        if norm is not None:
            if norm == True:
                return np.linspace(*self.freqRange,self.freqN), Y/Y.max()
            if norm == 'sweeps':       
                return np.linspace(*self.freqRange,self.freqN), Y/self.sweeps
            else:
                raise ValueError("norm should be None, True or 'sweeps'")
        else:    
            return np.linspace(*self.freqRange,self.freqN), np.sum(self.data,axis = 0)
    def getTime(self,
                binwidth = 1):
        if self.origin == "data":
            cut = np.sum(self.data,axis = 1)
            new_timebins_N = int(np.floor(self.timeN/binwidth))
            new_timebins = np.zeros(new_timebins_N,dtype = float)
            idx = np.array([int(i/binwidth) for i in range(self.timeN)])
            
            for i in range(self.timeN):
                if idx[i] < new_timebins_N:
                    new_timebins[idx[i]] += cut[i]
                else:
                    new_timebins[-1]+=cut[i]
            
            
                
            bin_dt = (self.timeRange[1]-self.timeRange[0])/(self.timeN-1)
            
            new_bin_dt = binwidth*bin_dt
            

            new_time = np.linspace(new_bin_dt/2,new_bin_dt/2+(new_timebins_N-1)*new_bin_dt,new_timebins_N)
            return new_time, new_timebins   
        else:
            return np.linspace(*self.timeRange,self.timeN), np.sum(self.data,axis = 1)
    def getTimeSlice(self,
                     freqRange,
                     binwidth = 1,
                     filename = None):
        
        cut = np.zeros(self.timeN,dtype = float)
        if isinstance(freqRange[0],list):
            for fR in freqRange:        
                [f1,f2] = fR
                i1 = int(np.floor(self.freqN*(f1*self.d*self.wavelength/(c*self.scanAmplitude)+0.5)))
                i2 = int(np.floor(self.freqN*(f2*self.d*self.wavelength/(c*self.scanAmplitude)+0.5)))
                cut += np.sum(self.data[:,i1:i2+1],axis = 1)
        else:            
            [f1,f2] = freqRange
            i1 = int(np.floor(self.freqN*(f1*self.d*self.wavelength/(c*self.scanAmplitude)+0.5)))
            i2 = int(np.floor(self.freqN*(f2*self.d*self.wavelength/(c*self.scanAmplitude)+0.5)))
            cut += np.sum(self.data[:,i1:i2+1],axis = 1)
                
        
        new_timebins_N = int(np.floor(self.timeN/binwidth))
        new_timebins = np.zeros(new_timebins_N,dtype = float)
        idx = np.array([int(i/binwidth) for i in range(self.timeN)])
        
        for i in range(self.timeN):
            if idx[i] < new_timebins_N:
                new_timebins[idx[i]] += cut[i]
            else:
                new_timebins[-1]+=cut[i]
        
        
            
        bin_dt = (self.timeRange[1]-self.timeRange[0])/(self.timeN)
        new_bin_dt = binwidth*bin_dt
        new_time = np.linspace(new_bin_dt/2,new_bin_dt/2+(new_timebins_N-1)*new_bin_dt,new_timebins_N)
        
        if filename is not None:
            np.savetxt(filename,np.transpose([new_time,new_timebins]))
        
        return new_time, np.array(new_timebins)
    
    def saveAsDat(self,filename):
        if self.origin == 'data':
            spectrogramToDat(filename,self.data,
                             origin = self.origin,
                             mirrorSeparation = self.d,
                             wavelength = self.wavelength,
                             scanAmplitude = self.scanAmplitude,
                             resolution = self.resolution,
                             freqN = self.freqN,
                             freqMin = self.freqRange[0],
                             freqMax = self.freqRange[1],
                             timeN = self.timeN,
                             timeMin = self.timeRange[0],
                             timeMax = self.timeRange[1],
                             sweeps = self.sweeps,
                             FSR = self.FSR,
                             pumpTime = self.laserdt)
            
        

def IndexToFrequency(i,FSR,channels):
    if channels%2 == 0:
        i0 = channels/2-0.5
    else:
        i0 = (channels-1)/2
    return 2*FSR*(i-i0)/channels

def LoadSpectrogram(fileName,
                    mirrorSeparation = None,
                    frequencyChannels = 1024,
                    scanAmplitude = 490,
                    probeWavelength = 532):
    if fileName[-4:] == '.pt3':
        if mirrorSeparation is None:
            raise ValueError('mirror separation cannot be None for .pt3 file import as .pt3 files do not store FPI mirror separation.')
        elif mirrorSeparation <= 0:
            raise ValueError('mirror separation must be positive!')
        t3records, pump_time, tcspc_resolution, _ = pt3_reader(fileName)
        
        data, photons, sweeps = process_t3records(t3records, pump_time, tcspc_resolution,
                                                  frequencyChannels = frequencyChannels)
        timebins = data.shape[0]
        d = mirrorSeparation
        FSR = c/(2*d)*scanAmplitude/probeWavelength
        f0 = IndexToFrequency(0,FSR,frequencyChannels)
        f1023 = IndexToFrequency(1023,FSR,frequencyChannels)
        
        return Spectrogram(data,
                           d,
                           frequencyChannels,[f0, f1023],
                           timebins,[tcspc_resolution/2,tcspc_resolution/2+(timebins)*tcspc_resolution],
                           probeWavelength,
                           scanAmplitude,
                           pump_time,
                           tcspc_resolution,
                           sweeps = sweeps,
                           origin = 'data')
    elif fileName[-4:] == '.dat':
        if mirrorSeparation is not None:
            print('The value of mirrorSeparation imported from the .dat Spectrogram file will be used.')
        with open(fileName,'r') as f:
            lines = f.readlines()[:SpectrogramHeaderLength]
            newlines = []
            for line in lines:
                if line != '\n':
                    newlines.append(line)
            args = {key.strip():value.strip() for key, value in (line.split(':',1) for line in newlines)}
        data = np.loadtxt(fileName,skiprows = SpectrogramHeaderLength)
        if args["origin"] == "data":
            return Spectrogram(data,
                               float(args["mirrorSeparation"]),
                               int(args["freqN"]),[float(args["freqMin"]),float(args["freqMax"])],
                               int(args["timeN"]),[float(args["timeMin"]),float(args["timeMax"])],
                               wavelength = float(args["wavelength"]),
                               scanAmplitude = float(args["scanAmplitude"]),
                               laserdt = float(args["pumpTime"]),
                               resolution = float(args["resolution"]),
                               sweeps = int(args["sweeps"]),
                               origin = args["origin"])
        elif args["origin"] == "model":
            return Spectrogram(data,
                               float(args["mirrorSeparation"]),
                               int(args["freqN"]),[float(args["freqMin"]),float(args["freqMax"])],
                               int(args["timeN"]),[float(args["timeMin"]),float(args["timeMax"])],
                               wavelength = float(args["wavelength"]),
                               origin = args["origin"])

    else:
        raise TypeError("'"+fileName[-4:]+"' is not a valid file extension. Only .dat and .pt3 files are permitted.")

def TimeRoll(A,timeShift,dt):
    return np.roll(A,int(np.round(timeShift/dt)))

def TimeSelector(dataTime,
                 modelTime, modelSignal):
    
    indices = np.array([int(np.floor((modelTime.size-1)*(t-modelTime[0])/(modelTime[-1]-modelTime[0]))) for t in dataTime])
    return dataTime, modelSignal[indices]+(dataTime-modelTime[indices])*(modelSignal[indices+1]-modelSignal[indices])/(modelTime[indices+1]-modelTime[indices])

def TimeRMS(dataTime,dataSignal,
            modelTime,modelSignal):
    '''
    This is a shorthand to calculate the RMS.
    It does not take any parameters and so cannot be used in minimize procedures on its own.
    '''
    discard, sievedSignal = TimeSelector(dataTime,modelTime,modelSignal)
    if sievedSignal.size != dataSignal.size:
        raise ValueError("size of data ("+str(int(dataSignal.size))+") does not equal the size of sieved model ("+str(int(sievedSignal.size))
                         +"). Something went horribly wrong in the TimeSelector function.")
    else:
        return np.sum(np.abs(dataSignal-sievedSignal)**2)
    


        
        
