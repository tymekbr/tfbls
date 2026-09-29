from tfbls import *

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from scipy.optimize import minimize

'''
To run this code, download .pt3 files from Zenodo:
UPDATE THIS AFTER PUBLISHING
'''



'''
Set optimize to True for the optimization to run (it takes a f.
Set optimize to False to see the result of the optimization immediately.
'''

optimize = True


'''
-----------
PREPARATION
-----------
In the fields below, the calculation parameters should be specified.
'''

'''
pumpFilename should be the .pt3 file with the pump laser's signal.
pumpFreqRange is the list of frequency shift ranges over wich the signal is averaged.
pumpMirrorSeparation is the FPI mirror separation at which the signal is measured.
'''
pumpFilename = 'pump-laser-signal-1.4.pt3'
pumpFreqRange = [[-90,-10],[10.0,90.0]]
pumpMirrorSeparation = 1.4


'''
D is the list of mirror separations for which the data was acquired.
filenames is the list of corrsesponding .pt3 files.
freqRanges is the list of frequency ranges over which the signal is averaged.
freqNs is the # of frequency channels in each range.
'''
D = [1.4,2,4]

filenames = ['MoS2-140nm-'+str(d)+'mm.pt3' for d in D]

freqRanges = [[[-10.70,-10.30],[10.70,11.08]],
              [[-10.73,-10.46],[10.86,11.13]],
              [[-10.76,-10.62],[10.83,10.96]]]

freqNs = [[3,3],
          [3,3],
          [3,3]]

'''
The FPI reflectance R needs to be established by measurement and provided
as a parameter to calculations.
'''
R = 0.933

'''
--------------------------------------------------------------------------
'''
show_all = True
plot_the_model_curve = True

color = ['xkcd:dark red',
         'xkcd:dark blue',
         'xkcd:dark green']
scatcolor = ['xkcd:red',
             'tab:blue',
             'xkcd:green']

axes_fontsize = 15
font = {'size'   : axes_fontsize}
matplotlib.rc('font', **font)

'''
------------
DETECTOR IRF
------------
'''
pumpS = LoadSpectrogram(pumpFilename,
                        mirrorSeparation = pumpMirrorSeparation)

[pumpX,pumpY] = pumpS.getTimeSlice(pumpFreqRange,binwidth = 5)

pumpTime = pumpS.laserdt

imax = np.where(pumpY == pumpY.max())[0]
shift = pumpX[imax]
pumpY = np.roll(pumpY,int(len(pumpX)/2)-imax)
imax = int(len(pumpX)/2)

pumpX = pumpX-pumpX[imax]
shift += pumpX[imax]
shift = shift[0]
pumpY = pumpY/pumpY.max()

pumpIRF = lambda x: np.interp(x,pumpX,pumpY)

if show_all:
    fig,ax = plt.subplots()
    ax.plot(pumpX,pumpY,
            color = 'xkcd:crimson')
    inset = ax.inset_axes([.13,.6,.3,.35])
    inset.plot(*pumpS.getSpectrum(norm = 'sweeps'),
               color = 'xkcd:crimson')
    CustomizeAxes(inset,
                  xlabel = 'Freq. shift (GHz)',
                  ylabel = 'Counts/sweep',
                  yticks = [0,250,500,750])
    CustomizePlot(fig,ax,
                  windowsize = [20,12],
                  xlabel = 'Time (ns)',
                  ylabel = 'Signal (a. u.)')
    fig.suptitle('Pump laser signal/photodetector response function')
    plt.show()

'''
-------------------------------------
DISPLAYING THE MODES AND SPECTROGRAMS
-------------------------------------
'''

dataS = [LoadSpectrogram(f,d) for f,d in zip(filenames,D)]
for S in dataS:
    S.timeRange = [S.timeRange[0]-shift,S.timeRange[1]-shift]
tlim = dataS[0].timeRange
X = []
Y = []
for i in range(len(D)):
    [dX,dY] = dataS[i].getTimeSlice(freqRanges[i],binwidth = 5)
    X.append(dX-shift)
    Y.append(dY)


if show_all:
    fig, ax = plt.subplots(3,2)
    for i in range(len(D)):
        dataS[i].plot(ax[i,0],labels = True,
                      color = color[i],
                      shutterPercent = 0.07)
        ax[i,0].set_xlim(0,dataS[i].freqRange[1])
        ax[i,0].text(.97,1,r'anti-Stokes, '+str(D[i])+' mm',
                     verticalalignment = 'top',
                     horizontalalignment = 'right',
                     color = 'white',
                     fontweight = 'bold',
                     fontsize = 12,
                     transform = ax[i,0].transAxes)
        ax[i,1].scatter(X[i],Y[i],
                        s = 10,
                        alpha = 0.5,
                        color = color[i])
        ax[i,0].fill()
        CustomizeAxes(ax[i,1],
                      xlabel = 'Time (ns)',
                      ylabel = 'Signal (a. u.)',
                      grid = True)
    CustomizeFig(fig,windowsize = [20,20])
    plt.show()

'''
---------------------
DEFINING THE FUNCTION
---------------------
Value of pumpTime is taken from the .pt3 file with the IR pump's signal.

'''

def fit(t,tau,f0,A):
    return (Exp(t,2*tau,0,A)+
            Exp(t,2*tau,-pumpTime,A)+
            Exp(t,2*tau,pumpTime,A)+
            Exp(t,2*tau,-2*pumpTime,A))*np.cos(2*np.pi*f0*t)


'''
----------------
DEFINING THE RMS
----------------
'''
T = np.linspace(-3*pumpTime,6*pumpTime,2**15)
rms0 = 1
def RMS(x,update = False):
    global rms0
    [tau,f0] = x[:2]
    E0 = x[2:5]
    noise = x[5:8]
    SIGNAL = fit(T,tau,f0,1)
    rms = 0
    for i in range(len(D)):    
        modelAS = CalcSpectrogram(T,SIGNAL,D[i],
                                  freqRange = freqRanges[i][1],
                                  frequencyChannels = freqNs[i][1],
                                  scanPercent = 1,
                                  IRF = pumpIRF,
                                  tlimIRF = [-5,5],
                                  R = R,
                                  saveFile = False,
                                  flag = False)
        modelS = CalcSpectrogram(T,SIGNAL,D[i],
                                 freqRange = freqRanges[i][0],
                                 frequencyChannels = freqNs[i][0],
                                 IRF = pumpIRF,
                                 tlimIRF = [-5,5],
                                 R = R,
                                 saveFile = False,
                                 flag = False)
        XS,YS = modelS.getTime()
        XAS,YAS = modelAS.getTime()

        mY = np.abs(noise[i])+E0[i]**2*(YAS+YS)
        mX = XS
        rms+=TimeRMS(X[i],Y[i],mX,mY)
    if update:
        rms0 = rms
    print(rms/rms0,*x)
    return rms
'''
--------------------------
DEFINING NICE-LOOKING PLOT
--------------------------
'''
def plot(x,recalculate = True,
         title = ''):

    fig = plt.figure()
    gs = fig.add_gridspec(1,5)
    ax = [fig.add_subplot(gs[0,:3]),
          fig.add_subplot(gs[0,3:])]
    d = .02




    [tau,f0] = x[:2]
    E0 = x[2:5]
    noise = x[5:8]

    T = np.linspace(-3*pumpTime,6*pumpTime,2**15)
    SIGNAL = fit(T,tau,f0,1)
    rms = 0
    
    for i in range(3):
        i1 = ax[0].inset_axes([0,1-(i+1)*(1-2*d)/3-i*d,1,(1-2*d)/3])
        i2 = ax[1].inset_axes([0,1-(i+1)*(1-2*d)/3-i*d,1,(1-2*d)/3])
        
        modelAS = CalcSpectrogram(T,SIGNAL,D[i],
                                  freqRange = freqRanges[i][1],
                                  frequencyChannels = freqNs[i][1],
                                  scanPercent = 1,
                                  IRF = pumpIRF,
                                  tlimIRF = [-5,5],
                                  R = R,
                                  saveFile = False,
                                  filename = 'model-spectrogram-antiStokes-'+str(D[i])+'.dat',
                                  flag = False)
        modelS = CalcSpectrogram(T,SIGNAL,D[i],
                                 freqRange = freqRanges[i][0],
                                 frequencyChannels = freqNs[i][0],
                                 IRF = pumpIRF,
                                 tlimIRF = [-5,5], 
                                 R = R,
                                 saveFile = False,
                                 filename = 'model-spectrogram-Stokes-'+str(D[i])+'.dat',
                                 flag = False)
        XS,YS = modelS.getTime()
        XAS,YAS = modelAS.getTime()

        mY = np.abs(noise[i])+E0[i]**2*(YAS+YS)
        mX = XS

        
        sX,sY = TimeSelector(X[i],mX,mY)

        i1.scatter(X[i],Y[i],
                   s = 10,
                   color = scatcolor[i],
                   alpha = 0.5)

        i1.plot(mX,mY,
                color = color[i],
                linewidth = 3)
        
        rms+=TimeRMS(X[i],Y[i],mX,mY)

        i2.scatter(X[i],sY-Y[i],
                   s = 10,
                   color = scatcolor[i],
                   alpha = 0.5)

        if plot_the_model_curve:
            i1.plot(T,max(mY)*np.abs(fit(T,tau,0,1))**2/np.abs(fit(0,tau,0,1))**2,
                    color = 'grey')

        i1.text(.98,.95,(str(D[i])+' mm\n'+
                         r'$y_0$ = '+str(round(np.abs(noise[i]),1))+'\n'+
                         r'$E_0$ = '+str(round(E0[i],1))),
                color = 'black',
                fontsize = 15,
                ha = 'right',
                va = 'top',
                transform = i1.transAxes)
        
        if i < 2:
            CustomizeAxes(i1,
                         xticks = [],
                         xlabel = None,
                         ylabel = None,
                         xlim = tlim)
            CustomizeAxes(i2,
                         xticks = [],
                         xlabel = None,
                         ylabel = None,
                         xlim = tlim)
        else:
            CustomizeAxes(i1,
                          xlabel = r'Time $t$'+' (ns)',
                          ylabel = None,
                          xlim = tlim)
            CustomizeAxes(i2,
                          xlabel = r'Time $t$'+' (ns)',
                          ylabel = None,
                          xlim = tlim)
    CustomizePlot(fig, ax[0],
                  yticks = [],
                  remove_axes = True,
                  ylabel = r'Intensity $I$ (a. u.)'+'\n\n',
                  xlim = tlim)
    CustomizePlot(fig, ax[1],
                  windowsize = [20,20],
                  yticks = [],
                  remove_axes = True,
                  ylabel = 'Residuals\n\n',
                  pad = 1,
                  w_pad = 1,
                  xlim = tlim)
    fig.suptitle(title+r'$\tau$ = '+str(round(x[0],2))+r' ns, $f_0$ = '+str(round(x[1],2))+' GHz\n')
    plt.show()

'''
------------
OPTIMIZATION
------------
'''
x_ini = [4,10.9,22,23,22,20, 15, 14]
plot(x_ini, title = 'Initial guess: ')
if optimize:
    RMS(x_ini,True)
    res = minimize(RMS,
                   x0 = x_ini,
                   method = 'BFGS')

    print('\n-----------------------\n')
    print('SAMPLE: 140 nm\n')
    print('MODE: S1\n')
    print('tau: '+str(round(res.x[0],6))+' +/- '+str(round(STDDEV[0],6))+' ns\n')
    print('frq: '+str(round(res.x[1],6))+' +/- '+str(round(STDDEV[1],6))+' GHz\n')
    print('E0 (1.4): '+str(round(res.x[2],6))+' +/- '+str(round(STDDEV[2],6))+'\n')
    print('E0 (2): '+str(round(res.x[3],6))+' +/- '+str(round(STDDEV[3],6))+'\n')
    print('E0 (4): '+str(round(res.x[4],6))+' +/- '+str(round(STDDEV[4],6))+'\n')
    print('noise (1.4): '+str(round(np.abs(res.x[5]),6))+' +/- '+str(round(STDDEV[5],6))+'\n')
    print('noise (2): '+str(round(np.abs(res.x[6]),6))+' +/- '+str(round(STDDEV[6],6))+'\n')
    print('noise (4): '+str(round(np.abs(res.x[7]),6))+' +/- '+str(round(STDDEV[7],6))+'\n')

    x_opti = res.x
else:
    '''
    Optimal parameters are:
    '''
    x_opti = [3.884049364330316,
              10.851401099571985,
              21.95568858926641, 22.790899566624514, 21.89394282304425,
              20.083127325904147, 14.887168681926, 14.081190531739315]

plot(x_opti, title = 'Optimal parameters: ')

quit()


