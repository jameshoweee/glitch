#!/usr/bin/env python3
'''
Created on 26.05.2014
Last updated 03.02.2016
@author: James
'''

import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from decimal import *
import pprint
import sys
import math
import csv

import statsmodels.api as sm
import matplotlib.gridspec as gridspec
import pylab as pl

getcontext()
Context(prec = 128, traps=[Overflow, DivisionByZero, InvalidOperation])

target_mean = Decimal(0)
target_sigma = Decimal(215.7277372731568368512617199)
target_precision = Decimal(128)
target_tail_cut = Decimal(math.sqrt(target_precision*Decimal(2)*Decimal(math.log(2))))
target_skew = Decimal(0)
target_kurt = Decimal(3)
target_hyper_skew = Decimal(0)
target_hyper_kurt = Decimal(15)
data = r'./samples/sample_size_68719476736/gauss_samples_general_ber_speed_236'
#data = 'C:/Users/40108992/Documents/Gauss_Sampler_Tests/gauss_sampler/coding/samples/buggy/gauss_samples_general_ber_speed_buggy_210'
#zig_data = 'C:/Users/40108992/Dropbox/gauss_samplers/zig_test_vectors'
x_vals11 = []
gauss_cnt11 = []
x_vals111 = []
gauss_cnt_plt = []
x_vals_plt = []
gauss_cnt_test = []

def binning(x_vals,gauss_cnt):
    xvalsdic=[]
    gausscntdic=[]
    expectedfreqdic = []
    for i in range(0,len(x_vals111)):
        count = 0
        gauss_cnt_temp = 0
        if count == 0:
            if gauss_cnt[i] >= 5:
                xvalsdic.append(int(x_vals[i]))
                gausscntdic.append(gauss_cnt[i])
                expectedfreqdic.append(Decimal(sum(gauss_cnt)) * Decimal(np.exp(-(Decimal(x_vals[i])-Decimal(target_mean))**Decimal(2)/(Decimal(2)*(Decimal(target_sigma)**Decimal(2))))) / Decimal(np.sqrt(Decimal(2)*Decimal(np.pi)*(Decimal(target_sigma)**Decimal(2)))))
                count = 0
                gauss_cnt_temp = 0
            else:
                count += gauss_cnt[i]
                gauss_cnt_temp += Decimal(sum(gauss_cnt)) * Decimal(np.exp(-(Decimal(x_vals[i])-Decimal(target_mean))**Decimal(2)/(Decimal(2)*(Decimal(target_sigma)**Decimal(2))))) / Decimal(np.sqrt(Decimal(2)*Decimal(np.pi)*(Decimal(target_sigma)**Decimal(2))))
        elif count < 5:
            if gauss_cnt[i]+count >= 5:
                xvalsdic.append(int(x_vals[i]))
                gausscntdic.append(gauss_cnt[i]+count)
                expectedfreqdic.append(Decimal(sum(gauss_cnt)) * Decimal(np.exp(-(Decimal(x_vals[i])-Decimal(target_mean))**Decimal(2)/(Decimal(2)*(Decimal(target_sigma)**Decimal(2))))) / Decimal(np.sqrt(Decimal(2)*Decimal(np.pi)*(Decimal(target_sigma)**Decimal(2)))))+Decimal(sum(gauss_cnt)) * Decimal(np.exp(-(Decimal(count)-Decimal(target_mean))**Decimal(2)/(Decimal(2)*(Decimal(target_sigma)**Decimal(2))))) / Decimal(np.sqrt(Decimal(2)*Decimal(np.pi)*(Decimal(target_sigma)**Decimal(2))))
                count = 0
                gauss_cnt_temp = 0
            else:
                count += gauss_cnt[i]
                gauss_cnt_temp += (Decimal(sum(gauss_cnt)) * Decimal(np.exp(-(Decimal(x_vals[i])-Decimal(target_mean))**Decimal(2)/(Decimal(2)*(Decimal(target_sigma)**Decimal(2))))) / Decimal(np.sqrt(Decimal(2)*Decimal(np.pi)*(Decimal(target_sigma)**Decimal(2)))))+Decimal(sum(gauss_cnt)) * Decimal(np.exp(-(Decimal(count)-Decimal(target_mean))**Decimal(2)/(Decimal(2)*(Decimal(target_sigma)**Decimal(2))))) / Decimal(np.sqrt(Decimal(2)*Decimal(np.pi)*(Decimal(target_sigma)**Decimal(2))))

        else:
            xvalsdic.append(int(x_vals[i]))
            gausscntdic.append(gauss_cnt[i]+count)
            expectedfreqdic.append(Decimal(sum(gauss_cnt)) * Decimal(np.exp(-(Decimal(x_vals[i])-Decimal(target_mean))**Decimal(2)/(Decimal(2)*(Decimal(target_sigma)**Decimal(2))))) / Decimal(np.sqrt(Decimal(2)*Decimal(np.pi)*(Decimal(target_sigma)**Decimal(2)))))+Decimal(sum(gauss_cnt)) * Decimal(np.exp(-(Decimal(count)-Decimal(target_mean))**Decimal(2)/(Decimal(2)*(Decimal(target_sigma)**Decimal(2))))) / Decimal(np.sqrt(Decimal(2)*Decimal(np.pi)*(Decimal(target_sigma)**Decimal(2))))
            count = 0
            gauss_cnt_temp = 0
    return (xvalsdic,gausscntdic,expectedfreqdic)

def twos_comp(val, bits):
    """compute the 2's compliment of int value val"""
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val                         # return positive value as is

##zig_data_new = []
##for i in zig_data:
##    j = twos_comp(int(i,2), len(i))
##    zig_data_new.append(j)


with open(data, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        x_vals11.append(Decimal(row[0]))
        x_vals111.append(int(row[0]))
        gauss_cnt11.append(Decimal(row[1]))
        x_vals_plt.append(int(row[0]))


x_vals_full = x_vals11
x_vals_full_int = [int(i) for i in x_vals_full]
gauss_cnt_full = gauss_cnt11
gauss_cnt_full_float = [float(i) for i in gauss_cnt_full]
binnin = binning(x_vals_full,gauss_cnt_full)
x_vals = binnin[0]
gauss_cnt = binnin[1]
exp_freq = binnin[2]
exp_freq1 = [float(i) for i in binnin[2]]
gauss_cnt_int = [int(i) for i in gauss_cnt_full]
gauss_cnt_prob = [float(i)/float(sum(gauss_cnt_full)) for i in binnin[1]]


def mean(x_vals_full, gauss_cnt_full):
    a = []
    for i in range(0, len(x_vals_full)):
        b = Decimal(x_vals_full[i]) * Decimal(gauss_cnt_full[i])
        a.append(b)
    mean = np.sum(a)/Decimal(sum(gauss_cnt_full))
    return(mean)

mean = mean(x_vals_full, gauss_cnt_full)

def second_moment(x_vals_full, gauss_cnt_full):
    c=[]
    for i in range(0, len(x_vals_full)):
        d = ( ( Decimal(x_vals_full[i]) - Decimal(mean) )**2 ) * Decimal(gauss_cnt_full[i])
        c.append(d)
    return(c)

def third_moment(x_vals_full, gauss_cnt_full):
    e=[]
    for i in range(0, len(x_vals_full)):
        f = ( ( Decimal(x_vals_full[i]) - Decimal(mean) )**3 ) * Decimal(gauss_cnt_full[i])
        e.append(f)
    return(e)

def forth_moment(x_vals_full, gauss_cnt_full):
    g=[]
    for i in range(0, len(x_vals_full)):
        h = ( ( Decimal(x_vals_full[i]) - Decimal(mean) )**4 ) * Decimal(gauss_cnt_full[i])
        g.append(h)
    return(g)

def fifth_moment(x_vals_full, gauss_cnt_full):
    j=[]
    for i in range(0, len(x_vals_full)):
        k = ( ( Decimal(x_vals_full[i]) - Decimal(mean) )**5 ) * Decimal(gauss_cnt_full[i])
        j.append(k)
    return(j)

def sixth_moment(x_vals_full, gauss_cnt_full):
    l=[]
    for i in range(0, len(x_vals_full)):
        m = ( ( Decimal(x_vals_full[i]) - Decimal(mean) )**6 ) * Decimal(gauss_cnt_full[i])
        l.append(m)
    return(l)

def chisquare(gauss_cnt, exp_freq):
    test_statistic=0
    for observed, expected in zip(gauss_cnt, exp_freq):
        test_statistic+=((Decimal(observed)-Decimal(expected))**2)/Decimal(expected)
    return(test_statistic)

def k_squared(x_vals_full, gauss_cnt_full):
    n = Decimal(len(gauss_cnt_full))
    skew = (Decimal(np.sum(third_moment(x_vals_full, gauss_cnt_full)))/Decimal(sum(gauss_cnt_full)))/(Decimal(std_dev**3))
    y = Decimal(skew)*  Decimal(math.sqrt(((n+1)*(n+3))/(6*(n-2))))
    beta_2 = Decimal( 3*(n**2 + 27*n - 70)*(n+1)*(n+3) ) / ( (n-2)*(n+5)*(n+7)*(n+9) )
    W2 = Decimal(-1+ math.sqrt((2*beta_2) -1))
    delta = Decimal(1/(math.sqrt((math.log(math.sqrt(W2))))))
    alpha = Decimal(math.sqrt( 2 / (W2-1) ))
    skew_stat = Decimal(delta*Decimal(math.log(y/alpha  + Decimal(math.sqrt((y/alpha)**2 +1)))))
    
    kurt = (Decimal(np.sum(forth_moment(x_vals_full, gauss_cnt_full)))/Decimal(sum(gauss_cnt_full)))/(Decimal(variance**2)) -Decimal(3)
    E = Decimal((3*(n-1))/(n+1))
    var = Decimal( 24*n*(n-2)*(n-3)) / ( (n+3)*(n+5)*(n+1)*(n+1) )
    std_moment = Decimal( (6*(n**2 - 5*n +2)/((n+7)*(n+9)) ) * Decimal(math.sqrt( (6*(n+3)*(n+5)) / (n*(n-2)*(n-3))) ))
    A = Decimal(6 + (8/float(std_moment))*( (2/float(std_moment))+ math.sqrt(1+ (4/float(std_moment))) ))
    kurt_stat = Decimal( (1-(2/(9*A))) - ( (1- (2/A)) / (1+ (((kurt)-(-6/(n+1)))/Decimal(math.sqrt(var)) )*Decimal(math.sqrt(2/(A-4))) )  )**Decimal(1/3) ) / Decimal(math.sqrt(2/(9*A)))
    chi2 = skew_stat**2 + kurt_stat**2
    return(skew_stat,kurt_stat,chi2)

def expected_freq(gauss_cnt_full,x_vals_full):
    freq = []
    for i in range(0,len(x_vals_full)):
        freq.append(Decimal(sum(gauss_cnt_full)) * Decimal(np.exp(-(Decimal(x_vals_full[i])-Decimal(target_mean))**Decimal(2)/(Decimal(2)*(Decimal(target_sigma)**Decimal(2))))) / Decimal(np.sqrt(Decimal(2)*Decimal(np.pi)*(Decimal(target_sigma)**Decimal(2)))))
    return(freq)
	
def in_range(val,min,max):
	if min <= val <= max:
		return(1)
	else:
		return(0)

if __name__ == '__main__':

    print("//Discrete Gaussian Sampling Test Results:")
	
    #Computation
    variance = Decimal(np.sum(second_moment(x_vals_full, gauss_cnt_full)))/Decimal(sum(gauss_cnt_full))
    std_dev = Decimal(np.sqrt(np.sum(second_moment(x_vals_full, gauss_cnt_full))/Decimal(sum(gauss_cnt_full))))
    tail_cut = Decimal((Decimal(len(x_vals_full))/Decimal(2))/std_dev)
    skew = ( (np.sqrt(sum(gauss_cnt_full)*(sum(gauss_cnt_full)-1))) / (sum(gauss_cnt_full)-2) )*   (Decimal(np.sum(third_moment(x_vals_full, gauss_cnt_full)))/Decimal(sum(gauss_cnt_full)))/(Decimal(std_dev**3))
    kurt = (Decimal(np.sum(forth_moment(x_vals_full, gauss_cnt_full)))/Decimal(sum(gauss_cnt_full)))/(Decimal(variance**2))
    hyper_skew = (Decimal(np.sum(fifth_moment(x_vals_full, gauss_cnt_full)))/Decimal(sum(gauss_cnt_full)))/(Decimal(std_dev**5))
    hyper_kurt = (Decimal(np.sum(sixth_moment(x_vals_full, gauss_cnt_full)))/Decimal(sum(gauss_cnt_full)))/(Decimal(variance**3))
    jarque_bera = Decimal(len(gauss_cnt_full))*Decimal( (skew**2 /6) + ((kurt-3)**2 /24) )
    chi_square = chisquare(gauss_cnt, exp_freq)
    k2 = k_squared(x_vals_full, gauss_cnt_full)[2]
    expected_freq_float = [float(i) for i in expected_freq(gauss_cnt_full_float,x_vals_full_int)]
    
    print("//Target Sigma: ",target_sigma,"--","Sampler: ", "data path" ,"--","Sampler Size: ",sum(gauss_cnt_full))
    print()
    print("(1) Sample Mean:                ", mean)
    print("    Standard Error Of The Mean: ", np.absolute(Decimal(std_dev)/Decimal(np.sqrt(sum(gauss_cnt_full)))))
    print("    C.I. Of The Sample Mean =   ",mean,"+/-",(Decimal(3.29)*np.absolute(Decimal(std_dev)/Decimal(np.sqrt(sum(gauss_cnt_full))))), "with 99.9% confidence")
    print("    ","**Accept**" if in_range(target_mean,mean-(Decimal(3.29)*np.absolute(Decimal(std_dev)/Decimal(np.sqrt(sum(gauss_cnt_full))))), mean+(Decimal(3.29)*np.absolute(Decimal(std_dev)/Decimal(np.sqrt(sum(gauss_cnt_full)))))) == 1 else "**Reject**", "Null Hypothesis For Sample Mean With 99.9% Confidence")
    print()
    print("(2) Sample Standard Deviation:                ", std_dev)
    print("    Standard Error Of The Standard Deviation: ", np.absolute( (std_dev)/(np.sqrt(2*(sum(gauss_cnt_full)-1))) ) )
    print("    C.I. Of The Sample Standard Deviation =   ",std_dev,"+/-", ( Decimal(3.29)*Decimal(np.absolute( Decimal(std_dev)/Decimal(np.sqrt(2*(sum(gauss_cnt_full)-1)))) )  ), "with 99.9% confidence")
    print("    ","**Accept**" if in_range(target_sigma,std_dev-( Decimal(3.29)*Decimal(np.absolute( Decimal(std_dev)/Decimal(np.sqrt(2*(sum(gauss_cnt_full)-1)))) )  ), std_dev+(( Decimal(3.29)*Decimal(np.absolute( Decimal(std_dev)/Decimal(np.sqrt(2*(sum(gauss_cnt_full)-1)))) )  )) ) == 1 else "**Reject**", "Null Hypothesis For Sample Mean With 99.9% Confidence")
    print()
    print("(3) Sample Tail-Cut Parameter (Tau): ", tail_cut)
    print("    Distance From Target Tail-Cut:   ", np.absolute((Decimal(target_tail_cut)-Decimal(tail_cut))))
    print()
    print("(4) Sample Skewness:                       ", skew)
    print("    Standard Error Of The Sample Skewness: ", np.sqrt( (6*sum(gauss_cnt_full)*(sum(gauss_cnt_full)-1)) / ((sum(gauss_cnt_full)-2)*(sum(gauss_cnt_full)+1)*(sum(gauss_cnt_full)+3)) ) )
    print()
    print("(5) Sample Excess Kurtosis:                ", kurt -3)
    print("    Standard Error Of The Sample Kurtosis: ", 2*( np.sqrt( (6*sum(gauss_cnt_full)*(sum(gauss_cnt_full)-1)) / ((sum(gauss_cnt_full)-2)*(sum(gauss_cnt_full)+1)*(sum(gauss_cnt_full)+3)) ) )*np.sqrt( (sum(gauss_cnt_full)**2 -1)/((sum(gauss_cnt_full)-3)*(sum(gauss_cnt_full)+5)) ) )
    print()
    print("(6) Sample Hyperskewness:        ", hyper_skew)
    print()
    print("(7) Sample Excess Hyperkurtosis: ", hyper_kurt -15)
    print()
    print("(8) Jarque-Bera Test For Normality (test statistic, p-value):       ", jarque_bera, sp.stats.chisqprob(float(jarque_bera),2))
    print()
    print("(9) D Agostino-Pearson K**2 Omnibus Test (test statistic, p-value): ", k2, sp.stats.chisqprob(float(k2),2))
    print()
    print("(10) Histogram and Quantile-Quantile (Q-Q) plots")
    print()
    print("NOTE: Expected values for 5th and 6th moments found in http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.409.5634&rep=rep1&type=pdf")
    print()
    print("NOTE: Anderson Darling test, Shapiro-Wilk W-test, and chi-sqaure tests do not perform well for sample (bin) sizes ~> 4000")
    print()
    print("NOTE: Testing skewness/kurtosis, with example a z-test, is unreliable for sample sizes greater than 300 (http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3591587/)")

    r_squared = '$R^2=%.20f$'%Decimal(sp.stats.linregress(gauss_cnt_full_float,expected_freq_float)[2]**2)
    x=x_vals
    y1=gauss_cnt
    y2=exp_freq

    pl.bar(x, y1, width=1.25, color='blue', edgecolor='none', label='Gauss Samples')
    pl.plot(x, y2, '-r', label='Gauss Expected')
    pl.legend(loc='upper right')
    plt.title("Gaussian Samples vs Expected")
    plot1 = plt.plot()

    pp_x = sm.ProbPlot(np.array(gauss_cnt_full), fit=True)
    pp_y = sm.ProbPlot(np.array(expected_freq(gauss_cnt_full,x_vals_full)), fit=True)
    fig = pp_x.qqplot(xlabel='Gauss Samples', ylabel='Gauss Expected',line='45', other=pp_y)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 2.5, r_squared, fontsize=14,
        verticalalignment='top', bbox=props)
    plt.title("Gaussian Q-Q plot")
    plt.show()
