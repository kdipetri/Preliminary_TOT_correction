import ROOT
from array import array
import os, sys, re

ROOT.gROOT.SetBatch(ROOT.kTRUE)
ROOT.gStyle.SetLabelFont(42,"xyz")
ROOT.gStyle.SetLabelSize(0.05,"xyz")
#ROOT.gStyle.SetTitleFont(42)
ROOT.gStyle.SetTitleFont(42,"xyz")
ROOT.gStyle.SetTitleFont(42,"t")
#ROOT.gStyle.SetTitleSize(0.05)
ROOT.gStyle.SetTitleSize(0.06,"xyz")
ROOT.gStyle.SetTitleSize(0.06,"t")
ROOT.gStyle.SetPadBottomMargin(0.14)
ROOT.gStyle.SetPadLeftMargin(0.14)
ROOT.gStyle.SetTitleOffset(1,'y')
ROOT.gStyle.SetLegendTextSize(0.035)
ROOT.gStyle.SetGridStyle(3)
ROOT.gStyle.SetGridColor(14)
ROOT.gStyle.SetOptFit(1)
one = ROOT.TColor(2001,0.906,0.153,0.094)
two = ROOT.TColor(2002,0.906,0.533,0.094)
three = ROOT.TColor(2003,0.086,0.404,0.576)
four =ROOT.TColor(2004,0.071,0.694,0.18)
five =ROOT.TColor(2005,0.388,0.098,0.608)
six=ROOT.TColor(2006,0.906,0.878,0.094)
colors = [1,2001,2002,2003,2004,2005,2006,6,2,3,4,6,7,5,1,8,9,29,38,46,1,2001,2002,2003,2004,2005,2006]

# default params
charge_thresh = 20 #fC
photek_res = 15 #ps

def get_min_amp(run):
        minAmp =15
        if run==151172 or run==151173: minAmp = 40
        if run>151244 and run <=151250: minAmp = 40
        if run>=2023 and run <=2025: minAmp=40
        if run>=2026 and run <=2028: minAmp=70
        if run==2022: minAmp=30
        if run >= 151357 and run<=151374: minAmp=10
        if run==151647 or run==151739: minAmp=30
        if run>=151639 and run<=151653: minAmp=45
        if run>=151818 and run<=151822: minAmp=45
        if run>=151866 and run<=151870: minAmp=45
        if run>=151848 and run<=151854: minAmp=45
        if run>=151835 and run<=151838: minAmp=45
        if run>=151649 and run<=151653: minAmp=45
        if run>=151168 and run<=151169: minAmp=45
        if run>=151896 and run<=151902: minAmp=45
        if run>=151994 and run<=151997: minAmp=45

        return minAmp

def get_mean_response_channel(tree,ch,run=-1):
    
    hist = ROOT.TH1D("h","",50,0,400)
    minAmp = get_min_amp(run)
    tree.Project("h","amp[%i]"%ch,"amp[%i]>%f&&amp[3]>10"%(ch,minAmp))

    #fitter = lg.LanGausFit()
    #f1 = fitter.fit(hist)
    # fuck that

    f1 = ROOT.TF1("f1","landau",0,150)
    hist.Fit(f1)

    if run>0:
        c = ROOT.TCanvas()
        hist.SetTitle(";Amplitude [mV];Events")
        hist.Draw()
        f1.Draw("same")
        if ch==2: c.Print("plots/runs/Run%i_amp.pdf"%run)
        else: c.Print("plots/runs/Run%i_ch%i_amp.pdf"%(run,ch))
        #c.Print("plots/runs/Run%i_amp.root"%run)
        return f1.GetParameter(1),f1.GetParError(1)

def get_time_res_channel(tree,ch,run=-1):
        #(70,-3.3e-9,-1.6e-9)
        mint = 3.8e-9
        maxt = 6.8e-9

        hist = ROOT.TH1D("h","",70,mint,maxt)

        photek_thresh = 15
        photek_max = 200

        tree.Project("h","LP2_15[%i]-LP2_20[3]"%ch,"amp[%i]>15 && amp[3]>%i && amp[3]<%i && LP2_20[3]!=0 && LP2_20[%i]!=0"%(ch,photek_thresh,photek_max,ch))
        f1 = ROOT.TF1("f1","gaus",mint,maxt)

        hist.Fit(f1)
        if run>0:
                c = ROOT.TCanvas()
                hist.Draw()
                f1.Draw("same")
                if ch==2: c.Print("plots/runs/Run%i_time.pdf"%run)
                else: c.Print("plots/runs/Run%i_ch%i_time.pdf"%(run,ch))

        print('Run Number %d,  %f, %f ' %(run,1e12*f1.GetParameter(2),1e12*f1.GetParError(2)))
        return 1e12*f1.GetParameter(2),1e12*f1.GetParError(2)


    
def get_time_walk(tree,ch,run=-1):
        #(70,-3.3e-9,-1.6e-9)
        mint = 3.8e-9
        maxt = 6.8e-9

        mintot = 0.1e-9
        maxtot = 8.1e-9

        hist = ROOT.TH2D("h",";TOT [s];t_{0}-t_{ref} [s]",50,mintot,maxtot,70,mint,maxt)
        #hist = ROOT.TH1D("h","",70,mint,maxt)

        photek_thresh = 15
        photek_max = 200

        tree.Project("h","t0_30[%i]-LP2_20[3]:tot_30[%i]"%(ch,ch),"amp[%i]>15 && amp[3]>%i && amp[3]<%i && LP2_20[3]!=0 && t0_30[%i]!=0"%(ch,photek_thresh,photek_max,ch),"COLZ")
        #f1 = ROOT.TF1("f1","gaus",5.8e-9,6.8e-9)

        #hist.Fit(f1)
        if run>0:
                c = ROOT.TCanvas()
                hist.Draw("COLZ")
                if ch==2: c.Print("plots/runs/Run%i_t0_v_tot.pdf"%run)
                else: c.Print("plots/runs/Run%i_ch%i_t0_v_tot.pdf"%(run,ch))

        
        profile = hist.ProfileX()
        spread  = hist.ProfileX("prof",1,-1,"s")
        
        f1 = ROOT.TF1("f1","pol3",mintot,maxtot)
        hist.Fit(f1)

        if run > 0 : 
            profile.Draw()
            profile.SetMarkerStyle(20);
            profile.SetMarkerColor(ROOT.kBlack);
            profile.SetLineColor(ROOT.kBlack);
            profile.SetLineWidth(3);
            profile.SetFillColorAlpha(ROOT.kBlack,0.35);
            profile.Draw("e2sames");

            profile.GetXaxis().SetRangeUser(mintot,maxtot) 
            spread .GetXaxis().SetRangeUser(mintot,maxtot)
            profile.GetYaxis().SetRangeUser(mint,maxt) 
            spread .GetYaxis().SetRangeUser(mint,maxt)
            profile.GetYaxis().SetTitle("t_{0}-t_{ref} [s]") 

            f1.Draw("same")

            if ch==2: c.Print("plots/runs/Run%i_timewalk.pdf"%run)
            else: c.Print("plots/runs/Run%i_ch%i_timewalk.pdf"%(run,ch))

        # fit should have four parameters
        print('Run Number {}, channel {}'.format(run,ch))
        print('p0 : {} , {} '.format(f1.GetParameter(0),f1.GetParError(0)))
        print('p1 : {} , {} '.format(f1.GetParameter(1),f1.GetParError(1)))
        print('p2 : {} , {} '.format(f1.GetParameter(2),f1.GetParError(2)))
        print('p3 : {} , {} '.format(f1.GetParameter(3),f1.GetParError(3)))
        #return 1e12*f1.GetParameter(2),1e12*f1.GetParError(2)

        params = (f1.GetParameter(0), f1.GetParameter(1), f1.GetParameter(2), f1.GetParameter(3))
        return params 

         
def get_time_res_tot(tree,ch,fit_params,run=-1):
        #(70,-3.3e-9,-1.6e-9)
        mint = -2e-9
        maxt =  2e-9

        mintot = 0.1e-9
        maxtot = 8.1e-9


        photek_thresh = 15
        photek_max = 200

        (x0,x1,x2,x3) = fit_params
        f_TOT = "{:e} + {:e}*tot_30[{}] + {:e}*tot_30[{}]**2 + {:e}*tot_30[{}]**3 - t0_30[{}] + LP2_20[3]".format(x0,x1,ch,x2,ch,x3,ch,ch) 
        print(f_TOT)

        # 2D hist to validate time walk correction

        hist2D = ROOT.TH2D("h2D",";TOT [s];t_{0}^{Corr} [s]",50,mintot,maxtot,70,mint,maxt)

        tree.Project("h2D","%s:tot_30[%i]"%(f_TOT,ch),"amp[%i]>15 && amp[3]>%i && amp[3]<%i && LP2_20[3]!=0 && t0_30[%i]!=0"%(ch,photek_thresh,photek_max,ch),"COLZ")
        if run>0:
                c = ROOT.TCanvas()
                hist2D.Draw("COLZ")
                if ch==2: c.Print("plots/runs/Run%i_t0Corr_v_tot.pdf"%run)
                else: c.Print("plots/runs/Run%i_ch%i_t0Corr_v_tot.pdf"%(run,ch))

        # 1D hist for final corrected time resolution
        hist = ROOT.TH1D("h",";t_{0}^{Corr} [s]",70,mint,maxt)
        
        tree.Project("h","%s"%(f_TOT),"amp[%i]>15 && amp[3]>%i && amp[3]<%i && LP2_20[3]!=0 && t0_30[%i]!=0"%(ch,photek_thresh,photek_max,ch))

        f1 = ROOT.TF1("f1","gaus",mint,maxt)

        hist.Fit(f1)

        if run>0:
                hist.Draw()
                f1.Draw("same")
                if ch==2: c.Print("plots/runs/Run%i_t0Corr.pdf"%run)
                else: c.Print("plots/runs/Run%i_ch%i_t0Corr.pdf"%(run,ch))
        return


run=1
ch_amp=2
ch_dis=1

# inputs
f = ROOT.TFile.Open("test_1.root")
t = f.Get("pulse")

# validations
get_mean_response_channel(t,ch_amp,run) 
get_mean_response_channel(t,ch_dis,run) 

get_time_res_channel(t,ch_amp,run)
get_time_res_channel(t,ch_dis,run)

amp_params = get_time_walk(t,ch_amp,run)
dis_params = get_time_walk(t,ch_dis,run)

get_time_res_tot(t,ch_amp,amp_params,run)
get_time_res_tot(t,ch_dis,dis_params,run)
